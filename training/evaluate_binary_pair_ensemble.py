#!/usr/bin/env python3
"""Evaluate ensemble of valence+arousal EEGNet models and mapped quadrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import train_test_split_cross_subject, train_test_split_cross_trial

from inference.model_arch import EEGNet
from inference.torcheeg_patch import apply_lmdb_patch

VALENCE_THRESHOLD = 3.0
AROUSAL_THRESHOLD = 3.0
NUM_CHANNELS = 14
CHUNK_SIZE = 256
STEP_SIZE = 128
QUADRANT_NAMES = ['Happy', 'Stressed', 'Depressed', 'Calm']


def quadrant_from_binary(high_valence: bool, high_arousal: bool) -> int:
    if high_valence and high_arousal:
        return 0
    if (not high_valence) and high_arousal:
        return 1
    if (not high_valence) and (not high_arousal):
        return 2
    return 3


def map_quadrant_label(y: Dict, **kwargs) -> Dict[str, int]:
    _ = kwargs
    v = y['valence'] > VALENCE_THRESHOLD
    a = y['arousal'] > AROUSAL_THRESHOLD
    return {'y': quadrant_from_binary(v, a)}


class EvalDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        info = base_dataset.info
        self.y_valence = (info['valence'].values > VALENCE_THRESHOLD).astype(np.int64)
        self.y_arousal = (info['arousal'].values > AROUSAL_THRESHOLD).astype(np.int64)
        self.y_quad = np.asarray(
            [
                quadrant_from_binary(bool(v), bool(a))
                for v, a in zip(self.y_valence, self.y_arousal)
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, _ = self.base[index]
        arr = x.numpy().astype(np.float32)
        return (
            arr,
            int(self.y_valence[index]),
            int(self.y_arousal[index]),
            int(self.y_quad[index]),
        )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }


def confusion_payload(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]
) -> Dict[str, List[List[float]]]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    return {
        'counts': cm.astype(int).tolist(),
        'row_normalized': cm_norm.tolist(),
    }


def load_from_report(report_path: Path, metric: str, top_k: int) -> List[Tuple[Path, Path, str, float]]:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    with report_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    folds = payload.get('folds', [])
    if not folds:
        raise ValueError(f'No folds found in report: {report_path}')
    ranked = sorted(
        folds,
        key=lambda row: float(row['metrics'][metric]),
        reverse=True,
    )
    selected = ranked[: max(1, top_k)]
    out: List[Tuple[Path, Path, str, float]] = []
    for row in selected:
        model_path = Path(str(row['checkpoint_path']))
        scaler_path = Path(str(row['scaler_path']))
        out.append((model_path, scaler_path, str(row['fold_id']), float(row['metrics'][metric])))
    return out


def parse_path_list(csv_value: str) -> List[Path]:
    values = [v.strip() for v in csv_value.split(',') if v.strip()]
    if not values:
        raise ValueError('Expected a non-empty comma-separated path list.')
    return [Path(v) for v in values]


def build_model_bundle(model_paths: Sequence[Path], scaler_paths: Sequence[Path], device: torch.device):
    if len(model_paths) != len(scaler_paths):
        raise ValueError('Model path count must match scaler path count.')
    bundle = []
    for model_path, scaler_path in zip(model_paths, scaler_paths):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not scaler_path.is_file():
            raise FileNotFoundError(scaler_path)
        model = EEGNet(n_channels=NUM_CHANNELS, n_samples=CHUNK_SIZE, n_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        scaler = joblib.load(scaler_path)
        bundle.append((model_path, scaler_path, model, scaler))
    return bundle


def ensemble_binary_probs(x_batch: np.ndarray, model_bundle, device: torch.device) -> np.ndarray:
    probs_list: List[np.ndarray] = []
    for _, _, model, scaler in model_bundle:
        bsz, n_ch, n_t = x_batch.shape
        scaled = scaler.transform(x_batch.reshape(bsz, n_ch * n_t)).reshape(x_batch.shape)
        x = torch.from_numpy(scaled.astype(np.float32)).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        probs_list.append(probs)
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    p.add_argument('--io-path', default=f'torcheeg_cache_raw_c{CHUNK_SIZE}_s{STEP_SIZE}')
    p.add_argument('--split', choices=['cross_trial', 'cross_subject'], default='cross_subject')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    p.add_argument('--valence-report', type=Path, default=None)
    p.add_argument('--arousal-report', type=Path, default=None)
    p.add_argument('--metric', default='balanced_accuracy', choices=['balanced_accuracy', 'accuracy', 'macro_f1'])
    p.add_argument('--top-k', type=int, default=3)

    p.add_argument('--valence-models', type=str, default=None)
    p.add_argument('--valence-scalers', type=str, default=None)
    p.add_argument('--arousal-models', type=str, default=None)
    p.add_argument('--arousal-scalers', type=str, default=None)
    p.add_argument('--report-out', type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mat_path.is_file():
        raise FileNotFoundError(args.mat_path)

    apply_lmdb_patch()
    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)

    if args.valence_report and args.arousal_report:
        v_selected = load_from_report(args.valence_report, metric=args.metric, top_k=args.top_k)
        a_selected = load_from_report(args.arousal_report, metric=args.metric, top_k=args.top_k)
        v_models = [row[0] for row in v_selected]
        v_scalers = [row[1] for row in v_selected]
        a_models = [row[0] for row in a_selected]
        a_scalers = [row[1] for row in a_selected]
    else:
        if not all([args.valence_models, args.valence_scalers, args.arousal_models, args.arousal_scalers]):
            raise ValueError(
                'Either provide both reports (--valence-report and --arousal-report) '
                'or all explicit model/scaler CSV lists.'
            )
        v_models = parse_path_list(args.valence_models)
        v_scalers = parse_path_list(args.valence_scalers)
        a_models = parse_path_list(args.arousal_models)
        a_scalers = parse_path_list(args.arousal_scalers)
        v_selected = [(m, s, f'manual_{i+1}', float('nan')) for i, (m, s) in enumerate(zip(v_models, v_scalers))]
        a_selected = [(m, s, f'manual_{i+1}', float('nan')) for i, (m, s) in enumerate(zip(a_models, a_scalers))]

    dataset = DREAMERDataset(
        mat_path=str(args.mat_path),
        io_path=args.io_path,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_SIZE - STEP_SIZE,
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=map_quadrant_label,
    )
    if args.split == 'cross_trial':
        _, val_raw = train_test_split_cross_trial(
            dataset,
            test_size=args.test_size,
            shuffle=True,
            random_state=args.seed,
            split_path='artifacts/eval_split_cross_trial',
        )
    else:
        _, val_raw = train_test_split_cross_subject(
            dataset,
            test_size=args.test_size,
            shuffle=True,
            random_state=args.seed,
            split_path='artifacts/eval_split_cross_subject',
        )

    eval_ds = EvalDataset(val_raw)
    loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=0)

    valence_bundle = build_model_bundle(v_models, v_scalers, device)
    arousal_bundle = build_model_bundle(a_models, a_scalers, device)

    yv_t: List[int] = []
    yv_p: List[int] = []
    ya_t: List[int] = []
    ya_p: List[int] = []
    yq_t: List[int] = []
    yq_p: List[int] = []

    for x_np, yv, ya, yq in loader:
        x_batch = x_np.numpy()
        v_probs = ensemble_binary_probs(x_batch, valence_bundle, device=device)
        a_probs = ensemble_binary_probs(x_batch, arousal_bundle, device=device)
        pv = v_probs.argmax(1).astype(np.int64)
        pa = a_probs.argmax(1).astype(np.int64)
        yv_np = yv.numpy()
        ya_np = ya.numpy()
        yq_np = yq.numpy()
        pq = np.asarray(
            [quadrant_from_binary(bool(v), bool(a)) for v, a in zip(pv, pa)],
            dtype=np.int64,
        )
        yv_t.extend(yv_np.tolist())
        yv_p.extend(pv.tolist())
        ya_t.extend(ya_np.tolist())
        ya_p.extend(pa.tolist())
        yq_t.extend(yq_np.tolist())
        yq_p.extend(pq.tolist())

    yv_true = np.asarray(yv_t)
    yv_pred = np.asarray(yv_p)
    ya_true = np.asarray(ya_t)
    ya_pred = np.asarray(ya_p)
    yq_true = np.asarray(yq_t)
    yq_pred = np.asarray(yq_p)

    report = {
        'ensemble': {
            'metric_for_topk_selection': args.metric,
            'top_k_per_task': args.top_k,
            'valence_members': [
                {
                    'fold_id': fold_id,
                    'metric_value': metric_value,
                    'checkpoint_path': str(model_path),
                    'scaler_path': str(scaler_path),
                }
                for model_path, scaler_path, fold_id, metric_value in v_selected
            ],
            'arousal_members': [
                {
                    'fold_id': fold_id,
                    'metric_value': metric_value,
                    'checkpoint_path': str(model_path),
                    'scaler_path': str(scaler_path),
                }
                for model_path, scaler_path, fold_id, metric_value in a_selected
            ],
        },
        'valence': metric_dict(yv_true, yv_pred),
        'arousal': metric_dict(ya_true, ya_pred),
        'quadrant_from_binary': metric_dict(yq_true, yq_pred),
        'confusion_matrices': {
            'valence': confusion_payload(yv_true, yv_pred, labels=[0, 1]),
            'arousal': confusion_payload(ya_true, ya_pred, labels=[0, 1]),
            'quadrant_from_binary': confusion_payload(
                yq_true, yq_pred, labels=[0, 1, 2, 3]
            ),
        },
        'quadrant_labels': QUADRANT_NAMES,
        'samples': len(yq_t),
    }

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'Wrote report: {args.report_out}')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
