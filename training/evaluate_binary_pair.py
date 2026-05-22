#!/usr/bin/env python3
"""Evaluate valence+arousal binary EEGNet checkpoints and mapped quadrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import train_test_split_cross_subject, train_test_split_cross_trial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    def __init__(self, base_dataset, scaler):
        self.base = base_dataset
        self.scaler = scaler
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
        scaled = self.scaler.transform(arr.reshape(1, -1)).reshape(arr.shape)
        return (
            torch.from_numpy(scaled),
            int(self.y_valence[index]),
            int(self.y_arousal[index]),
            int(self.y_quad[index]),
        )


def load_eegnet(path: Path, device: torch.device) -> EEGNet:
    model = EEGNet(n_channels=NUM_CHANNELS, n_samples=CHUNK_SIZE, n_classes=2).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--valence-model', type=Path, required=True)
    p.add_argument('--arousal-model', type=Path, required=True)
    p.add_argument('--scaler', type=Path, required=True)
    p.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    p.add_argument('--io-path', default=f'torcheeg_cache_raw_c{CHUNK_SIZE}_s{STEP_SIZE}')
    p.add_argument('--split', choices=['cross_trial', 'cross_subject'], default='cross_subject')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.valence_model, args.arousal_model, args.scaler, args.mat_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    apply_lmdb_patch()
    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)

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

    scaler = joblib.load(args.scaler)
    eval_ds = EvalDataset(val_raw, scaler)
    loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=0)

    model_v = load_eegnet(args.valence_model, device)
    model_a = load_eegnet(args.arousal_model, device)

    yv_t: List[int] = []
    yv_p: List[int] = []
    ya_t: List[int] = []
    ya_p: List[int] = []
    yq_t: List[int] = []
    yq_p: List[int] = []

    with torch.no_grad():
        for x, yv, ya, yq in loader:
            x = x.to(device)
            pv = model_v(x).argmax(1).cpu().numpy()
            pa = model_a(x).argmax(1).cpu().numpy()
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

    report = {
        'valence': metric_dict(np.asarray(yv_t), np.asarray(yv_p)),
        'arousal': metric_dict(np.asarray(ya_t), np.asarray(ya_p)),
        'quadrant_from_binary': metric_dict(np.asarray(yq_t), np.asarray(yq_p)),
        'quadrant_labels': QUADRANT_NAMES,
        'samples': len(yq_t),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
