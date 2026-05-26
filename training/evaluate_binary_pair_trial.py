#!/usr/bin/env python3
"""Trial-level evaluation for valence+arousal EEGNet model pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import (
    train_test_split_cross_subject,
    train_test_split_cross_trial,
)

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
        info = base_dataset.info.reset_index(drop=True)
        required_cols = {'subject_id', 'trial_id', 'valence', 'arousal'}
        missing = required_cols.difference(info.columns)
        if missing:
            raise ValueError(
                'Validation metadata missing required columns: '
                f'{sorted(missing)}'
            )
        self.subject_id = info['subject_id'].to_numpy(dtype=np.int64)
        self.trial_id = info['trial_id'].to_numpy(dtype=np.int64)
        self.y_valence = (info['valence'].to_numpy() > VALENCE_THRESHOLD).astype(
            np.int64
        )
        self.y_arousal = (info['arousal'].to_numpy() > AROUSAL_THRESHOLD).astype(
            np.int64
        )
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
            int(self.subject_id[index]),
            int(self.trial_id[index]),
        )


def metric_dict(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': cm.astype(int).tolist(),
    }


def load_eegnet(path: Path, device: torch.device) -> EEGNet:
    model = EEGNet(
        n_channels=NUM_CHANNELS, n_samples=CHUNK_SIZE, n_classes=2
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def resolve_scaler_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.valence_scaler and args.arousal_scaler:
        return args.valence_scaler, args.arousal_scaler
    if args.scaler:
        return args.scaler, args.scaler
    raise ValueError(
        'Provide either --scaler for both tasks, or both '
        '--valence-scaler and --arousal-scaler.'
    )


def vote_with_tie_break(
    labels: np.ndarray, num_classes: int, tie_scores: Optional[np.ndarray] = None
) -> int:
    counts = np.bincount(labels, minlength=num_classes)
    winners = np.flatnonzero(counts == counts.max())
    if winners.size == 1:
        return int(winners[0])
    if tie_scores is not None:
        return int(winners[int(np.argmax(tie_scores[winners]))])
    return int(winners[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--valence-model', type=Path, required=True)
    p.add_argument('--arousal-model', type=Path, required=True)
    p.add_argument('--scaler', type=Path, default=None)
    p.add_argument('--valence-scaler', type=Path, default=None)
    p.add_argument('--arousal-scaler', type=Path, default=None)
    p.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    p.add_argument(
        '--io-path', default=f'torcheeg_cache_raw_c{CHUNK_SIZE}_s{STEP_SIZE}'
    )
    p.add_argument(
        '--split', choices=['cross_trial', 'cross_subject'], default='cross_subject'
    )
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument(
        '--aggregation', choices=['vote', 'mean_prob'], default='mean_prob'
    )
    p.add_argument(
        '--report-out',
        type=Path,
        default=None,
        help='Optional JSON output path. Prints to stdout if omitted.',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    valence_scaler_path, arousal_scaler_path = resolve_scaler_paths(args)
    for path in [
        args.valence_model,
        args.arousal_model,
        valence_scaler_path,
        arousal_scaler_path,
        args.mat_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    apply_lmdb_patch()
    device_name = (
        'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    )
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

    eval_ds = EvalDataset(val_raw)
    loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model_v = load_eegnet(args.valence_model, device)
    model_a = load_eegnet(args.arousal_model, device)
    scaler_v = joblib.load(valence_scaler_path)
    scaler_a = joblib.load(arousal_scaler_path)

    sid_all: List[int] = []
    tid_all: List[int] = []
    yv_true_all: List[int] = []
    ya_true_all: List[int] = []
    yq_true_all: List[int] = []
    yv_pred_all: List[int] = []
    ya_pred_all: List[int] = []
    yq_pred_all: List[int] = []
    v_probs_all: List[np.ndarray] = []
    a_probs_all: List[np.ndarray] = []
    q_probs_all: List[np.ndarray] = []

    with torch.no_grad():
        for x_np, yv, ya, yq, sid, tid in loader:
            x_batch = x_np.numpy().astype(np.float32)
            bsz, n_ch, n_t = x_batch.shape
            flat = x_batch.reshape(bsz, n_ch * n_t)

            x_v = scaler_v.transform(flat).reshape(x_batch.shape).astype(np.float32)
            x_a = scaler_a.transform(flat).reshape(x_batch.shape).astype(np.float32)

            p_v = (
                F.softmax(model_v(torch.from_numpy(x_v).to(device)), dim=1)
                .cpu()
                .numpy()
            )
            p_a = (
                F.softmax(model_a(torch.from_numpy(x_a).to(device)), dim=1)
                .cpu()
                .numpy()
            )
            pred_v = p_v.argmax(1).astype(np.int64)
            pred_a = p_a.argmax(1).astype(np.int64)
            pred_q = np.asarray(
                [
                    quadrant_from_binary(bool(v == 1), bool(a == 1))
                    for v, a in zip(pred_v, pred_a)
                ],
                dtype=np.int64,
            )
            p_q = np.stack(
                [
                    p_v[:, 1] * p_a[:, 1],  # Happy
                    p_v[:, 0] * p_a[:, 1],  # Stressed
                    p_v[:, 0] * p_a[:, 0],  # Depressed
                    p_v[:, 1] * p_a[:, 0],  # Calm
                ],
                axis=1,
            )

            sid_all.extend(sid.numpy().astype(np.int64).tolist())
            tid_all.extend(tid.numpy().astype(np.int64).tolist())
            yv_true_all.extend(yv.numpy().astype(np.int64).tolist())
            ya_true_all.extend(ya.numpy().astype(np.int64).tolist())
            yq_true_all.extend(yq.numpy().astype(np.int64).tolist())
            yv_pred_all.extend(pred_v.tolist())
            ya_pred_all.extend(pred_a.tolist())
            yq_pred_all.extend(pred_q.tolist())
            v_probs_all.append(p_v)
            a_probs_all.append(p_a)
            q_probs_all.append(p_q)

    yv_true = np.asarray(yv_true_all, dtype=np.int64)
    ya_true = np.asarray(ya_true_all, dtype=np.int64)
    yq_true = np.asarray(yq_true_all, dtype=np.int64)
    yv_pred = np.asarray(yv_pred_all, dtype=np.int64)
    ya_pred = np.asarray(ya_pred_all, dtype=np.int64)
    yq_pred = np.asarray(yq_pred_all, dtype=np.int64)
    sid_arr = np.asarray(sid_all, dtype=np.int64)
    tid_arr = np.asarray(tid_all, dtype=np.int64)
    v_probs = np.concatenate(v_probs_all, axis=0)
    a_probs = np.concatenate(a_probs_all, axis=0)
    q_probs = np.concatenate(q_probs_all, axis=0)

    trial_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (sid, tid) in enumerate(zip(sid_arr, tid_arr)):
        trial_groups[(int(sid), int(tid))].append(idx)

    yv_true_t: List[int] = []
    ya_true_t: List[int] = []
    yq_true_t: List[int] = []
    yv_pred_t: List[int] = []
    ya_pred_t: List[int] = []
    yq_pred_t: List[int] = []
    trial_sizes: List[int] = []

    for key in sorted(trial_groups.keys()):
        idxs = np.asarray(trial_groups[key], dtype=np.int64)
        trial_sizes.append(int(idxs.size))

        yv_true_t.append(vote_with_tie_break(yv_true[idxs], num_classes=2))
        ya_true_t.append(vote_with_tie_break(ya_true[idxs], num_classes=2))
        yq_true_t.append(vote_with_tie_break(yq_true[idxs], num_classes=4))

        if args.aggregation == 'mean_prob':
            yv_pred_t.append(int(np.argmax(v_probs[idxs].mean(axis=0))))
            ya_pred_t.append(int(np.argmax(a_probs[idxs].mean(axis=0))))
            yq_pred_t.append(int(np.argmax(q_probs[idxs].mean(axis=0))))
        else:
            yv_pred_t.append(
                vote_with_tie_break(
                    yv_pred[idxs], num_classes=2, tie_scores=v_probs[idxs].mean(axis=0)
                )
            )
            ya_pred_t.append(
                vote_with_tie_break(
                    ya_pred[idxs], num_classes=2, tie_scores=a_probs[idxs].mean(axis=0)
                )
            )
            yq_pred_t.append(
                vote_with_tie_break(
                    yq_pred[idxs], num_classes=4, tie_scores=q_probs[idxs].mean(axis=0)
                )
            )

    trial_size_arr = np.asarray(trial_sizes, dtype=np.int64)
    report = {
        'config': {
            'valence_model': str(args.valence_model),
            'arousal_model': str(args.arousal_model),
            'valence_scaler': str(valence_scaler_path),
            'arousal_scaler': str(arousal_scaler_path),
            'split': args.split,
            'seed': args.seed,
            'aggregation': args.aggregation,
            'grouping_key': 'subject_id+trial_id',
            'samples': int(yq_true.shape[0]),
            'trials': int(trial_size_arr.shape[0]),
        },
        'window_level': {
            'valence': metric_dict(yv_true, yv_pred, num_classes=2),
            'arousal': metric_dict(ya_true, ya_pred, num_classes=2),
            'quadrant_from_binary': metric_dict(yq_true, yq_pred, num_classes=4),
        },
        'trial_level': {
            'valence': metric_dict(
                np.asarray(yv_true_t, dtype=np.int64),
                np.asarray(yv_pred_t, dtype=np.int64),
                num_classes=2,
            ),
            'arousal': metric_dict(
                np.asarray(ya_true_t, dtype=np.int64),
                np.asarray(ya_pred_t, dtype=np.int64),
                num_classes=2,
            ),
            'quadrant_from_binary': metric_dict(
                np.asarray(yq_true_t, dtype=np.int64),
                np.asarray(yq_pred_t, dtype=np.int64),
                num_classes=4,
            ),
            'trial_size_summary': {
                'min': int(trial_size_arr.min()),
                'max': int(trial_size_arr.max()),
                'mean': float(trial_size_arr.mean()),
                'median': float(np.median(trial_size_arr)),
            },
        },
        'quadrant_labels': QUADRANT_NAMES,
    }

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Saved report to: {args.report_out}')
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
