#!/usr/bin/env python3
"""Production-style trainer for binary DREAMER EEGNet models.

This script trains either valence or arousal binary classifiers using raw
14x256 EEG windows (scaled by a train-only StandardScaler), with reproducible
splits, class weighting, early stopping, and JSON metrics export.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import (
    LeaveOneSubjectOut,
    train_test_split_cross_subject,
    train_test_split_cross_trial,
)
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_arch import EEGNet
from inference.torcheeg_patch import apply_lmdb_patch

VALENCE_THRESHOLD = 3.0
AROUSAL_THRESHOLD = 3.0
NUM_CHANNELS = 14
CHUNK_SIZE = 256
DEFAULT_STEP_SIZE = 128
DEFAULT_TRAIN_CONFIG = {
    'task': 'both',
    'split': 'group_kfold_subject',
    'n_splits': 5,
    'epochs': 30,
    'batch_size': 64,
    'output_dir': 'artifacts_cv5',
    'device': 'auto',
}
PRESET_CONFIGS = {
    # Stable baseline; best first run when debugging performance.
    'baseline': {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'patience': 6,
        'loss': 'cross_entropy',
        'label_smoothing': 0.0,
        'focal_gamma': 2.0,
        'weighted_sampling': False,
        'train_noise_std': 0.0,
        'train_channel_drop_prob': 0.0,
        'train_time_mask_width': 0,
        'grad_clip_norm': 0.0,
    },
    # Better imbalance handling; strong default for most users.
    'balanced': {
        'lr': 8e-4,
        'weight_decay': 1e-4,
        'patience': 7,
        'loss': 'cross_entropy',
        'label_smoothing': 0.02,
        'focal_gamma': 2.0,
        'weighted_sampling': True,
        'train_noise_std': 0.0,
        'train_channel_drop_prob': 0.0,
        'train_time_mask_width': 0,
        'grad_clip_norm': 0.5,
    },
    # Moderate regularization and focal loss.
    'robust': {
        'lr': 5e-4,
        'weight_decay': 5e-4,
        'patience': 8,
        'loss': 'focal',
        'label_smoothing': 0.04,
        'focal_gamma': 2.0,
        'weighted_sampling': True,
        'train_noise_std': 0.005,
        'train_channel_drop_prob': 0.03,
        'train_time_mask_width': 8,
        'grad_clip_norm': 1.0,
    },
    # Heavy regularization; use only if robust overfits badly.
    'aggressive': {
        'lr': 3e-4,
        'weight_decay': 1e-3,
        'patience': 10,
        'loss': 'focal',
        'label_smoothing': 0.05,
        'focal_gamma': 2.5,
        'weighted_sampling': True,
        'train_noise_std': 0.01,
        'train_channel_drop_prob': 0.05,
        'train_time_mask_width': 12,
        'grad_clip_norm': 1.0,
    },
}


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BinaryLabelDataset(Dataset):
    """Wrap TorchEEG subset and emit scaled samples + one binary target."""

    def __init__(self, base_dataset, scaler: StandardScaler, task: str):
        self.base = base_dataset
        self.scaler = scaler
        self.task = task
        info = base_dataset.info
        if task == 'valence':
            self.labels = (info['valence'].values > VALENCE_THRESHOLD).astype(np.int64)
        elif task == 'arousal':
            self.labels = (info['arousal'].values > AROUSAL_THRESHOLD).astype(np.int64)
        else:
            raise ValueError(f'Unsupported task: {task}')

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.base[index]
        arr = x.numpy().astype(np.float32)
        scaled = self.scaler.transform(arr.reshape(1, -1)).reshape(arr.shape)
        x_t = torch.from_numpy(scaled)
        y_t = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x_t, y_t


class IndexedDataset(Dataset):
    """Index view over a TorchEEG dataset with aligned `info` metadata."""

    def __init__(self, base_dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.info = base_dataset.info.iloc[self.indices].reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        return self.base[int(self.indices[index])]


def fit_channel_scaler(dataset, batch_size: int) -> StandardScaler:
    scaler = StandardScaler()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x_batch, _ in tqdm(loader, desc='Fitting scaler (train only)'):
        bsz, n_ch, n_t = x_batch.shape
        scaler.partial_fit(x_batch.numpy().reshape(bsz, n_ch * n_t))
    return scaler


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / (2.0 * counts)
    return torch.tensor(inv_freq, dtype=torch.float32)


def class_distribution(labels: np.ndarray) -> Dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    total = float(max(counts.sum(), 1))
    return {
        'count_low': int(counts[0]),
        'count_high': int(counts[1]),
        'share_low': float(counts[0] / total),
        'share_high': float(counts[1] / total),
    }


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


@dataclass
class FoldResult:
    fold_id: str
    best_epoch: int
    metrics: Dict[str, float]
    checkpoint_path: str
    scaler_path: str
    confusion_matrix: List[List[int]]
    train_size: int
    val_size: int


class FocalLoss(nn.Module):
    """Focal loss for hard/minority samples in imbalanced binary classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def augment_batch(
    x: torch.Tensor,
    noise_std: float,
    channel_drop_prob: float,
    time_mask_width: int,
) -> torch.Tensor:
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    if channel_drop_prob > 0.0:
        # Drop random channels per-sample to improve robustness to channel noise.
        drop = (
            torch.rand(x.size(0), x.size(1), 1, device=x.device) < channel_drop_prob
        )
        x = x.masked_fill(drop, 0.0)

    if time_mask_width > 0:
        t_len = x.size(2)
        width = min(time_mask_width, max(t_len - 1, 1))
        if width > 0:
            starts = torch.randint(
                low=0, high=max(t_len - width + 1, 1), size=(x.size(0),), device=x.device
            )
            for i in range(x.size(0)):
                s = int(starts[i].item())
                x[i, :, s : s + width] = 0.0
    return x


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train_noise_std: float = 0.0,
    train_channel_drop_prob: float = 0.0,
    train_time_mask_width: int = 0,
    grad_clip_norm: float = 0.0,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                x = augment_batch(
                    x,
                    noise_std=train_noise_std,
                    channel_drop_prob=train_channel_drop_prob,
                    time_mask_width=train_time_mask_width,
                )
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    denom = max(len(all_true), 1)
    return EpochMetrics(
        loss=total_loss / denom,
        accuracy=accuracy_score(all_true, all_pred),
        balanced_accuracy=balanced_accuracy_score(all_true, all_pred),
        macro_f1=f1_score(all_true, all_pred, average='macro'),
    )


def build_raw_dataset(mat_path: Path, io_path: str, chunk_size: int, step_size: int):
    overlap = chunk_size - step_size
    return DREAMERDataset(
        mat_path=str(mat_path),
        io_path=io_path,
        chunk_size=chunk_size,
        overlap=overlap,
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=map_quadrant_label,
    )


def split_iterator(
    dataset,
    split: str,
    test_size: float,
    seed: int,
    split_root: Path,
    n_splits: int,
):
    if split == 'cross_trial':
        train_ds, val_ds = train_test_split_cross_trial(
            dataset,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            split_path=str(split_root / 'cross_trial'),
        )
        yield 'cross_trial', train_ds, val_ds
        return

    if split == 'cross_subject':
        train_ds, val_ds = train_test_split_cross_subject(
            dataset,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            split_path=str(split_root / 'cross_subject'),
        )
        yield 'cross_subject', train_ds, val_ds
        return

    if split == 'loso':
        cv = LeaveOneSubjectOut(split_path=str(split_root / 'loso'))
        for fold_idx, (train_ds, val_ds) in enumerate(cv.split(dataset), start=1):
            subject = str(val_ds.info['subject_id'].iloc[0])
            yield f'loso_subject_{subject}_{fold_idx}', train_ds, val_ds
        return

    if split == 'group_kfold_subject':
        if n_splits < 2:
            raise ValueError('--n-splits must be >= 2 for group_kfold_subject.')
        groups = dataset.info['subject_id'].to_numpy()
        gkf = GroupKFold(n_splits=n_splits)
        dummy_x = np.zeros(len(groups), dtype=np.int8)
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(dummy_x, groups=groups),
            start=1,
        ):
            train_ds = IndexedDataset(dataset, train_idx)
            val_ds = IndexedDataset(dataset, val_idx)
            yield f'gkf_subject_{fold_idx}', train_ds, val_ds
        return

    raise ValueError(f'Unsupported split mode: {split}')


def train_fold(
    fold_id: str,
    task: str,
    train_raw,
    val_raw,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    class_weighting: bool,
    dropout: float,
    weighted_sampling: bool,
    loss_name: str,
    label_smoothing: float,
    focal_gamma: float,
    train_noise_std: float,
    train_channel_drop_prob: float,
    train_time_mask_width: int,
    grad_clip_norm: float,
) -> FoldResult:
    fold_dir = output_dir / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)

    scaler = fit_channel_scaler(train_raw, batch_size=batch_size)
    scaler_path = fold_dir / f'{task}_scaler.pkl'
    joblib.dump(scaler, scaler_path)

    train_ds = BinaryLabelDataset(train_raw, scaler, task=task)
    val_ds = BinaryLabelDataset(val_raw, scaler, task=task)
    tr_dist = class_distribution(train_ds.labels)
    va_dist = class_distribution(val_ds.labels)
    print(
        f'[{fold_id}] class mix train(low/high)='
        f"{tr_dist['count_low']}/{tr_dist['count_high']} "
        f"({tr_dist['share_low']:.2%}/{tr_dist['share_high']:.2%}), "
        f"val(low/high)={va_dist['count_low']}/{va_dist['count_high']} "
        f"({va_dist['share_low']:.2%}/{va_dist['share_high']:.2%})"
    )

    if weighted_sampling:
        class_counts = np.bincount(train_ds.labels, minlength=2).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        sample_weights = 1.0 / class_counts[train_ds.labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = EEGNet(
        n_channels=NUM_CHANNELS,
        n_samples=CHUNK_SIZE,
        n_classes=2,
        dropout=dropout,
    ).to(device)

    weight = None
    if class_weighting:
        weight = compute_class_weights(train_ds.labels).to(device)
    if loss_name == 'focal':
        criterion = FocalLoss(
            gamma=focal_gamma, weight=weight, label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_score = -1.0
    best_epoch = 0
    best_path = fold_dir / f'eegnet_{task}_best.pth'
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            train_noise_std=train_noise_std,
            train_channel_drop_prob=train_channel_drop_prob,
            train_time_mask_width=train_time_mask_width,
            grad_clip_norm=grad_clip_norm,
        )
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        score = (0.7 * va.balanced_accuracy) + (0.3 * va.macro_f1)
        scheduler.step(score)
        row = {
            'epoch': epoch,
            'train_loss': tr.loss,
            'train_accuracy': tr.accuracy,
            'train_balanced_accuracy': tr.balanced_accuracy,
            'train_macro_f1': tr.macro_f1,
            'val_loss': va.loss,
            'val_accuracy': va.accuracy,
            'val_balanced_accuracy': va.balanced_accuracy,
            'val_macro_f1': va.macro_f1,
            'selection_score': score,
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        print(
            f'[{fold_id}] {task} epoch {epoch:02d}: '
            f'val_bal_acc={va.balanced_accuracy:.4f}, val_f1={va.macro_f1:.4f}'
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print(f'[{fold_id}] Early stopping at epoch {epoch}.')
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }

    with (fold_dir / f'{task}_history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return FoldResult(
        fold_id=fold_id,
        best_epoch=best_epoch,
        metrics=metrics,
        checkpoint_path=str(best_path),
        scaler_path=str(scaler_path),
        confusion_matrix=cm.astype(int).tolist(),
        train_size=len(train_ds),
        val_size=len(val_ds),
    )


def summarize_results(results: Sequence[FoldResult]) -> Dict[str, float]:
    if not results:
        return {}
    acc = np.asarray([r.metrics['accuracy'] for r in results], dtype=np.float64)
    bal = np.asarray(
        [r.metrics['balanced_accuracy'] for r in results], dtype=np.float64
    )
    f1 = np.asarray([r.metrics['macro_f1'] for r in results], dtype=np.float64)
    return {
        'folds': len(results),
        'accuracy_mean': float(acc.mean()),
        'accuracy_std': float(acc.std(ddof=0)),
        'balanced_accuracy_mean': float(bal.mean()),
        'balanced_accuracy_std': float(bal.std(ddof=0)),
        'macro_f1_mean': float(f1.mean()),
        'macro_f1_std': float(f1.std(ddof=0)),
    }


def _json_safe_config(args: argparse.Namespace) -> Dict[str, object]:
    config: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == 'custom':
        return args
    preset = PRESET_CONFIGS[args.preset]
    args.lr = float(preset['lr'])
    args.weight_decay = float(preset['weight_decay'])
    args.patience = int(preset['patience'])
    args.loss = str(preset['loss'])
    args.label_smoothing = float(preset['label_smoothing'])
    args.focal_gamma = float(preset['focal_gamma'])
    args.weighted_sampling = bool(preset['weighted_sampling'])
    args.train_noise_std = float(preset['train_noise_std'])
    args.train_channel_drop_prob = float(preset['train_channel_drop_prob'])
    args.train_time_mask_width = int(preset['train_time_mask_width'])
    args.grad_clip_norm = float(preset['grad_clip_norm'])
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Train production-grade EEGNet binary model on DREAMER. '
            'Runs with delivery defaults even if no arguments are provided.'
        )
    )
    parser.add_argument(
        '--task',
        choices=['valence', 'arousal', 'both'],
        default=DEFAULT_TRAIN_CONFIG['task'],
        help=(
            'Training target. Default: both. '
            "Use 'both' to train valence and arousal sequentially."
        ),
    )
    parser.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    parser.add_argument(
        '--io-path',
        type=str,
        default=f'torcheeg_cache_raw_c{CHUNK_SIZE}_s{DEFAULT_STEP_SIZE}',
    )
    parser.add_argument(
        '--split',
        choices=['cross_trial', 'cross_subject', 'loso', 'group_kfold_subject'],
        default=DEFAULT_TRAIN_CONFIG['split'],
    )
    parser.add_argument(
        '--n-splits', type=int, default=DEFAULT_TRAIN_CONFIG['n_splits']
    )
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAIN_CONFIG['epochs'])
    parser.add_argument(
        '--batch-size', type=int, default=DEFAULT_TRAIN_CONFIG['batch_size']
    )
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument(
        '--preset',
        choices=['baseline', 'balanced', 'robust', 'aggressive', 'custom'],
        default='balanced',
        help=(
            'Tuning bundle for training parameters. '
            "Use custom to honor manual flags as-is."
        ),
    )
    parser.add_argument(
        '--loss',
        choices=['cross_entropy', 'focal'],
        default='cross_entropy',
        help='Loss function; focal often improves imbalance robustness.',
    )
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument(
        '--weighted-sampling',
        action='store_true',
        help='Enable weighted random sampling on training batches.',
    )
    parser.add_argument('--train-noise-std', type=float, default=0.0)
    parser.add_argument('--train-channel-drop-prob', type=float, default=0.0)
    parser.add_argument('--train-time-mask-width', type=int, default=0)
    parser.add_argument('--grad-clip-norm', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--max-folds', type=int, default=0)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_TRAIN_CONFIG['output_dir']),
    )
    parser.add_argument(
        '--device',
        default=DEFAULT_TRAIN_CONFIG['device'],
        choices=['auto', 'cpu', 'cuda'],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args = apply_preset(args)
    if not args.mat_path.is_file():
        raise FileNotFoundError(
            f'{args.mat_path} not found. Place DREAMER.mat in project root or pass --mat-path.'
        )

    set_seed(args.seed)
    apply_lmdb_patch()
    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device}')

    dataset = build_raw_dataset(
        mat_path=args.mat_path,
        io_path=args.io_path,
        chunk_size=CHUNK_SIZE,
        step_size=DEFAULT_STEP_SIZE,
    )
    print(f'Dataset windows: {len(dataset)}')

    tasks = ['valence', 'arousal'] if args.task == 'both' else [args.task]
    for task in tasks:
        print(f'\n######## Training task: {task} ########')
        split_root = args.output_dir / 'splits'
        all_results: List[FoldResult] = []
        for fold_idx, (fold_id, train_raw, val_raw) in enumerate(
            split_iterator(
                dataset=dataset,
                split=args.split,
                test_size=args.test_size,
                seed=args.seed,
                split_root=split_root,
                n_splits=args.n_splits,
            ),
            start=1,
        ):
            if args.max_folds > 0 and fold_idx > args.max_folds:
                break
            print(f'\n=== Fold {fold_idx}: {fold_id} ===')
            result = train_fold(
                fold_id=fold_id,
                task=task,
                train_raw=train_raw,
                val_raw=val_raw,
                output_dir=args.output_dir / task,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                class_weighting=not args.no_class_weights,
                dropout=args.dropout,
                weighted_sampling=args.weighted_sampling,
                loss_name=args.loss,
                label_smoothing=args.label_smoothing,
                focal_gamma=args.focal_gamma,
                train_noise_std=args.train_noise_std,
                train_channel_drop_prob=args.train_channel_drop_prob,
                train_time_mask_width=args.train_time_mask_width,
                grad_clip_norm=args.grad_clip_norm,
            )
            all_results.append(result)

        summary = summarize_results(all_results)
        report = {
            'config': _json_safe_config(args) | {'resolved_task': task},
            'summary': summary,
            'folds': [asdict(r) for r in all_results],
        }
        out_file = args.output_dir / f'{task}_{args.split}_report.json'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f'\nTraining complete for task={task}.')
        print(json.dumps(summary, indent=2))
        print(f'Report: {out_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
