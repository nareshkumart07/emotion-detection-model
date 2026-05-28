#!/usr/bin/env python3
"""Train 4-class quadrant BiLSTM on DREAMER with notebook-equivalent pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torcheeg import transforms
from torcheeg.datasets import DREAMERDataset
from torcheeg.model_selection import (
    train_test_split_cross_subject,
    train_test_split_cross_trial,
)
from tqdm.auto import tqdm

from inference.config import (
    BAND_DICT,
    CHUNK_SIZE,
    NUM_CHANNELS,
    QUADRANT_NAMES,
    SAMPLING_RATE,
    STEP_SIZE,
)
from inference.model_arch import EEGBiLSTMClassifier
from inference.torcheeg_patch import apply_lmdb_patch

NUM_CLASSES = 4
VALENCE_THRESHOLD = 3.0
AROUSAL_THRESHOLD = 3.0


def quadrant_from_binary(high_valence: bool, high_arousal: bool) -> int:
    if high_valence and high_arousal:
        return 0
    if (not high_valence) and high_arousal:
        return 1
    if (not high_valence) and (not high_arousal):
        return 2
    return 3


def map_to_four_quadrants(y: Dict, **kwargs) -> Dict[str, int]:
    _ = kwargs
    valence = float(y['valence'])
    arousal = float(y['arousal'])
    label = quadrant_from_binary(
        high_valence=valence > VALENCE_THRESHOLD,
        high_arousal=arousal > AROUSAL_THRESHOLD,
    )
    return {'y': int(label)}


class ScaledDataset(Dataset):
    def __init__(self, base_dataset, scaler: StandardScaler):
        self.base_dataset = base_dataset
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        x, y = self.base_dataset[index]
        arr = x.numpy().astype(np.float32)
        scaled = self.scaler.transform(arr.reshape(1, -1)).astype(np.float32).reshape(
            arr.shape
        )
        return torch.from_numpy(scaled), int(y)


def fit_standard_scaler(ds, batch_size: int = 512) -> StandardScaler:
    scaler = StandardScaler()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for x_batch, _ in tqdm(loader, desc='Fitting StandardScaler (train only)'):
        scaler.partial_fit(x_batch.numpy().reshape(len(x_batch), -1))
    return scaler


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    return_preds: bool = False,
) -> Tuple[EpochStats, Optional[np.ndarray], Optional[np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_pred: List[int] = []
    all_true: List[int] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long()
        with torch.set_grad_enabled(is_train):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        preds = outputs.argmax(1)
        total_loss += float(loss.item()) * x_batch.size(0)
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        all_true.extend(y_batch.detach().cpu().numpy().tolist())

    n = max(len(all_true), 1)
    stats = EpochStats(
        loss=total_loss / n,
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        macro_f1=float(f1_score(all_true, all_pred, average='macro')),
    )

    if return_preds:
        return (
            stats,
            np.asarray(all_pred, dtype=np.int64),
            np.asarray(all_true, dtype=np.int64),
        )
    return stats, None, None


def save_learning_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    val_loss = [row['val_loss'] for row in history]
    train_acc = [row['train_accuracy'] for row in history]
    val_acc = [row['val_accuracy'] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, 'o-', label='Train')
    axes[0].plot(epochs, val_loss, 'o-', label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy')
    axes[0].legend()

    axes[1].plot(epochs, train_acc, 'o-', label='Train')
    axes[1].plot(epochs, val_acc, 'o-', label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=QUADRANT_NAMES,
        yticklabels=QUADRANT_NAMES,
        ax=axes[0],
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion matrix (counts)')

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=QUADRANT_NAMES,
        yticklabels=QUADRANT_NAMES,
        ax=axes[1],
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion matrix (row-normalized)')

    axes[2].bar(QUADRANT_NAMES, per_class_f1, color='steelblue')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('F1 score')
    axes[2].set_title('Per-class F1')
    axes[2].tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train notebook-equivalent 4-class BiLSTM quadrant classifier.'
    )
    p.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    p.add_argument(
        '--io-path',
        default=f'torcheeg_cache_c{CHUNK_SIZE}_s{STEP_SIZE}',
        help='TorchEEG cache path for PSD+baseline features.',
    )
    p.add_argument(
        '--split',
        choices=['cross_trial', 'cross_subject'],
        default='cross_trial',
    )
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/quadrant'),
        help='Directory for model/scaler/metrics artifacts.',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mat_path.is_file():
        raise FileNotFoundError(
            f'{args.mat_path} not found. Place DREAMER.mat in project root or pass --mat-path.'
        )

    apply_lmdb_patch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f'Using device: {device}')

    dataset = DREAMERDataset(
        mat_path=str(args.mat_path),
        io_path=args.io_path,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_SIZE - STEP_SIZE,
        online_transform=transforms.Compose(
            [
                transforms.BandPowerSpectralDensity(
                    sampling_rate=SAMPLING_RATE,
                    band_dict=BAND_DICT,
                    apply_to_baseline=True,
                ),
                transforms.BaselineRemoval(),
                transforms.ToTensor(),
            ]
        ),
        label_transform=map_to_four_quadrants,
    )

    if args.split == 'cross_trial':
        train_raw, val_raw = train_test_split_cross_trial(
            dataset,
            test_size=args.test_size,
            shuffle=True,
            random_state=args.seed,
            split_path='artifacts/split_quadrant_cross_trial',
        )
    else:
        train_raw, val_raw = train_test_split_cross_subject(
            dataset,
            test_size=args.test_size,
            shuffle=True,
            random_state=args.seed,
            split_path='artifacts/split_quadrant_cross_subject',
        )

    scaler = fit_standard_scaler(train_raw)
    train_ds = ScaledDataset(train_raw, scaler)
    val_ds = ScaledDataset(val_raw, scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EEGBiLSTMClassifier(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    history: List[Dict[str, float]] = []
    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        tr, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        va, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step(va.loss)
        row = {
            'epoch': epoch,
            'train_loss': tr.loss,
            'val_loss': va.loss,
            'train_accuracy': tr.accuracy,
            'val_accuracy': va.accuracy,
            'train_balanced_accuracy': tr.balanced_accuracy,
            'val_balanced_accuracy': va.balanced_accuracy,
            'train_macro_f1': tr.macro_f1,
            'val_macro_f1': va.macro_f1,
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: val_acc={va.accuracy:.4f}, "
            f"val_bal_acc={va.balanced_accuracy:.4f}, val_f1={va.macro_f1:.4f}"
        )

    val_stats, y_pred, y_true = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        return_preds=True,
    )

    if y_pred is None or y_true is None:
        raise RuntimeError('Validation predictions were not collected.')

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    report = {
        'config': {
            'mat_path': str(args.mat_path),
            'io_path': args.io_path,
            'split': args.split,
            'test_size': args.test_size,
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'device': str(device),
            'num_channels': NUM_CHANNELS,
            'chunk_size': CHUNK_SIZE,
        },
        'final_validation': asdict(val_stats),
        'macro_f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'classification_report': classification_report(
            y_true,
            y_pred,
            target_names=QUADRANT_NAMES,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        'confusion_matrix': cm.astype(int).tolist(),
        'confusion_matrix_row_normalized': cm_norm.tolist(),
        'quadrant_names': QUADRANT_NAMES,
        'history': history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.output_dir / 'model.pth'
    scaler_out = args.output_dir / 'scaler.pkl'
    metrics_out = args.output_dir / 'metrics.json'
    curves_out = args.output_dir / 'learning_curves.png'
    cm_out = args.output_dir / 'confusion_matrix.png'

    torch.save(model.state_dict(), model_out)
    joblib.dump(scaler, scaler_out)
    with metrics_out.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    save_learning_curves(history, curves_out)
    save_confusion_matrix_plots(y_true=y_true, y_pred=y_pred, out_path=cm_out)

    print('\nTraining complete.')
    print(f"Validation accuracy: {100.0 * val_stats.accuracy:.2f}%")
    print(f"Validation balanced accuracy: {100.0 * val_stats.balanced_accuracy:.2f}%")
    print(f"Validation macro F1: {val_stats.macro_f1:.4f}")
    print(f'Model saved: {model_out}')
    print(f'Scaler saved: {scaler_out}')
    print(f'Metrics saved: {metrics_out}')
    print(f'Learning curves: {curves_out}')
    print(f'Confusion matrix plot: {cm_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
