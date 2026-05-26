#!/usr/bin/env python3
"""EDA + 4-class BiLSTM training/evaluation with confusion-matrix artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
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
from torcheeg.model_selection import train_test_split_cross_trial
from tqdm.auto import tqdm

from inference.torcheeg_patch import apply_lmdb_patch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

VALENCE_THRESHOLD = 3.0
AROUSAL_THRESHOLD = 3.0
NUM_CLASSES = 4
QUADRANT_NAMES = ['Happy', 'Stressed', 'Depressed', 'Calm']
DEFAULT_CHUNK_SIZE = 256
DEFAULT_STEP_SIZE = 128
DEFAULT_BAND_DICT = {
    'theta': [4, 8],
    'alpha': [8, 14],
    'beta': [14, 30],
    'gamma': [30, 47],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadrant_from_valence_arousal(valence: float, arousal: float) -> int:
    if valence > VALENCE_THRESHOLD and arousal > AROUSAL_THRESHOLD:
        return 0
    if valence <= VALENCE_THRESHOLD and arousal > AROUSAL_THRESHOLD:
        return 1
    if valence <= VALENCE_THRESHOLD and arousal <= AROUSAL_THRESHOLD:
        return 2
    return 3


def map_to_four_quadrants(y: Dict, **kwargs) -> Dict[str, int]:
    _ = kwargs
    return {'y': quadrant_from_valence_arousal(y['valence'], y['arousal'])}


class ScaledDataset(Dataset):
    def __init__(self, base_dataset, scaler: StandardScaler):
        self.base_dataset = base_dataset
        self.scaler = scaler
        info = base_dataset.info
        self.labels = np.asarray(
            [
                quadrant_from_valence_arousal(v, a)
                for v, a in zip(info['valence'].values, info['arousal'].values)
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x, _ = self.base_dataset[index]
        flat = x.numpy().astype(np.float32).reshape(1, -1)
        scaled = self.scaler.transform(flat).astype(np.float32).reshape(x.shape)
        return torch.from_numpy(scaled), int(self.labels[index])


class EEGBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=max(num_layers - 1, 1),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 2 else 0.0,
        )
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)


def fit_standard_scaler(ds, batch_size: int) -> StandardScaler:
    scaler = StandardScaler()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for x_batch, _ in tqdm(loader, desc='Fitting StandardScaler (train-only)'):
        scaler.partial_fit(x_batch.numpy().reshape(len(x_batch), -1))
    return scaler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds: List[int] = []
    all_true: List[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(1)
            total_loss += float(loss.item()) * x_batch.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_true.extend(y_batch.detach().cpu().numpy().tolist())

    denom = max(len(all_true), 1)
    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    return {
        'loss': total_loss / denom,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'y_true': y_true,
        'y_pred': y_pred,
    }


def add_quadrant_columns(info: pd.DataFrame) -> pd.DataFrame:
    info = info.copy()
    info['quadrant'] = [
        quadrant_from_valence_arousal(v, a)
        for v, a in zip(info['valence'].values, info['arousal'].values)
    ]
    info['quadrant_name'] = info['quadrant'].map(dict(enumerate(QUADRANT_NAMES)))
    return info


def save_eda_outputs(dataset, output_dir: Path) -> Dict[str, object]:
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    info = add_quadrant_columns(dataset.info)
    trial_info = info.drop_duplicates(['subject_id', 'trial_id'])

    window_counts = (
        info['quadrant_name'].value_counts().reindex(QUADRANT_NAMES, fill_value=0)
    )
    trial_counts = (
        trial_info['quadrant_name'].value_counts().reindex(QUADRANT_NAMES, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(
        x=window_counts.index,
        y=window_counts.values,
        hue=window_counts.index,
        palette='viridis',
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title('Windows per quadrant')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=15)

    sns.barplot(
        x=trial_counts.index,
        y=trial_counts.values,
        hue=trial_counts.index,
        palette='magma',
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title('Trials per quadrant')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    fig.savefig(plots_dir / 'eda_label_distribution.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(
        info['valence'],
        bins=20,
        kde=True,
        color='tab:blue',
        ax=axes[0],
    )
    axes[0].axvline(VALENCE_THRESHOLD, color='black', linestyle='--', linewidth=1)
    axes[0].set_title('Valence distribution')

    sns.histplot(
        info['arousal'],
        bins=20,
        kde=True,
        color='tab:orange',
        ax=axes[1],
    )
    axes[1].axvline(AROUSAL_THRESHOLD, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Arousal distribution')
    plt.tight_layout()
    fig.savefig(plots_dir / 'eda_valence_arousal_hist.png', dpi=160)
    plt.close(fig)

    scatter = info[
        ['valence', 'arousal', 'quadrant_name']
    ].sample(n=min(12000, len(info)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=scatter,
        x='valence',
        y='arousal',
        hue='quadrant_name',
        hue_order=QUADRANT_NAMES,
        alpha=0.45,
        s=14,
        ax=ax,
    )
    ax.axvline(VALENCE_THRESHOLD, color='black', linestyle='--', linewidth=1)
    ax.axhline(AROUSAL_THRESHOLD, color='black', linestyle='--', linewidth=1)
    ax.set_title('Valence-Arousal scatter with quadrant boundaries')
    plt.tight_layout()
    fig.savefig(plots_dir / 'eda_valence_arousal_scatter.png', dpi=160)
    plt.close(fig)

    summary = {
        'subjects': int(info['subject_id'].nunique()),
        'trials': int(trial_info.shape[0]),
        'windows': int(info.shape[0]),
        'window_counts': {k: int(v) for k, v in window_counts.items()},
        'trial_counts': {k: int(v) for k, v in trial_counts.items()},
        'trial_share': {
            k: float(v / max(trial_counts.sum(), 1)) for k, v in trial_counts.items()
        },
    }
    (output_dir / 'eda_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    return summary


def save_learning_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'o-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Cross-entropy loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], 'o-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'o-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Classification accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=QUADRANT_NAMES,
        yticklabels=QUADRANT_NAMES,
        ax=axes[0],
    )
    axes[0].set_title('Confusion matrix (count)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=QUADRANT_NAMES,
        yticklabels=QUADRANT_NAMES,
        ax=axes[1],
    )
    axes[1].set_title('Confusion matrix (row-normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    axes[2].bar(QUADRANT_NAMES, per_class_f1, color='steelblue')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('F1 score')
    axes[2].set_title('Per-class F1')
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig.savefig(output_dir / 'plots' / 'confusion_matrices.png', dpi=170)
    plt.close(fig)

    return {
        'confusion_matrix': cm.astype(int).tolist(),
        'confusion_matrix_row_normalized': cm_norm.tolist(),
        'per_class_f1': {
            QUADRANT_NAMES[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run EDA + 4-class BiLSTM training and save confusion matrix artifacts.'
    )
    parser.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/eda_quadrant'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mat_path.is_file():
        raise FileNotFoundError(
            f'{args.mat_path} not found. Place DREAMER.mat in project root or pass --mat-path.'
        )

    set_seed(args.seed)
    apply_lmdb_patch()

    device_name = (
        'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    )
    if device_name == 'auto':
        device_name = 'cpu'
    device = torch.device(device_name)

    overlap = args.chunk_size - args.step_size
    if overlap < 0:
        raise ValueError('--step-size must be <= --chunk-size.')
    io_path = f'torcheeg_cache_bandpower_c{args.chunk_size}_s{args.step_size}'

    output_dir = args.output_dir
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)

    dataset = DREAMERDataset(
        mat_path=str(args.mat_path),
        io_path=io_path,
        chunk_size=args.chunk_size,
        overlap=overlap,
        online_transform=transforms.Compose(
            [
                transforms.BandPowerSpectralDensity(
                    sampling_rate=128,
                    band_dict=DEFAULT_BAND_DICT,
                    apply_to_baseline=True,
                ),
                transforms.BaselineRemoval(),
                transforms.ToTensor(),
            ]
        ),
        label_transform=map_to_four_quadrants,
    )
    sample_x, _ = dataset[0]
    print(
        f'Dataset windows={len(dataset)} | sample shape={tuple(sample_x.shape)} | device={device}'
    )

    eda_summary = save_eda_outputs(dataset, output_dir)
    print('Saved EDA plots and summary.')

    train_raw, val_raw = train_test_split_cross_trial(
        dataset,
        test_size=args.test_size,
        shuffle=True,
        random_state=args.seed,
        split_path=str(output_dir / 'splits' / 'cross_trial'),
    )

    scaler = fit_standard_scaler(train_raw, batch_size=max(args.batch_size, 32))
    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)

    train_dataset = ScaledDataset(train_raw, scaler)
    val_dataset = ScaledDataset(val_raw, scaler)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = EEGBiLSTMClassifier(
        input_size=sample_x.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    print(f'Starting training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step(val_metrics['loss'])

        history['train_loss'].append(float(train_metrics['loss']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['train_acc'].append(100.0 * float(train_metrics['accuracy']))
        history['val_acc'].append(100.0 * float(val_metrics['accuracy']))

        score = (
            0.6 * float(val_metrics['balanced_accuracy'])
            + 0.4 * float(val_metrics['macro_f1'])
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f'Epoch {epoch:02d}/{args.epochs}: '
            f'train_acc={100.0 * float(train_metrics["accuracy"]):.2f}% | '
            f'val_acc={100.0 * float(val_metrics["accuracy"]):.2f}% | '
            f'val_bal_acc={100.0 * float(val_metrics["balanced_accuracy"]):.2f}% | '
            f'val_macro_f1={float(val_metrics["macro_f1"]):.4f}'
        )

    model.load_state_dict(best_state)
    model.eval()
    final_train = run_epoch(
        model=model, loader=train_loader, criterion=criterion, device=device, optimizer=None
    )
    final_val = run_epoch(
        model=model, loader=val_loader, criterion=criterion, device=device, optimizer=None
    )

    curves_path = output_dir / 'plots' / 'learning_curves.png'
    save_learning_curves(history, curves_path)

    confusion_report = save_confusion_outputs(
        y_true=final_val['y_true'], y_pred=final_val['y_pred'], output_dir=output_dir
    )

    classif_report = classification_report(
        final_val['y_true'],
        final_val['y_pred'],
        labels=list(range(NUM_CLASSES)),
        target_names=QUADRANT_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    model_path = output_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)

    report = {
        'config': {
            'mat_path': str(args.mat_path),
            'chunk_size': args.chunk_size,
            'step_size': args.step_size,
            'test_size': args.test_size,
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'device': str(device),
            'io_path': io_path,
        },
        'data': eda_summary,
        'split': {
            'train_windows': len(train_dataset),
            'val_windows': len(val_dataset),
            'train_trials': int(train_raw.info.groupby(['subject_id', 'trial_id']).ngroups),
            'val_trials': int(val_raw.info.groupby(['subject_id', 'trial_id']).ngroups),
        },
        'best_epoch': best_epoch,
        'best_selection_score': best_score,
        'train_metrics': {
            'loss': float(final_train['loss']),
            'accuracy': float(final_train['accuracy']),
            'balanced_accuracy': float(final_train['balanced_accuracy']),
            'macro_f1': float(final_train['macro_f1']),
        },
        'val_metrics': {
            'loss': float(final_val['loss']),
            'accuracy': float(final_val['accuracy']),
            'balanced_accuracy': float(final_val['balanced_accuracy']),
            'macro_f1': float(final_val['macro_f1']),
            'weighted_f1': float(
                f1_score(
                    final_val['y_true'],
                    final_val['y_pred'],
                    average='weighted',
                    zero_division=0,
                )
            ),
        },
        'classification_report': classif_report,
        **confusion_report,
        'artifacts': {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'learning_curves_plot': str(curves_path),
            'confusion_plot': str(output_dir / 'plots' / 'confusion_matrices.png'),
        },
    }
    report_path = output_dir / 'eda_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nFinal validation metrics:')
    print(
        f'  Accuracy={100.0 * float(final_val["accuracy"]):.2f}% | '
        f'Balanced={100.0 * float(final_val["balanced_accuracy"]):.2f}% | '
        f'Macro-F1={float(final_val["macro_f1"]):.4f}'
    )
    print(f'Saved report: {report_path}')
    print(f'Saved model: {model_path}')
    print(f'Saved scaler: {scaler_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
