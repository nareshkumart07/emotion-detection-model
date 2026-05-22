"""Command-line interface for emotion quadrant inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import CHUNK_SIZE, DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH, NUM_CHANNELS, STEP_SIZE
from .predictor import load_predictor
from .preprocess import validate_eeg_window
from .torcheeg_patch import apply_lmdb_patch


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _smoke_test_dataset(index: int) -> tuple[np.ndarray, np.ndarray]:
    from torcheeg.datasets import DREAMERDataset

    apply_lmdb_patch()

    overlap = CHUNK_SIZE - STEP_SIZE
    io_path = f'torcheeg_cache_c{CHUNK_SIZE}_s{STEP_SIZE}'
    mat_path = Path('DREAMER.mat')
    if not mat_path.is_file():
        raise FileNotFoundError('DREAMER.mat not found in the current directory.')

    ds = DREAMERDataset(
        mat_path=str(mat_path),
        io_path=io_path,
        chunk_size=CHUNK_SIZE,
        overlap=overlap,
    )
    info = ds.read_info(index)
    eeg = ds.read_eeg(str(info['_record_id']), str(info['clip_id']))
    baseline = ds.read_eeg(str(info['_record_id']), str(info['baseline_id']))
    return np.asarray(eeg), np.asarray(baseline)


def main() -> int:
    parser = argparse.ArgumentParser(description='Predict emotion quadrant from EEG.')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--scaler', type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument('--device', default=None, help='cpu or cuda')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--features',
        type=Path,
        help='NumPy file with shape (14, 4) baseline-corrected PSD',
    )
    group.add_argument(
        '--eeg',
        type=Path,
        help=f'NumPy file with shape ({NUM_CHANNELS}, {CHUNK_SIZE}) stimulus window',
    )
    group.add_argument(
        '--smoke-dataset-index',
        type=int,
        help='Load window/baseline from cached DREAMER dataset by index',
    )

    parser.add_argument(
        '--baseline',
        type=Path,
        help='NumPy baseline (14, T) required with --eeg',
    )
    parser.add_argument('--json', action='store_true', help='Print JSON only')
    args = parser.parse_args()

    predictor = load_predictor(args.model, args.scaler, args.device)

    if args.features:
        features = _load_array(args.features)
        result = predictor.predict_features(features)
    elif args.smoke_dataset_index is not None:
        eeg, baseline = _smoke_test_dataset(args.smoke_dataset_index)
        result = predictor.predict_window(eeg, baseline)
    else:
        if args.baseline is None:
            parser.error('--baseline is required when using --eeg')
        eeg = validate_eeg_window(_load_array(args.eeg))
        baseline = _load_array(args.baseline)
        result = predictor.predict_window(eeg, baseline)

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Prediction: {payload['label_name']} (class {payload['label_id']})")
        print(f"Confidence: {payload['confidence']:.4f}")
        print('Probabilities:')
        for name, prob in payload['probabilities'].items():
            print(f'  {name}: {prob:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
