#!/usr/bin/env python3
"""Pick best fold checkpoint from a training report JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True, help='Path to *_report.json')
    p.add_argument(
        '--metric',
        default='balanced_accuracy',
        choices=['balanced_accuracy', 'accuracy', 'macro_f1'],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(args.report)

    with args.report.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    folds = payload.get('folds', [])
    if not folds:
        raise ValueError('No folds found in report.')

    best = max(folds, key=lambda x: float(x['metrics'][args.metric]))
    output = {
        'metric': args.metric,
        'best_fold': best['fold_id'],
        'best_metric_value': float(best['metrics'][args.metric]),
        'checkpoint_path': best['checkpoint_path'],
        'scaler_path': best['scaler_path'],
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
