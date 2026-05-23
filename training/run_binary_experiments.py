#!/usr/bin/env python3
"""Run multiple binary training experiments and rank by evaluation metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / 'training' / 'train_eegnet_binary.py'
EVAL_SCRIPT = ROOT / 'training' / 'evaluate_binary_pair_ensemble.py'


def run_cmd(cmd: List[str]) -> int:
    print('Running:', ' '.join(cmd))
    return int(subprocess.call(cmd, cwd=ROOT))


def read_json_from_stdout(cmd: List[str]) -> Dict:
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command failed ({proc.returncode}): {" ".join(cmd)}\n'
            f'STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
        )
    start = proc.stdout.find('{')
    end = proc.stdout.rfind('}')
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f'Could not parse JSON output.\nSTDOUT:\n{proc.stdout}')
    return json.loads(proc.stdout[start : end + 1])


def parse_experiments(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    exps = payload.get('experiments')
    if not isinstance(exps, list) or not exps:
        raise ValueError('experiments JSON must contain a non-empty "experiments" list.')
    return exps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True, help='Path to experiments JSON plan.')
    p.add_argument('--mat-path', type=Path, default=Path('DREAMER.mat'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--eval-split', choices=['cross_subject', 'cross_trial'], default='cross_subject')
    p.add_argument('--ensemble-top-k', type=int, default=3)
    p.add_argument('--report-out', type=Path, default=Path('artifacts/experiment_leaderboard.json'))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.plan.is_file():
        raise FileNotFoundError(args.plan)
    if not args.mat_path.is_file():
        raise FileNotFoundError(args.mat_path)

    experiments = parse_experiments(args.plan)
    leaderboard: List[Dict] = []

    for idx, exp in enumerate(experiments, start=1):
        name = str(exp.get('name', f'experiment_{idx}'))
        output_dir = str(exp.get('output_dir', f'artifacts/{name}'))
        split = str(exp.get('split', 'group_kfold_subject'))
        preset = str(exp.get('preset', 'balanced'))
        epochs = int(exp.get('epochs', 30))
        batch_size = int(exp.get('batch_size', 64))
        n_splits = int(exp.get('n_splits', 5))
        max_folds = int(exp.get('max_folds', 0))
        task = str(exp.get('task', 'both'))

        train_cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            '--task',
            task,
            '--split',
            split,
            '--n-splits',
            str(n_splits),
            '--epochs',
            str(epochs),
            '--batch-size',
            str(batch_size),
            '--preset',
            preset,
            '--output-dir',
            output_dir,
            '--device',
            args.device,
            '--mat-path',
            str(args.mat_path),
        ]
        if max_folds > 0:
            train_cmd.extend(['--max-folds', str(max_folds)])
        if exp.get('no_class_weights', False):
            train_cmd.append('--no-class-weights')
        if int(exp.get('seed', 42)) != 42:
            train_cmd.extend(['--seed', str(int(exp['seed']))])

        code = run_cmd(train_cmd)
        if code != 0:
            leaderboard.append(
                {
                    'name': name,
                    'status': 'train_failed',
                    'return_code': code,
                    'config': exp,
                }
            )
            continue

        v_report = Path(output_dir) / f'valence_{split}_report.json'
        a_report = Path(output_dir) / f'arousal_{split}_report.json'
        if not v_report.is_file() or not a_report.is_file():
            leaderboard.append(
                {
                    'name': name,
                    'status': 'missing_reports',
                    'valence_report': str(v_report),
                    'arousal_report': str(a_report),
                    'config': exp,
                }
            )
            continue

        eval_cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            '--valence-report',
            str(v_report),
            '--arousal-report',
            str(a_report),
            '--top-k',
            str(args.ensemble_top_k),
            '--metric',
            'balanced_accuracy',
            '--split',
            args.eval_split,
            '--device',
            args.device,
            '--mat-path',
            str(args.mat_path),
        ]
        try:
            eval_json = read_json_from_stdout(eval_cmd)
        except Exception as exc:  # noqa: BLE001
            leaderboard.append(
                {
                    'name': name,
                    'status': 'eval_failed',
                    'error': str(exc),
                    'config': exp,
                }
            )
            continue

        quad = eval_json['quadrant_from_binary']
        leaderboard.append(
            {
                'name': name,
                'status': 'ok',
                'config': exp,
                'train_output_dir': output_dir,
                'quadrant_accuracy': float(quad['accuracy']),
                'quadrant_balanced_accuracy': float(quad['balanced_accuracy']),
                'quadrant_macro_f1': float(quad['macro_f1']),
                'evaluation': eval_json,
            }
        )

    successful = [row for row in leaderboard if row.get('status') == 'ok']
    successful.sort(
        key=lambda row: (
            row['quadrant_balanced_accuracy'],
            row['quadrant_macro_f1'],
            row['quadrant_accuracy'],
        ),
        reverse=True,
    )

    output = {
        'summary': {
            'total_experiments': len(leaderboard),
            'successful_experiments': len(successful),
            'ranking_key': [
                'quadrant_balanced_accuracy',
                'quadrant_macro_f1',
                'quadrant_accuracy',
            ],
        },
        'leaderboard': successful,
        'failures': [row for row in leaderboard if row.get('status') != 'ok'],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open('w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output['summary'], indent=2))
    if successful:
        best = successful[0]
        print(
            json.dumps(
                {
                    'best_experiment': best['name'],
                    'quadrant_accuracy': best['quadrant_accuracy'],
                    'quadrant_balanced_accuracy': best['quadrant_balanced_accuracy'],
                    'quadrant_macro_f1': best['quadrant_macro_f1'],
                    'train_output_dir': best['train_output_dir'],
                },
                indent=2,
            )
        )
    print(f'Leaderboard report saved to: {args.report_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
