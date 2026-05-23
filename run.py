#!/usr/bin/env python3
"""Simple launcher for normal users.

Use this file when you do not want to remember long commands.
Examples:
  python run.py train
  python run.py streamlit
  python run.py api
  python run.py evaluate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def cmd_train(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / 'training' / 'train_eegnet_binary.py'),
        '--task',
        args.task,
        '--split',
        args.split,
        '--n-splits',
        str(args.n_splits),
        '--epochs',
        str(args.epochs),
        '--batch-size',
        str(args.batch_size),
        '--preset',
        args.preset,
        '--output-dir',
        args.output_dir,
        '--device',
        args.device,
        '--mat-path',
        args.mat_path,
    ]
    if args.max_folds > 0:
        cmd.extend(['--max-folds', str(args.max_folds)])
    return _run(cmd)


def cmd_streamlit(_: argparse.Namespace) -> int:
    cmd = [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py']
    return _run(cmd)


def cmd_api(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        '-m',
        'uvicorn',
        'app:app',
        '--reload',
        '--host',
        args.host,
        '--port',
        str(args.port),
    ]
    return _run(cmd)


def cmd_evaluate(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / 'training' / 'evaluate_binary_pair.py'),
        '--valence-model',
        args.valence_model,
        '--arousal-model',
        args.arousal_model,
        '--scaler',
        args.scaler,
        '--split',
        args.split,
        '--device',
        args.device,
        '--mat-path',
        args.mat_path,
    ]
    return _run(cmd)


def cmd_evaluate_ensemble(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / 'training' / 'evaluate_binary_pair_ensemble.py'),
        '--valence-report',
        args.valence_report,
        '--arousal-report',
        args.arousal_report,
        '--top-k',
        str(args.top_k),
        '--metric',
        args.metric,
        '--split',
        args.split,
        '--device',
        args.device,
        '--mat-path',
        args.mat_path,
    ]
    return _run(cmd)


def cmd_experiments(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / 'training' / 'run_binary_experiments.py'),
        '--plan',
        args.plan,
        '--mat-path',
        args.mat_path,
        '--device',
        args.device,
        '--eval-split',
        args.eval_split,
        '--ensemble-top-k',
        str(args.ensemble_top_k),
        '--report-out',
        args.report_out,
    ]
    return _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='One-command launcher for training and demo.'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train binary EEGNet model(s)')
    p_train.add_argument('--task', choices=['valence', 'arousal', 'both'], default='both')
    p_train.add_argument(
        '--split',
        choices=['group_kfold_subject', 'cross_subject', 'cross_trial', 'loso'],
        default='group_kfold_subject',
    )
    p_train.add_argument('--n-splits', type=int, default=5)
    p_train.add_argument('--epochs', type=int, default=30)
    p_train.add_argument('--batch-size', type=int, default=64)
    p_train.add_argument(
        '--preset',
        choices=['baseline', 'balanced', 'robust', 'aggressive', 'custom'],
        default='balanced',
    )
    p_train.add_argument('--output-dir', default='artifacts_cv5')
    p_train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p_train.add_argument('--mat-path', default='DREAMER.mat')
    p_train.add_argument('--max-folds', type=int, default=0)
    p_train.set_defaults(func=cmd_train)

    p_streamlit = sub.add_parser('streamlit', help='Run Streamlit dashboard')
    p_streamlit.set_defaults(func=cmd_streamlit)

    p_api = sub.add_parser('api', help='Run FastAPI server')
    p_api.add_argument('--host', default='127.0.0.1')
    p_api.add_argument('--port', type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    p_eval = sub.add_parser('evaluate', help='Evaluate valence+arousal model pair')
    p_eval.add_argument('--valence-model', default='models/binary/eegnet_valence.pth')
    p_eval.add_argument('--arousal-model', default='models/binary/eegnet_arousal.pth')
    p_eval.add_argument('--scaler', default='models/binary/eegnet_scaler.pkl')
    p_eval.add_argument('--split', choices=['cross_subject', 'cross_trial'], default='cross_subject')
    p_eval.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p_eval.add_argument('--mat-path', default='DREAMER.mat')
    p_eval.set_defaults(func=cmd_evaluate)

    p_eval_ens = sub.add_parser(
        'evaluate-ensemble',
        help='Evaluate ensemble of folds for valence+arousal',
    )
    p_eval_ens.add_argument(
        '--valence-report',
        default='artifacts_cv5/valence_group_kfold_subject_report.json',
    )
    p_eval_ens.add_argument(
        '--arousal-report',
        default='artifacts_cv5/arousal_group_kfold_subject_report.json',
    )
    p_eval_ens.add_argument('--top-k', type=int, default=3)
    p_eval_ens.add_argument(
        '--metric',
        choices=['balanced_accuracy', 'accuracy', 'macro_f1'],
        default='balanced_accuracy',
    )
    p_eval_ens.add_argument(
        '--split',
        choices=['cross_subject', 'cross_trial'],
        default='cross_subject',
    )
    p_eval_ens.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p_eval_ens.add_argument('--mat-path', default='DREAMER.mat')
    p_eval_ens.set_defaults(func=cmd_evaluate_ensemble)

    p_exp = sub.add_parser(
        'experiments',
        help='Run multiple train/evaluate experiments from a JSON plan',
    )
    p_exp.add_argument('--plan', default='training/experiments_plan.json')
    p_exp.add_argument('--mat-path', default='DREAMER.mat')
    p_exp.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p_exp.add_argument(
        '--eval-split',
        choices=['cross_subject', 'cross_trial'],
        default='cross_subject',
    )
    p_exp.add_argument('--ensemble-top-k', type=int, default=3)
    p_exp.add_argument('--report-out', default='artifacts/experiment_leaderboard.json')
    p_exp.set_defaults(func=cmd_experiments)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
