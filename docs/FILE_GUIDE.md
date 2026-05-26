# Python File Guide

## Main files

- `run.py`
  - easiest launcher for the main workflows
  - commands:
    - `python3 run.py train`
    - `python3 run.py streamlit`
    - `python3 run.py api`
    - `python3 run.py evaluate`
    - `python3 run.py evaluate-trial`
    - `python3 run.py evaluate-ensemble`
    - `python3 run.py experiments`
    - `python3 run.py eda`

- `streamlit_app.py`
  - Streamlit dashboard for manual testing and demo
  - usually run through `python3 run.py streamlit`

- `app.py`
  - FastAPI entrypoint for the REST API
  - usually run through `python3 run.py api`

- `predict.py`
  - CLI entrypoint for direct prediction from files
  - useful for advanced or manual use

## Training files

- `training/train_eegnet_binary.py`
  - binary EEGNet training for `valence`, `arousal`, or `both`
  - preprocessing and scaling are handled inside the script

- `training/evaluate_binary_pair.py`
  - evaluates valence + arousal checkpoints together

- `training/evaluate_binary_pair_trial.py`
  - evaluates both window-level and trial-level performance
  - supports trial aggregation with `vote` and `mean_prob`

- `training/evaluate_binary_pair_ensemble.py`
  - evaluates top-k fold ensemble for valence+arousal pair
  - reports window/trial-level metrics and confusion matrices

- `training/run_binary_experiments.py`
  - runs multi-configuration train/evaluate plans from JSON
  - writes leaderboard-style report for comparison

- `training/eda_bilstm_quadrant.py`
  - end-to-end EDA + 4-class BiLSTM training pipeline
  - saves plots, confusion matrices, report JSON, model, and scaler

- `training/fast_accuracy_sweep.sh`
  - multi-seed accuracy sweep + top-k auto selection + ensemble evaluation

- `training/select_best_checkpoint.py`
  - selects the best fold checkpoint from a report JSON

## Inference package

- `inference/predictor.py`
  - core prediction logic and model loading

- `inference/api.py`
  - request validation and API routes

- `inference/cli.py`
  - CLI implementation used by `predict.py`

- `inference/model_arch.py`
  - model definitions for BiLSTM and EEGNet

- `inference/preprocess.py`
  - signal-to-feature pipeline for the quadrant model

- `inference/config.py`
  - constants and default model/scaler paths

## Data and models

- `DREAMER.mat`
  - required for training and evaluation
  - not required for the Streamlit demo or API if you use the bundled pretrained models

- `models/quadrant/*`
  - pretrained 4-class model files

- `models/binary/*`
  - pretrained binary valence/arousal model files
