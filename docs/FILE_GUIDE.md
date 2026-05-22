# Python File Guide (What to Run)

## Main files (normal user)

- `run.py`
  - easiest launcher for all important actions
  - commands:
    - `python run.py train`
    - `python run.py streamlit`
    - `python run.py api`
    - `python run.py evaluate`

- `streamlit_app.py`
  - Streamlit dashboard UI for manual testing and demo
  - usually run via `python run.py streamlit`

- `app.py`
  - FastAPI entrypoint for REST API
  - usually run via `python run.py api`

- `predict.py`
  - CLI entrypoint for direct prediction from files
  - advanced/manual use

## Training files

- `training/train_eegnet_binary.py`
  - binary EEGNet training (`valence`, `arousal`, or `both`)
  - preprocessing/scaling is handled inside this script

- `training/evaluate_binary_pair.py`
  - evaluates valence + arousal checkpoints together

- `training/select_best_checkpoint.py`
  - selects best fold checkpoint from report JSON

## Inference package

- `inference/predictor.py`
  - core prediction logic and model loading

- `inference/api.py`
  - request/response validation + API routes

- `inference/cli.py`
  - CLI implementation used by `predict.py`

- `inference/model_arch.py`
  - model definitions (BiLSTM and EEGNet)

- `inference/preprocess.py`
  - signal-to-feature pipeline for quadrant model

- `inference/config.py`
  - constants and default model/scaler paths

## Data and models

- `DREAMER.mat`
  - required for training

- `models/quadrant/*`
  - pretrained 4-class model

- `models/binary/*`
  - pretrained binary valence/arousal models
