# Python File Guide

## Main files

- `run.py`
  - easiest launcher for the main workflows
  - commands:
    - `python run.py train`
    - `python run.py streamlit`
    - `python run.py api`
    - `python run.py evaluate`

- `streamlit_app.py`
  - Streamlit dashboard for manual testing and demo
  - usually run through `python run.py streamlit`

- `app.py`
  - FastAPI entrypoint for the REST API
  - usually run through `python run.py api`

- `predict.py`
  - CLI entrypoint for direct prediction from files
  - useful for advanced or manual use

## Training files

- `training/train_eegnet_binary.py`
  - binary EEGNet training for `valence`, `arousal`, or `both`
  - preprocessing and scaling are handled inside the script

- `training/evaluate_binary_pair.py`
  - evaluates valence + arousal checkpoints together

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
