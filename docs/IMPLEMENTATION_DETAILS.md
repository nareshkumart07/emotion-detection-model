# Complete Implementation Details

## 1) Project Objective

This project implements EEG-based emotion classification on the DREAMER dataset with two complementary model paths:

- 4-class quadrant inference:
  - classes: `Happy`, `Stressed`, `Depressed`, `Calm`
  - model family: BiLSTM
  - inference inputs: engineered features or raw window+baseline

- Binary inference/training:
  - tasks: `valence` (low/high), `arousal` (low/high)
  - model family: EEGNet
  - input: raw EEG window `(14, 256)`

The repository is organized as a delivery-ready package for:

- training
- evaluation
- API deployment
- Streamlit demo

## 2) What Has Been Implemented

### 2.1 End-to-end workflows

- Binary training pipeline with:
  - subject-aware cross-validation (`group_kfold_subject`)
  - cross-subject, cross-trial, and LOSO protocols
  - fold-wise checkpoint saving and JSON reporting
- Pair evaluation for valence+arousal checkpoints
- 4-class inference through API, CLI, and Streamlit
- Streamlit testing UI for both model families

### 2.2 Model and inference improvements

- Added confidence score in prediction response
- Added scaler feature-shape mismatch validation
- Added deterministic tie-break logic in trial voting using mean probabilities
- Added strict request shape validation in FastAPI

### 2.3 Training improvements

- Default training configuration for easy run:
  - task `both`
  - split `group_kfold_subject`
  - 5 folds, 30 epochs
- Preset-based tuning:
  - `baseline`, `balanced`, `robust`, `aggressive`, `custom`
- Loss options:
  - Cross entropy
  - Focal loss
- Imbalance handling:
  - class weights
  - weighted random sampler
- Regularization and stability:
  - label smoothing
  - gradient clipping
  - noise/channel-drop/time-mask augmentation
  - early stopping + ReduceLROnPlateau scheduler

### 2.4 Delivery and usability improvements

- Cleaned project structure (removed heavy cache/artifact folders from delivery folder)
- Added one-command launcher `run.py`:
  - `train`, `streamlit`, `api`, `evaluate`
- Simplified docs for normal user execution

## 3) End-to-End Data and Prediction Flow

### 3.1 Quadrant BiLSTM path (4-class)

1. Input can be:
   - precomputed features `(14, 4)`, or
   - raw EEG window `(14, 256)` plus baseline `(14, T>=128)`
2. Raw path uses preprocessing:
   - band power spectral density (theta/alpha/beta/gamma)
   - baseline removal
3. Features are scaled using saved scaler
4. BiLSTM predicts logits for 4 classes
5. Softmax probabilities + top label + confidence are returned

### 3.2 Binary EEGNet path

1. Input window shape `(14, 256)` is validated
2. Flatten + scale using fold/model scaler
3. EEGNet predicts 2-class logits
4. Softmax gives low/high class probability and confidence

### 3.3 Trial aggregation path

- For multi-window trial prediction:
  - `vote`: majority class with deterministic tie-break by average probability
  - `mean_prob`: direct average of per-window probability vectors

## 4) Training Pipeline Design

### 4.1 Dataset and labels

- Dataset: `DREAMER.mat` through TorchEEG `DREAMERDataset`
- Windowing:
  - chunk size: 256
  - step size: 128
- Binary targets:
  - valence threshold: `> 3.0` -> high
  - arousal threshold: `> 3.0` -> high

### 4.2 Split modes

- `cross_trial`
- `cross_subject`
- `loso` (Leave-One-Subject-Out)
- `group_kfold_subject` (GroupKFold on subject_id)

### 4.3 Fold training

Per fold:

1. Fit scaler on training windows only
2. Build train/val binary datasets
3. Create dataloaders with optional weighted sampler
4. Train EEGNet with chosen preset/loss
5. Select best epoch by:
   - `selection_score = 0.7 * balanced_accuracy + 0.3 * macro_f1`
6. Early stop by patience
7. Save:
   - best checkpoint
   - scaler
   - per-epoch history JSON
8. Compute final fold metrics and confusion matrix

### 4.4 Reports and outputs

- Fold outputs:
  - `artifacts_cv5/<task>/<fold>/eegnet_<task>_best.pth`
  - `artifacts_cv5/<task>/<fold>/<task>_scaler.pkl`
  - `artifacts_cv5/<task>/<fold>/<task>_history.json`
- Global report:
  - `artifacts_cv5/<task>_<split>_report.json`
- Summary metrics:
  - mean/std of accuracy, balanced_accuracy, macro_f1

## 5) API and Dashboard Coverage

### 5.1 FastAPI endpoints

- `GET /health`
- `POST /predict/features`
- `POST /predict/window`
- `POST /predict/trial`
- `GET /preprocess/demo`

The API validates shapes before prediction and returns consistent response:

- `label_id`
- `label_name`
- `probabilities`
- `confidence`

### 5.2 Streamlit dashboard capabilities

- Model family switch:
  - Quadrant BiLSTM (4-class)
  - Binary EEGNet (2-class)
- Quadrant input modes:
  - Features
  - Window + Baseline
  - Trial (multi-window)
- Binary input mode:
  - raw window JSON
- Built-in example payload loading
- Probability visualization and JSON output

## 6) File-by-File Detailed Explanation

### Root-level files

- `app.py`
  - FastAPI entrypoint that exposes `app` from `inference.api`
  - server run target: `uvicorn app:app --reload`

- `predict.py`
  - CLI entrypoint
  - calls `inference.cli.main()`

- `run.py`
  - user-friendly launcher to avoid long commands
  - subcommands:
    - `train`
    - `streamlit`
    - `api`
    - `evaluate`
  - normal users should prefer this file

- `streamlit_app.py`
  - full dashboard implementation
  - loads models/scalers
  - supports both 4-class and binary paths
  - handles payload parsing, validation, prediction rendering

- `requirements.txt`
  - package dependencies for training, inference, API, dashboard

- `DREAMER.mat`
  - required dataset file for training/evaluation scripts

- `.gitignore`
  - excludes caches, local artifacts, and local raw data from git tracking

### `inference/` package

- `inference/config.py`
  - constants:
    - channels, chunk size, class names, band ranges
  - default model/scaler paths for quadrant inference

- `inference/model_arch.py`
  - model definitions:
    - `EEGBiLSTMClassifier` for 4-class quadrant prediction
    - `EEGNet` for binary prediction/training

- `inference/preprocess.py`
  - raw EEG preprocessing helpers
  - uses TorchEEG transforms:
    - `BandPowerSpectralDensity`
    - `BaselineRemoval`
  - validates EEG and baseline shapes
  - returns `(14, 4)` feature map

- `inference/predictor.py`
  - core inference class `EmotionPredictor`
  - loads BiLSTM + scaler
  - methods:
    - `predict_features`
    - `predict_window`
    - `predict_trial`
  - returns unified `PredictionResult` with confidence

- `inference/api.py`
  - FastAPI route definitions
  - pydantic request/response schemas
  - strict shape validation
  - model preload via app lifespan and cached predictor

- `inference/cli.py`
  - command-line prediction logic
  - supports:
    - feature file
    - raw eeg+baseline file
    - dataset smoke index mode

- `inference/torcheeg_patch.py`
  - applies TorchEEG LMDB-related compatibility patch used during runs

- `inference/__init__.py`
  - package marker

### `training/` package

- `training/train_eegnet_binary.py`
  - main production-style trainer for binary tasks
  - key features:
    - multiple split protocols
    - presets and advanced flags
    - weighted sampling and class weighting
    - focal loss option
    - augmentations and grad clipping
    - early stopping and LR scheduler
    - fold metrics and JSON reporting

- `training/evaluate_binary_pair.py`
  - loads valence and arousal checkpoints together
  - evaluates each binary task and mapped 4-class quadrant metrics
  - prints report JSON

- `training/select_best_checkpoint.py`
  - reads report JSON
  - selects best fold by metric (`balanced_accuracy` default)
  - outputs selected checkpoint/scaler path

### `models/` folder

- `models/quadrant/model.pth`
  - pretrained weights for 4-class BiLSTM inference

- `models/quadrant/scaler.pkl`
  - scaler for 4-class feature input

- `models/binary/eegnet_valence.pth`
  - binary valence checkpoint

- `models/binary/eegnet_arousal.pth`
  - binary arousal checkpoint

- `models/binary/eegnet_scaler.pkl`
  - scaler for binary EEGNet input

### `examples/` folder

- `examples/dummy_features.json`
  - sample payload for feature-based 4-class prediction

- `examples/dummy_window.json`
  - sample payload for window-based prediction (used in both families)

- `examples/dummy_trial.json`
  - sample payload for multi-window trial inference

### `docs/` folder

- `docs/PROJECT_REPORT.md`
  - project report and narrative summary

- `docs/TRAIN_AND_STREAMLIT.md`
  - practical run instructions for users

- `docs/FILE_GUIDE.md`
  - quick "which file does what" guide

- `docs/IMPLEMENTATION_DETAILS.md`
  - this complete technical implementation documentation

## 7) What Is Not Included in This Delivery Build

- No large cache folders (`torcheeg_cache_*`)
- No temporary training artifact dumps
- No notebook-heavy exploratory files

These can be generated again during local training runs.

## 8) Current Scope and Limitations

- Research/demo scope, not medical or clinical product
- Accuracy depends strongly on split protocol and subject variability
- Binary models and quadrant model are separate paths (not a single unified multitask net)

## 9) Recommended Use for Review or Handover

For reviewer/client demo sequence:

1. `python run.py train`
2. `python run.py evaluate`
3. `python run.py streamlit`
4. `python run.py api` and open `/docs`

This sequence demonstrates training, metrics, interactive prediction, and deployable API.
