# Train + Streamlit Guide

## 1) Open the correct folder

```bash
cd <project-folder>
```

If you run commands from another folder, you may get file path errors.

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Train the model

Default training:

```bash
python run.py train
```

Default training does this:
- trains `valence` and `arousal`
- uses `group_kfold_subject` with 5 folds
- runs for 30 epochs
- uses batch size 64
- uses the `balanced` preset
- uses `auto` device selection

Fast test run:

```bash
python run.py train --max-folds 1 --epochs 1
```

Custom training example:

```bash
python run.py train --task valence --preset robust --epochs 25
```

## 4) Run the Streamlit dashboard

```bash
python run.py streamlit
```

In the dashboard:
- use `Quadrant BiLSTM (4-class)` for final emotion class prediction
- use `Binary EEGNet (2-class)` for valence/arousal binary prediction
- use the `Load Example ...` buttons for quick testing

## 5) Run the API

```bash
python run.py api
```

API docs:
- `http://127.0.0.1:8000/docs`

## 6) Evaluate a trained pair

```bash
python run.py evaluate
```

For higher scoring potential, evaluate fold ensemble:

```bash
python run.py evaluate-ensemble --top-k 3 --split cross_subject
```

## 7) Run multi-experiment search

```bash
python run.py experiments --plan training/experiments_plan.json --eval-split cross_subject
```

This produces a ranked report at:
- `artifacts/experiment_leaderboard.json`

## 8) Common errors

- `DREAMER.mat not found`
  - Put `DREAMER.mat` in the project root folder
  - or pass `python run.py train --mat-path /full/path/to/DREAMER.mat`

- `can't open file ... training/train_eegnet_binary.py`
  - You are in the wrong folder
  - Run step 1 first
