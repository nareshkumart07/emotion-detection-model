# Train + Streamlit (Simple Guide)

## 1) Open correct folder

```bash
cd /Users/fudode/Project/Second.zip_1/ASAP_brainwave_classification
```

If you run from another folder, you will get `No such file or directory` errors.

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Train model (no manual preprocessing needed)

Default training (recommended):

```bash
python run.py train
```

What default training does:
- trains `valence` and `arousal`
- split: `group_kfold_subject` (5-fold)
- epochs: `30`
- batch size: `64`
- preset: `balanced`
- device: `auto` (GPU if available, else CPU)

Fast test run (only 1 fold):

```bash
python run.py train --max-folds 1 --epochs 1
```

Custom training example:

```bash
python run.py train --task valence --preset robust --epochs 25
```

## 4) Run Streamlit dashboard

```bash
python run.py streamlit
```

In dashboard:
- `Quadrant BiLSTM (4-class)` for final class prediction
- `Binary EEGNet (2-class)` for valence/arousal binary prediction
- use `Load Example ...` buttons for quick testing

## 5) Run API

```bash
python run.py api
```

API docs:
- `http://127.0.0.1:8000/docs`

## 6) Evaluate trained pair

```bash
python run.py evaluate
```

## 7) Common errors

- `DREAMER.mat not found`
  - Put `DREAMER.mat` in project root folder
  - or pass: `python run.py train --mat-path /full/path/to/DREAMER.mat`

- `can't open file ... training/train_eegnet_binary.py`
  - You are in wrong folder. Run step 1 first.
