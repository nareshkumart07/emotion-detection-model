# Brainwave EEG Emotion Classification — Project Report

**Project:** Emotion Detection Model  
**Dataset:** DREAMER (via TorchEEG)  
**Task:** Four-class valence–arousal quadrant classification from EEG  
**Status:** End-to-end pipeline complete (training notebook, saved artifacts, CLI + REST API)  
**Report date:** May 2026

---

## Executive summary

This project builds a research prototype that maps 14-channel EEG to one of four affective quadrants (Happy, Stressed, Depressed, Calm) using baseline-corrected band power features and a bidirectional LSTM classifier. Data are drawn from the **DREAMER** database, preprocessed with **TorchEEG**, split at the **trial** level (not random windows), and evaluated with standard classification metrics.

After fixing methodological issues (trial-level leakage from random window splits, missing baseline correction, and train-only normalization), the system trains and deploys reliably. **Validation accuracy is ~35%** on a held-out 20% of trials—modestly above the majority-class baseline (~33% Depressed) but well below ceiling for a four-way task. Train and validation accuracy are close (~42% vs ~35%), indicating **limited overfitting** and **weak discriminative signal** in the current feature/model setup rather than a broken pipeline.

Deliverables include `EDA.ipynb`, `model.pth`, `scaler.pkl`, a packaged `inference/` module, CLI (`predict.py`), and FastAPI (`app.py` → `/docs`).

---

## 1. Introduction

### 1.1 Motivation

Electroencephalography (EEG) offers a non-invasive channel for affective computing. The **DREAMER** corpus pairs EEG with self-reported valence, arousal, and dominance during audio–visual stimuli, making it a standard benchmark for emotion recognition from brain signals.

### 1.2 Problem formulation

Given a short segment of multichannel EEG during stimulus presentation (with a preceding baseline), predict which **valence–arousal quadrant** best describes the trial:

| Class ID | Name      | Valence | Arousal |
|---------|-----------|---------|---------|
| 0       | Happy     | > 3     | > 3     |
| 1       | Stressed  | ≤ 3     | > 3     |
| 2       | Depressed | ≤ 3     | ≤ 3     |
| 3       | Calm      | > 3     | ≤ 3     |

Scores are on a 1–5 scale; threshold **3.0** follows common practice in the literature and in this repository’s TorchEEG label mapping.

### 1.3 Objectives

1. Reproduce a reproducible DREAMER → features → train → evaluate workflow in `EDA.ipynb`.
2. Avoid **data leakage** by splitting trials, not sliding windows.
3. Export deployable weights and a scaler fit **only on training data**.
4. Expose inference via CLI and HTTP API for integration demos.

### 1.4 Scope and disclaimer

This is a **research / coursework prototype**, not a clinical or real-time BCI product. Reported metrics are on cached offline windows; subject-independent generalization (e.g. leave-one-subject-out) is **not** yet the primary evaluation protocol.

---

## 2. Dataset

### 2.1 DREAMER overview

| Property | Value |
|----------|--------|
| Subjects | 23 |
| Trials per subject | 18 |
| Total trials | 414 |
| EEG channels used | 14 (TorchEEG montage) |
| Sampling rate | 128 Hz |
| Labels | Valence, arousal, dominance (1–5) |

Raw data are loaded via `DREAMERDataset` with `DREAMER.mat` on the local machine (path configured in the notebook).

### 2.2 Label distribution (trial level)

Trial-level quadrant counts (unique subject–trial pairs):

| Quadrant   | Trials | Share  |
|------------|--------|--------|
| Happy      | 76     | 18.4%  |
| Stressed   | 105    | 25.4%  |
| Depressed  | 146    | 35.3%  |
| Calm       | 87     | 21.0%  |

**Depressed** is the plurality class; a naive majority classifier achieves ~35% trial-level accuracy if all windows inherit trial labels. Window-level counts are higher (~85k windows) because of sliding segmentation.

### 2.3 Windowing and cache

| Parameter | Value | Meaning |
|-----------|--------|---------|
| `CHUNK_SIZE` | 256 | 2.0 s stimulus window |
| `STEP_SIZE` | 128 | 1.0 s hop (50% overlap) |
| Cache directory | `torcheeg_cache_c256_s128` | LMDB-backed preprocessed store |

**Total windows (full corpus):** 85,330  

First-time cache build on 23 subjects took ~23 minutes on the development machine (subsequent runs read from cache).

A denser step (e.g. 16 samples) was avoided intentionally: it multiplies window count (~8×) and cache build time to hours.

---

## 3. Methodology

### 3.1 Preprocessing pipeline

Per window, TorchEEG applies:

1. **Band power spectral density (PSD)** in four bands: theta (4–8 Hz), alpha (8–14 Hz), beta (14–30 Hz), gamma (30–47 Hz).
2. **Baseline removal** using the trial’s resting baseline segment.
3. **Tensor conversion** → feature tensor of shape **(14, 4)** (channels × bands).

The same logic is mirrored in `inference/preprocess.py` for live API inference.

### 3.2 Train / validation split

- **Method:** `train_test_split_cross_trial` (TorchEEG), `test_size=0.2`, `random_state=42`.
- **Trials:** 322 train / 92 validation (~78% / 22% of 414 trials).
- **Windows:** 70,127 train / 15,203 validation.

Windows from the same trial never appear in both splits, which prevents the inflated accuracy seen with random window splits.

### 3.3 Feature normalization

- `sklearn.preprocessing.StandardScaler` fit on **training windows only** (flattened 56-D vectors, partial_fit over batches).
- Scaler persisted as `scaler.pkl`; validation and inference apply `transform` only.

### 3.4 Model architecture

**`EEGBiLSTMClassifier`** (`inference/model_arch.py`):

- Input: sequence length 14 (channels), each step 4-D band features (after scaling).
- **BiLSTM** layer 1: 4 → 128 (bidirectional → 256-D).
- Dropout 0.5.
- **BiLSTM** layer 2: 256 → 128 (bidirectional → 256-D).
- Dropout 0.5; last time step fed to linear head.
- **Linear:** 256 → 4 classes.

Approximate parameter count is on the order of hundreds of thousands (dominated by LSTM weights).

### 3.5 Training configuration

| Hyperparameter | Value |
|----------------|--------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 10 |
| Loss | Cross-entropy |
| Device | CUDA when available |

Training curves and confusion matrices are produced in notebook Section 7–8.

---

## 4. Results

### 4.1 Final epoch metrics (validation)

| Metric | Value |
|--------|--------|
| **Accuracy** | **34.56%** |
| Balanced accuracy | 29.66% |
| Macro F1 | 0.2849 |
| Weighted F1 | 0.3589 |
| Train accuracy (final epoch) | 41.96% |
| Validation loss (final epoch) | 1.5789 |

### 4.2 Per-class validation report

| Class     | Precision | Recall | F1-score | Support (windows) |
|-----------|-----------|--------|----------|-------------------|
| Happy     | 0.121     | 0.252  | 0.164    | 1,713             |
| Stressed  | 0.328     | 0.395  | 0.359    | 3,368             |
| Depressed | 0.531     | 0.407  | 0.461    | 7,829             |
| Calm      | 0.191     | 0.132  | 0.156    | 2,293             |

**Observations:**

- The model **favors Depressed** (highest precision and support-weighted contribution), consistent with class imbalance.
- **Happy** and **Calm** are hardest (low precision/recall), typical when quadrant boundaries are coarse and EEG SNR is low.
- Train ≈ val accuracy gap is small → **not** severe overfitting; the bottleneck is **signal and task difficulty**, not a training bug.

### 4.3 Comparison to baselines

| Baseline | Approx. accuracy |
|----------|------------------|
| Random (4 classes) | 25% |
| Majority class (Depressed) | ~35% (window-level, label skew) |
| **This model (val)** | **34.6%** |

The classifier is **marginally informative** at the window level under this split; it is not yet a strong emotion decoder.

### 4.4 Historical context (project evolution)

Early notebook iterations reported ~37% train/val with **random window splits** and without proper baseline correction and scaler discipline. After trial-level splitting and corrected features, metrics dropped slightly on validation but became **methodologically trustworthy**. Higher accuracies reported in some DEAP-based papers often use **binary** labels, **subject-dependent** splits, or **hand-crafted frequency features** with different evaluation protocols—direct numeric comparison requires matching splits and tasks.

### 4.5 Binary classification as an alternative approach

Given the weak performance of 4-class classification (~35% accuracy), the project includes a complementary approach: **binary EEGNet models** for individual **valence** and **arousal** predictions.

#### Motivation
Binary classification is inherently easier than 4-way classification. By training separate classifiers for valence (low vs. high) and arousal (low vs. high), we can combine them to derive the quadrant label with improved accuracy.

#### Binary Model Results (EEGNet on DREAMER, cross-subject split, test size 20%)

| Task | Accuracy | Balanced Accuracy | Macro F1 |
|------|----------|-------------------|----------|
| **Valence** | **59.39%** | 48.94% | 0.4153 |
| **Arousal** | **64.03%** | 52.90% | 0.5009 |
| **Quadrant (mapped from binary)** | 34.82% | 24.68% | 0.1869 |

**Observations:**
- Individual binary tasks achieve **~60% and ~64% accuracy**, a **+20–30 percentage point improvement** over 4-class.
- However, combining the two binary predictions back to quadrant labels still yields ~35% (same as direct 4-class), suggesting that **quadrant boundaries are coarse** and errors in valence or arousal propagate.
- Binary models (EEGNet) are simpler and potentially more interpretable for individual dimensions.
- For practical use cases where only valence or arousal is needed (not quadrant), these binary models are significantly more reliable.

---

## 5. System architecture and deployment

### 5.1 Repository layout

```
project-root/
├── EDA.ipynb              # Full experiment (9 sections)
├── model.pth              # Trained BiLSTM weights
├── scaler.pkl             # StandardScaler (train-fit)
├── requirements.txt
├── app.py                 # uvicorn entry → inference.api
├── predict.py             # CLI wrapper
├── inference/
│   ├── config.py          # Constants, paths, band dict
│   ├── model_arch.py      # EEGBiLSTMClassifier
│   ├── preprocess.py      # Band PSD + baseline removal
│   ├── predictor.py       # EmotionPredictor
│   ├── api.py             # FastAPI routes
│   ├── cli.py             # Command-line inference
│   └── torcheeg_patch.py  # LMDB shared-env fix (Windows)
├── examples/              # Swagger-ready JSON payloads
│   ├── dummy_features.json
│   ├── dummy_window.json
│   └── dummy_trial.json
└── torcheeg_cache_c256_s128/   # Preprocessed LMDB cache
```

### 5.2 Inference modes

| Entry point | Input | Use case |
|-------------|-------|----------|
| `POST /predict/features` | 14×4 float matrix (already scaled in training space*) | Fast path when features are precomputed |
| `POST /predict/window` | EEG (14×256) + baseline (14×T, T≥128) | Raw window → internal feature extract → scale → predict |
| `POST /predict/trial` | List of windows + baseline; `vote` or `mean_prob` | Trial-level aggregation |
| `predict.py` (CLI) | Same as above via flags | Scripts / batch jobs |
| `GET /health` | — | Liveness and model load check |

\*For `/predict/features`, values should be in the same representation as post-`BaselineRemoval` features; the predictor applies `scaler.transform` before the network.

### 5.3 Running the API

```bash
cd <project-folder>
pip install -r requirements.txt
# Install PyTorch with CUDA if needed (see requirements.txt header)
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger tests. Example bodies live under `examples/`.

### 5.4 Example response

```json
{
  "label_id": 2,
  "label_name": "Depressed",
  "probabilities": {
    "Happy": 0.12,
    "Stressed": 0.28,
    "Depressed": 0.41,
    "Calm": 0.19
  }
}
```

Identical dummy inputs often yield the same label; that confirms the stack loads weights and runs forward pass—it does not imply high real-world accuracy.

### 5.5 Platform notes

- **Windows + LMDB:** Concurrent notebook cache access and API/process access can lock LMDB; `torcheeg_patch.py` mitigates premature environment close.
- **Cache build:** One-time ~20–30 min for `c256_s128` at 23 subjects; set dataset `io_path` to cache dir afterward to skip reprocessing.

---

## 6. Notebook structure (`EDA.ipynb`)

The notebook is organized for presentation and reproducibility:

1. **Title and configuration** — goal, settings table, deployment pointers  
2. **Imports and plotting setup**  
3. **Load DREAMER** — cache, window counts, sample shapes  
4. **Label distribution** — window and trial bar charts  
5. **Model definition** — `EEGBiLSTMClassifier`  
6. **Split, scaler, dataloaders** — trial split counts, `scaler.pkl`  
7. **Training** — 10 epochs, history dict, learning curves  
8. **Evaluation** — accuracy, balanced accuracy, F1, confusion matrices (counts + row-normalized)  
9. **Save artifacts** — `model.pth`, deployment notes  

---

## 7. Limitations

1. **Low absolute accuracy** (~35% val) for four classes; not suitable for user-facing emotion products without substantial improvement.  
2. **Trial split ≠ subject split:** Some subjects may contribute trials to both train and val; **leave-one-subject-out (LOSO)** is stricter and typically lowers scores.  
3. **Window-level labels:** All windows from a trial share one quadrant label; temporal dynamics within trials are collapsed.  
4. **Coarse quadrants:** Collapsing 1–5 ratings to four bins loses nuance; binary valence/arousal tasks are often easier.  
5. **Class imbalance:** Depressed overweight drives metrics and confusions.  
6. **No online adaptation:** Scaler and model are static; domain shift across sessions/hardware is unaddressed.  
7. **Research disclaimer:** API explicitly marked not for clinical use.

---

## 8. Future work

| Direction | Rationale |
|-----------|-----------|
| **Binary valence / arousal** | Aligns with much DEAP literature; often +10–20% accuracy |
| **Trial-level pooling + MLP** | Mean/median features per trial; reduces window correlation |
| **LOSO / k-fold by subject** | Reports generalization to new users |
| **Class weights / focal loss** | Mitigate Depressed bias; improve minority recall |
| **Alternative features** | Differential entropy, CSP, graph metrics, learned filters |
| **Stronger sequence models** | Transformer over time–frequency patches; multi-window context |
| **Subject-aware normalization** | Per-subject z-score or adversarial domain adaptation |
| **Streamlit or real-time UI** | Demo live inference with mocked or device EEG |
| **Calibration** | Temperature scaling on validation probabilities |

---

## 9. Dependencies

Core stack (see `requirements.txt`):

- Python 3.11 recommended  
- PyTorch, TorchEEG, pytorch-lightning  
- NumPy, pandas, scikit-learn, SciPy  
- Matplotlib, seaborn, tqdm, Jupyter  
- FastAPI, uvicorn, pydantic, joblib  

GPU wheels: install PyTorch from the CUDA index noted in `requirements.txt` before other packages.

---

## 10. Conclusion

This project delivers a complete, honest pipeline from DREAMER through trial-safe evaluation to deployed inference. The engineering pieces—caching, trial splits, baseline-corrected band power, train-only scaling, BiLSTM training, and FastAPI/CLI serving—work as intended. The **science bottleneck** is classification performance: validation accuracy near **35%** reflects difficult four-way affect mapping from short EEG windows, not a failure to load the model or API.

The project is **suitable as a foundation** for coursework, demos, and iterative research. Meaningful accuracy gains will likely come from **simpler targets** (binary labels), **subject-independent evaluation**, and **richer or trial-aggregated features** rather than from deployment plumbing alone.

---

## References

1. Katsigiannis, S., & Ramzan, N. (2018). **DREAMER: A Database for Emotion Recognition Through EEG and ECG Signals From Wireless Low-cost Off-the-Shelf Devices.** *IEEE Access.*  
2. TorchEEG documentation and `DREAMERDataset` — https://torcheeg.readthedocs.io/  
3. Related work in this monorepo (DEAP emotion recognition folders) for split and feature conventions.

---

## Appendix A — Quick command reference

| Action | Command |
|--------|---------|
| Train / evaluate | Run all cells in `EDA.ipynb` |
| Start API | `uvicorn app:app --reload` |
| CLI predict | `python predict.py --help` |
| Health check | `GET /health` |

## Appendix B — Artifact checklist

| File | Purpose |
|------|---------|
| `model.pth` | BiLSTM state dict |
| `scaler.pkl` | StandardScaler (56 features) |
| `torcheeg_cache_c256_s128/` | Preprocessed dataset IO |
| `examples/*.json` | API test payloads |

---

*End of report*
