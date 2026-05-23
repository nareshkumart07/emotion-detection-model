# Project Title, Abstract, and Objectives

## Recommended Project Title

**EEG-Based Emotion Recognition Using DREAMER Dataset: A BiLSTM and EEGNet Prototype with API and Dashboard Deployment**

## Short Title (Slide Friendly)

**EEG Emotion Detection Prototype**

## Abstract (Submission Version)

This project presents an EEG-based emotion recognition prototype using the DREAMER dataset and a complete machine learning deployment workflow. The system predicts emotional states using two complementary approaches: a 4-class valence-arousal quadrant model (`Happy`, `Stressed`, `Depressed`, `Calm`) and binary EEGNet classifiers for valence and arousal. The pipeline includes preprocessing, model training, evaluation, REST API inference through FastAPI, and an interactive Streamlit dashboard for demonstration.

Methodologically, the work emphasizes data-leakage prevention through trial-safe splitting and train-only normalization. The deployment layer supports inference from precomputed features, single EEG windows, and multi-window trials.

**Performance Summary:**
- **4-class quadrant model (BiLSTM):** ~34.6% validation accuracy—moderate and reflecting the known difficulty of EEG emotion classification and cross-subject variability.
- **Binary models (EEGNet, seed 999, cross-trial):** 68.61% valence accuracy and 57.17% arousal accuracy at window level.
- **Trial-level aggregation (same model pair):** up to 75.00% valence and 61.96% arousal with trial window aggregation; quadrant from binary reaches 44.57%.

The engineering pipeline is complete, reproducible, and suitable for academic demonstration and iterative research. Binary classification is presented as a practical alternative when individual valence or arousal prediction is more important than 4-way classification.

This project is positioned as a research and coursework prototype rather than a clinical or production-ready emotion diagnosis system.

## Problem Statement

Given 14-channel EEG signals from DREAMER, classify emotional state in the valence-arousal space either as:
- a 4-class quadrant label (`Happy`, `Stressed`, `Depressed`, `Calm`), or
- binary valence/arousal states (low or high).

## Objectives

1. Build a reproducible end-to-end EEG emotion classification pipeline.
2. Train and evaluate binary EEGNet models for valence and arousal.
3. Support 4-class quadrant inference with a BiLSTM-based model.
4. Provide practical deployment interfaces via CLI, API, and Streamlit.
5. Document methodology, limitations, and realistic interpretation of results.

## Scope and Limitations

- This work is for academic research and demonstration.
- The model output should not be treated as clinical or psychological diagnosis.
- Reported 4-class accuracy is moderate and sensitive to protocol and subject variability.

## Future Work (Recommended to Mention in Report/Presentation)

1. Run stricter LOSO-first benchmarking and report confidence intervals.
2. Improve minority-class recall with focal-loss tuning and calibration.
3. Add stronger time-frequency feature pipelines and artifact rejection.
4. Explore subject adaptation and domain generalization strategies.
5. Compare against additional baselines and publish statistical tests.
6. Integrate real-time EEG headset streaming with latency benchmarking.
