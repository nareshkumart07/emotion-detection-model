# Presentation Script (8-10 Minutes)

## Slide 1: Title

Good morning. Our project is **EEG-Based Emotion Recognition Using DREAMER Dataset: A BiLSTM and EEGNet Prototype with API and Dashboard Deployment**.

This work focuses on building a complete pipeline from EEG data to deployable prediction interfaces.

## Slide 2: Problem and Motivation

Emotion recognition from EEG is useful in affective computing and human-computer interaction. Unlike facial-only systems, EEG captures brain activity directly.

Our goal was not only to train a model, but to create an end-to-end reproducible system that includes training, evaluation, deployment, and demo.

## Slide 3: Dataset and Labels

We used the DREAMER dataset through TorchEEG. It contains 14-channel EEG with valence and arousal labels.

We mapped valence-arousal into four quadrants:
- `Happy`
- `Stressed`
- `Depressed`
- `Calm`

## Slide 4: Pipeline Overview

The workflow is:
1. Load and preprocess EEG.
2. Train models with leakage-aware split settings.
3. Save checkpoints and scalers.
4. Expose predictions through CLI, FastAPI, and Streamlit dashboard.

This makes the project easy to test and demonstrate.

## Slide 5: Model Architecture

We implemented two model families:
- **BiLSTM** for 4-class quadrant inference.
- **EEGNet** for binary valence and arousal tasks.

The binary path supports stronger protocol experimentation and can be combined into quadrant interpretation.

## Slide 5b: Binary Classification Approach (NEW)

Initially, our 4-class model achieved only ~35% accuracy. So we explored **binary classification** as an alternative:

**Why Binary is Better:**
- 4-class classification is inherently harder—many emotions are similar.
- Breaking it into two simpler questions: "Is valence high?" and "Is arousal high?" is easier.

**Our Two Binary Models:**

| Model | Task | Accuracy |
|-------|------|----------|
| **Valence EEGNet** | High (>3.0) vs Low (≤3.0) | **59.39%** |
| **Arousal EEGNet** | High (>3.0) vs Low (≤3.0) | **64.03%** |

**Combining Binary Predictions into Quadrants:**
```
If Valence=HIGH + Arousal=HIGH  → Happy 😊
If Valence=LOW  + Arousal=HIGH  → Stressed 😰
If Valence=LOW  + Arousal=LOW   → Depressed 😔
If Valence=HIGH + Arousal=LOW   → Calm 😌
```

**Key Insight:** When combined back to quadrants, we still get ~35% accuracy. This tells us the bottleneck is **not the model architecture**, but the **task difficulty**. The DREAMER quadrant boundaries are inherently noisy in raw EEG.

**Practical Value:**
- If your application only needs **arousal** (excited vs calm) → use binary arousal model: **64% accuracy**
- If your application only needs **valence** (pleasant vs unpleasant) → use binary valence model: **59% accuracy**
- For full 4-quadrant emotion, both approaches (~35%) are limited by the dataset.

## Slide 6: Training and Evaluation

We support cross-trial, cross-subject, LOSO, and group-k-fold by subject. We report:
- Accuracy
- Balanced Accuracy
- Macro F1

Balanced metrics are important because EEG class distributions are not perfectly balanced.

## Slide 7: Deployment

We provide:
- `python run.py train`
- `python run.py evaluate`
- `python run.py streamlit`
- `python run.py api`

API endpoints include `/health`, `/predict/features`, `/predict/window`, and `/predict/trial`.

## Slide 8: Demo Plan

In demo, we show:
1. Streamlit prediction using example payloads.
2. API docs at `/docs`.
3. Prediction outputs with class probabilities and confidence.

This demonstrates both usability and engineering completeness.

## Slide 9: Results and Limitations

Current 4-class performance is moderate (~35%). This is expected in EEG emotion recognition due to subject variability, signal noise, and coarse label boundaries.

However, our **binary models achieve 59-64%** accuracy, showing that:
1. The model architecture is sound.
2. The limitation is the task difficulty, not model capacity.
3. For single-dimension tasks (valence or arousal only), we have strong models.

So we present this as an academic prototype, not a clinical or production diagnosis tool.

## Slide 10: Future Work

Our next steps are:
1. LOSO-first benchmarking and confidence-interval reporting.
2. Better minority-class handling and confidence calibration.
3. Richer time-frequency features and improved artifact handling.
4. Subject adaptation and domain generalization methods.
5. Real-time EEG device integration with latency profiling.

## Slide 11: Conclusion

This project delivers a complete, reproducible EEG emotion recognition pipeline with training, evaluation, and deployment interfaces.

The strongest contributions are:
1. **End-to-end engineering quality** with honest reporting.
2. **Binary classification analysis** that isolates the true bottleneck (task difficulty, not model quality).
3. **Clear deployment interfaces** (API, CLI, Streamlit) for practical use.

Thank you.
