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
| **Valence EEGNet** | High (>3.0) vs Low (≤3.0) | **68.61%** *(seed 999, window-level)* |
| **Arousal EEGNet** | High (>3.0) vs Low (≤3.0) | **57.17%** *(seed 999, window-level)* |

**Combining Binary Predictions into Quadrants:**
```
If Valence=HIGH + Arousal=HIGH  → Happy 😊
If Valence=LOW  + Arousal=HIGH  → Stressed 😰
If Valence=LOW  + Arousal=LOW   → Depressed 😔
If Valence=HIGH + Arousal=LOW   → Calm 😌
```

**Key Insight:** Even after binary decomposition, quadrant accuracy stays much lower (about 44.57% at trial-level) than single-dimension binary scores. This tells us the bottleneck is **task difficulty and label granularity**, not simply model architecture.

**Practical Value:**
- If your application only needs **valence** → trial-level `vote` reaches **75.00%**
- If your application only needs **arousal** → trial-level is about **61.96%**
- For full 4-quadrant emotion, performance remains limited (about 44.57% trial-level in our setup).

## Slide 6: Training and Evaluation

We support cross-trial, cross-subject, LOSO, and group-k-fold by subject. We report:
- Accuracy
- Balanced Accuracy
- Macro F1

Balanced metrics are important because EEG class distributions are not perfectly balanced.

Seed-999 metrics used in this presentation:
- Window-level: valence 68.61%, arousal 57.17%
- Trial-level (`vote`): valence 75.00%, arousal 61.96%, quadrant 44.57%

## Slide 7: Deployment

We provide:
- `python3 run.py train`
- `python3 run.py evaluate`
- `python3 run.py evaluate-trial`
- `python3 run.py streamlit`
- `python3 run.py api`

API endpoints include `/health`, `/predict/features`, `/predict/window`, and `/predict/trial`.

## Slide 8: Demo Plan

In demo, we show:
1. Streamlit prediction using example payloads.
2. API docs at `/docs`.
3. Prediction outputs with class probabilities and confidence.

This demonstrates both usability and engineering completeness.

## Slide 9: Results and Limitations

Current 4-class performance is moderate (~35%). This is expected in EEG emotion recognition due to subject variability, signal noise, and coarse label boundaries.

However, our **binary models reach 68.61%/57.17% window-level and 75.00%/61.96% trial-level**, showing that:
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
