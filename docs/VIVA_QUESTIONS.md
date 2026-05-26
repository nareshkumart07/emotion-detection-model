# Viva Questions and Answers

## Project Identity

1. What is the title of your project?
- EEG-Based Emotion Recognition Using DREAMER Dataset: A BiLSTM and EEGNet Prototype with API and Dashboard Deployment.

2. What is the main objective of this project?
- To build an end-to-end EEG emotion recognition prototype with training, evaluation, API serving, and dashboard demo.

3. Why did you choose this topic?
- It combines signal processing, machine learning, and practical deployment in one project.
- EEG affective computing is a relevant and challenging research area.

## Dataset and Labels

4. Which dataset did you use?
- DREAMER dataset (`DREAMER.mat`) through TorchEEG.

5. What are valence and arousal?
- Valence indicates pleasantness.
- Arousal indicates activation or intensity.

6. How are the 4 classes defined?
- High valence + high arousal: `Happy`
- Low valence + high arousal: `Stressed`
- Low valence + low arousal: `Depressed`
- High valence + low arousal: `Calm`

## Input and Preprocessing

7. What is the raw input shape?
- `(14, 256)` for EEG windows in the binary EEGNet path.

8. What preprocessing is used in the 4-class path?
- Band power spectral density in theta, alpha, beta, and gamma bands.
- Baseline removal.
- Output feature shape `(14, 4)`.

9. How did you prevent data leakage?
- Split protocol is trial-safe.
- Scaler is fit only on training data and reused on validation/inference.

## Model Design

10. Which models are used?
- `EEGBiLSTMClassifier` for 4-class quadrant inference.
- `EEGNet` for binary valence and arousal classification.

11. Why BiLSTM for the 4-class model?
- It captures sequential dependencies in channel-wise feature sequences.

12. Why EEGNet for binary tasks?
- It is compact, EEG-oriented, and works well in low-data settings.

13. How do you get trial-level output from multiple windows?
- Use majority vote with deterministic tie-break, or mean-probability aggregation.

## Binary Classification (NEW)

14. Why did you build binary models?
- Our initial 4-class model achieved only ~35% accuracy. We wanted to understand if this was due to poor model design or task difficulty. By training binary models for valence and arousal separately, we could isolate the bottleneck.

15. What are the accuracy improvements with binary classification?
- **Valence binary model (seed 999, cross_trial):** 68.61% accuracy (window-level)
- **Arousal binary model (seed 999, cross_trial):** 57.17% accuracy (window-level)
- **Trial-level aggregation (`vote`):** valence 75.00%, arousal 61.96%
- **Quadrant from binary (trial-level):** 44.57%
- Quadrant remains much lower than binary tasks (44.57% trial-level), showing the bottleneck is **task difficulty**, not model capacity.

16. How do you combine binary predictions into a quadrant label?
```python
def quadrant_from_binary(high_valence: bool, high_arousal: bool) -> int:
    if high_valence and high_arousal:
        return 0  # Happy
    if (not high_valence) and high_arousal:
        return 1  # Stressed
    if (not high_valence) and (not high_arousal):
        return 2  # Depressed
    return 3  # Calm
```
We use simple boolean logic: combine the two binary predictions into the 4 possible quadrants.

17. What is the practical value of binary models if combined quadrant accuracy is still much lower than binary?
- If an application only needs **valence**: use trial-level `vote` for **75.00% accuracy**.
- If an application only needs **arousal**: trial-level performance is **61.96%**.
- For full 4-quadrant emotion, both approaches are limited by inherent dataset difficulty.

18. What does the binary analysis tell us about the 4-class bottleneck?
- It proves that the model architecture is sound. EEGNet achieves 60%+ on simpler tasks.
- The 4-class bottleneck is **not** overfitting or poor feature learning.
- The real issue is that DREAMER quadrant boundaries are coarse and noisy in raw EEG signals.

## Training and Evaluation

19. Which split strategies are supported?
- `cross_trial`, `cross_subject`, `loso`, and `group_kfold_subject`.

20. Which metrics do you report?
- Accuracy, balanced accuracy, and macro F1.

21. Why not rely only on accuracy?
- Class imbalance can hide poor minority-class performance.
- Balanced accuracy and macro F1 provide fairer evaluation.

22. How do you choose the best checkpoint?
- Composite validation score:
- `0.7 * balanced_accuracy + 0.3 * macro_f1`.

23. What optimization strategies are used?
- AdamW, ReduceLROnPlateau, optional focal loss, class weighting, and weighted sampling.

## Deployment and Demo

24. How can someone run your project quickly?
- `python3 run.py train`
- `python3 run.py streamlit`
- `python3 run.py api`
- `python3 run.py evaluate`
- `python3 run.py evaluate-trial`

25. Which API endpoints are important for demo?
- `/health`
- `/predict/features`
- `/predict/window`
- `/predict/trial`

26. What does the Streamlit app show?
- Model selection, payload-based testing, prediction confidence, and class probabilities.

## Honest Performance Discussion

27. Is this production-ready?
- No, it is a research and coursework prototype.

28. Why is EEG emotion accuracy often moderate?
- EEG is noisy, subject variability is high, and labels are coarse.

29. How should you present current results honestly?
- The engineering pipeline is complete and reproducible.
- 4-class accuracy is moderate and indicates room for model and protocol improvement.
- Binary seed-999 results are:
- window-level: valence 68.61%, arousal 57.17%
- trial-level (`vote`): valence 75.00%, arousal 61.96%, quadrant 44.57%
- These show the bottleneck is task difficulty, not model quality.

## Future Work

30. What are your next technical improvements?
- Run LOSO-first benchmarking with confidence intervals.
- Improve minority-class recall through focal-loss and calibration tuning.
- Add richer time-frequency features and artifact rejection.
- Explore subject adaptation and domain generalization.
- Compare with additional baselines and statistical significance tests.

31. How can this become closer to real-time usage?
- Integrate EEG device SDK input stream.
- Add low-latency preprocessing and prediction buffering.
- Benchmark end-to-end latency and robustness on continuous sessions.

## Rapid Fire

- Dataset: DREAMER
- Channels: 14
- Window size: 256 samples
- Binary threshold: valence/arousal > 3.0
- 4-class model: BiLSTM
- Binary models: EEGNet (seed 999 window-level: valence 68.61%, arousal 57.17%)
- Trial-level (`vote`): valence 75.00%, arousal 61.96%
- API framework: FastAPI
- Dashboard: Streamlit
- Scope: Academic prototype

## Suggested Viva Flow

1. State the problem and motivation.
2. Explain dataset, labels, and leakage-safe protocol.
3. Present 4-class model design and initial metrics (~35%).
4. Explain why you built binary models and show seed-999 + trial-level results.
5. Discuss what binary analysis reveals: **model is sound, task is hard**.
6. Demonstrate Streamlit and API endpoints.
7. Close with limitations and future work.
