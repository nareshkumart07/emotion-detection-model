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

## Training and Evaluation

14. Which split strategies are supported?
- `cross_trial`, `cross_subject`, `loso`, and `group_kfold_subject`.

15. Which metrics do you report?
- Accuracy, balanced accuracy, and macro F1.

16. Why not rely only on accuracy?
- Class imbalance can hide poor minority-class performance.
- Balanced accuracy and macro F1 provide fairer evaluation.

17. How do you choose the best checkpoint?
- Composite validation score:
- `0.7 * balanced_accuracy + 0.3 * macro_f1`.

18. What optimization strategies are used?
- AdamW, ReduceLROnPlateau, optional focal loss, class weighting, and weighted sampling.

## Deployment and Demo

19. How can someone run your project quickly?
- `python run.py train`
- `python run.py streamlit`
- `python run.py api`
- `python run.py evaluate`

20. Which API endpoints are important for demo?
- `/health`
- `/predict/features`
- `/predict/window`
- `/predict/trial`

21. What does the Streamlit app show?
- Model selection, payload-based testing, prediction confidence, and class probabilities.

## Honest Performance Discussion

22. Is this production-ready?
- No, it is a research and coursework prototype.

23. Why is EEG emotion accuracy often moderate?
- EEG is noisy, subject variability is high, and labels are coarse.

24. How should you present current results honestly?
- The engineering pipeline is complete and reproducible.
- 4-class accuracy is moderate and indicates room for model and protocol improvement.

## Future Work

25. What are your next technical improvements?
- Run LOSO-first benchmarking with confidence intervals.
- Improve minority-class recall through focal-loss and calibration tuning.
- Add richer time-frequency features and artifact rejection.
- Explore subject adaptation and domain generalization.
- Compare with additional baselines and statistical significance tests.

26. How can this become closer to real-time usage?
- Integrate EEG device SDK input stream.
- Add low-latency preprocessing and prediction buffering.
- Benchmark end-to-end latency and robustness on continuous sessions.

## Rapid Fire

- Dataset: DREAMER
- Channels: 14
- Window size: 256 samples
- Binary threshold: valence/arousal > 3.0
- 4-class model: BiLSTM
- Binary model: EEGNet
- API framework: FastAPI
- Dashboard: Streamlit
- Scope: Academic prototype

## Suggested Viva Flow

1. State the problem and motivation.
2. Explain dataset, labels, and leakage-safe protocol.
3. Present model design and training metrics.
4. Demonstrate Streamlit and API endpoints.
5. Close with limitations and future work.
