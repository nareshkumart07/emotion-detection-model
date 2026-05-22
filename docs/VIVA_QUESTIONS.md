# Viva Questions and Answers (ASAP Brainwave Classification)

## A) Basic Project Understanding

1. What is the main objective of this project?
- To classify emotional state from EEG signals using DREAMER dataset.
- We implemented both:
  - 4-class quadrant model (`Happy`, `Stressed`, `Depressed`, `Calm`)
  - Binary models (`valence` low/high, `arousal` low/high)

2. Why did you choose EEG-based emotion recognition?
- EEG captures brain activity directly, which can be useful for affective computing.
- It is a strong ML use-case combining signal processing + deep learning.

3. Which dataset did you use and why?
- DREAMER dataset (`DREAMER.mat`).
- It is a known benchmark in EEG emotion recognition research.

4. What are valence and arousal?
- Valence: pleasantness (negative to positive).
- Arousal: activation level (calm to excited).

5. How do you map valence-arousal to 4 classes?
- High Valence + High Arousal -> Happy
- Low Valence + High Arousal -> Stressed
- Low Valence + Low Arousal -> Depressed
- High Valence + Low Arousal -> Calm

## B) Data and Preprocessing

6. What is the EEG input shape used in your model?
- Raw window shape is `(14 channels, 256 samples)`.

7. Did you do manual preprocessing?
- No separate manual step is needed for user.
- Pipeline does preprocessing internally during training/inference.

8. Which preprocessing is used for 4-class path?
- Band Power Spectral Density extraction for theta/alpha/beta/gamma.
- Baseline removal.
- Final feature shape `(14, 4)`.

9. Why baseline correction is important?
- It reduces subject/session bias and highlights stimulus-driven changes.

10. How do you avoid data leakage?
- Scaler is fit only on training split per fold, then applied to validation data.

## C) Model Architecture

11. Which models are implemented?
- `EEGBiLSTMClassifier` for 4-class quadrant prediction.
- `EEGNet` for binary valence/arousal prediction.

12. Why BiLSTM for quadrant model?
- It can model sequential dependencies in channel-wise feature sequences.

13. Why EEGNet for binary tasks?
- EEGNet is compact and designed for EEG signals, efficient for low-data settings.

14. What is confidence score in output?
- Probability of predicted class from softmax top value.

15. How is trial-level prediction done?
- By aggregating multiple window predictions:
  - majority vote (with deterministic tie-break)
  - mean probability strategy

## D) Training Strategy

16. What split strategies are supported?
- `cross_trial`, `cross_subject`, `loso`, `group_kfold_subject`.

17. Which split is most reliable for realistic performance?
- Subject-independent splits (`group_kfold_subject` or `loso`) are more realistic.

18. Why not trust single-fold accuracy?
- Single fold may be lucky/unlucky; multi-fold mean+std is more reliable.

19. Which metrics are reported?
- Accuracy
- Balanced Accuracy
- Macro F1
- (with mean/std across folds)

20. Why Balanced Accuracy and Macro F1 are important?
- EEG classes can be imbalanced; these metrics are fairer than plain accuracy.

21. What is early stopping and why used?
- Stops training when validation score stops improving.
- Prevents overfitting and saves compute time.

22. What optimization methods are used?
- AdamW optimizer
- ReduceLROnPlateau scheduler
- Optional focal loss, class weights, weighted sampling

23. What augmentations are used?
- Gaussian noise, channel dropout, and time masking (configurable).

24. How is best checkpoint selected?
- By composite score:
  - `0.7 * balanced_accuracy + 0.3 * macro_f1`

## E) Engineering and Deployment

25. How can a normal user run the project easily?
- Through `run.py` commands:
  - `python run.py train`
  - `python run.py streamlit`
  - `python run.py api`
  - `python run.py evaluate`

26. What does Streamlit dashboard provide?
- Interactive model selection, payload testing, confidence/probability display.

27. What API endpoints are available?
- `/health`
- `/predict/features`
- `/predict/window`
- `/predict/trial`
- `/preprocess/demo`

28. How did you make project delivery-ready?
- Clean folder structure, removed cache/artifact clutter, added documentation.

29. What files are most important for reviewer?
- `run.py`, `streamlit_app.py`, `training/train_eegnet_binary.py`, `inference/api.py`, docs.

30. How do you ensure reproducibility?
- Fixed random seed + controlled split protocol + saved reports/checkpoints.

## F) Performance and Limitations

31. Is this production-ready clinical system?
- No. It is a research/demo project for academic use.

32. Why accuracy may be moderate in EEG emotion tasks?
- High inter-subject variability, noisy signals, limited dataset size.

33. Is 75% guaranteed?
- No. It depends on split protocol and training conditions.

34. What is a realistic way to present results?
- Report protocol used, fold-wise mean+std, and limitations honestly.

35. What are current limitations?
- Moderate performance, subject variability, no real EEG device live streaming integration.

## G) Improvement Questions (Future Work)

36. What improvements can be done next?
- Better hyperparameter search
- Subject adaptation/domain generalization
- Advanced time-frequency features
- Ensemble or multitask model
- Better artifact/noise handling

37. Can this support live prediction from EEG headset?
- Yes, with additional device SDK integration and real-time preprocessing pipeline.

38. Why camera-based live test is not equivalent here?
- This model expects EEG signals, not facial/video input.

39. How can confidence calibration be improved?
- Temperature scaling or calibration on validation set.

40. How can project be extended for publication-level work?
- Compare with more baselines, strict LOSO protocol, statistical significance tests.

## H) Short 1-Line Answers (Rapid Fire)

- Dataset: DREAMER
- Input channels: 14
- Window size: 256 samples
- Binary thresholds: valence/arousal > 3.0
- Main DL models: BiLSTM + EEGNet
- Best validation selection: weighted score of balanced accuracy + macro F1
- Dashboard: Streamlit
- API framework: FastAPI
- Purpose: Academic research/demo

## I) Demo Flow for Viva (Recommended)

1. Explain objective and dataset.
2. Show project structure quickly.
3. Run `python run.py train` (or show existing report).
4. Show evaluation metrics JSON.
5. Run `python run.py streamlit` and test sample payload.
6. Show `python run.py api` and `/docs` endpoint.
7. End with limitations + future work.

---

Prepared for: College Viva / Project Defense
Project: ASAP Brainwave Classification
