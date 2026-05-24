# Emotion Detection Model

EEG emotion classification project with:
- 4-class inference (`Happy`, `Stressed`, `Depressed`, `Calm`)
- binary EEGNet training (`valence`, `arousal`)
- Streamlit dashboard demo
- FastAPI inference API

## Project Structure

```text
emotion-detection-model/
|-- app.py
|-- predict.py
|-- run.py                # launcher for train / streamlit / api / evaluate
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- inference/
|-- training/
|-- models/
|   |-- quadrant/
|   `-- binary/
|-- examples/
`-- docs/
```

## For Normal User

Go to the project folder first:

```bash
cd <project-folder>
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train model (requires `DREAMER.mat` in the project root or `--mat-path`):

```bash
python run.py train
```

Train notebook-equivalent 4-class quadrant BiLSTM (saves `model.pth`, `scaler.pkl`, metrics, and confusion matrix plot):

```bash
python run.py train-quadrant
```

Run Streamlit dashboard with the bundled pretrained models:

```bash
python run.py streamlit
```

Run API server:

```bash
python run.py api
```

Evaluate the valence + arousal model pair:

```bash
python run.py evaluate
```

Save evaluation report (now includes confusion matrices):

```bash
python run.py evaluate --report-out artifacts/eval_binary_pair.json
```

Evaluate fold-ensemble performance (recommended for stronger results):

```bash
python run.py evaluate-ensemble --top-k 3 --split cross_subject
```

Save ensemble evaluation report (includes confusion matrices):

```bash
python run.py evaluate-ensemble --top-k 3 --split cross_subject --report-out artifacts/eval_binary_ensemble.json
```

Run multi-configuration search:

```bash
python run.py experiments --plan training/experiments_plan.json --eval-split cross_subject
```

## Model Performance Summary

### 4-Class Quadrant Classification (BiLSTM)
- **Validation accuracy: ~34.6%**
- Balanced accuracy: 29.7%
- Train accuracy: 41.96%
- *Note: Weak performance reflects task difficulty; marginally above majority-class baseline.*

### Binary Classification (EEGNet)
Due to the challenge of 4-class classification, we also trained separate binary models for **valence** and **arousal** emotions:
- **Valence binary accuracy: ~59.4%**
- **Arousal binary accuracy: ~64.0%**
- *Note: Binary tasks are significantly easier than 4-way classification.*

## Important Notes

- Manual preprocessing is not required. Training handles it automatically.
- `DREAMER.mat` is needed for training and evaluation, but not for opening the Streamlit demo or API with the included model files.
- If GPU is available, `--device auto` uses GPU; otherwise CPU is used.
- Advanced commands are documented in [docs/TRAIN_AND_STREAMLIT.md](docs/TRAIN_AND_STREAMLIT.md).
- Python file roles are documented in [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md).
- Full technical details are documented in [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md).
- Viva prep Q&A is available at [docs/VIVA_QUESTIONS.pdf](docs/VIVA_QUESTIONS.pdf).

## Submission Support Docs

- Project title, abstract, objectives: [docs/ABSTRACT_AND_OBJECTIVES.md](docs/ABSTRACT_AND_OBJECTIVES.md)
- Viva Q&A sheet: [docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md)
- Presentation script: [docs/PRESENTATION_SCRIPT.md](docs/PRESENTATION_SCRIPT.md)
- Full report with limitations and future work: [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)
- Accuracy optimization workflow: [docs/ACCURACY_OPTIMIZATION.md](docs/ACCURACY_OPTIMIZATION.md)
