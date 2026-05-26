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
pip install -r requirements.txt
```

Train model (requires `DREAMER.mat` in the project root or `--mat-path`):

```bash
python3 run.py train
```

Run Streamlit dashboard with the bundled pretrained models:

```bash
python3 run.py streamlit
```

Run API server:

```bash
python3 run.py api
```

Evaluate the valence + arousal model pair:

```bash
python3 run.py evaluate
```

Evaluate with trial-level aggregation (`vote` or `mean_prob`):

```bash
python3 run.py evaluate-trial --help
```

Evaluate fold-ensemble performance (recommended for stronger results):

```bash
python3 run.py evaluate-ensemble --top-k 3 --split cross_subject
```

Run multi-configuration search:

```bash
python3 run.py experiments --plan training/experiments_plan.json --eval-split cross_subject
```

Run EDA + 4-class BiLSTM training + confusion matrix artifacts:

```bash
python3 run.py eda --epochs 10 --output-dir artifacts/eda_quadrant
```

## Model Performance Summary

### 4-Class Quadrant Classification (BiLSTM)
- **Validation accuracy: ~34.6%**
- Balanced accuracy: 29.7%
- Train accuracy: 41.96%
- *Note: Weak performance reflects task difficulty; marginally above majority-class baseline.*

### Binary Classification (EEGNet)
Due to the challenge of 4-class classification, we also trained separate binary models for **valence** and **arousal** emotions:
- **Window-level (seed 999, cross-trial):**
- Valence: **68.61%** (balanced accuracy 53.38%, macro F1 0.5345)
- Arousal: **57.17%** (balanced accuracy 58.38%, macro F1 0.5534)
- **Trial-level (same model pair):**
- `mean_prob`: valence 72.83%, arousal 61.96%, quadrant 44.57%
- `vote`: valence **75.00%**, arousal **61.96%**, quadrant **44.57%**

## Important Notes

- Manual preprocessing is not required. Training handles it automatically.
- `DREAMER.mat` is needed for training and evaluation, but not for opening the Streamlit demo or API with the included model files.
- If GPU is available, `--device auto` uses GPU; otherwise CPU is used.
- The EDA + confusion-matrix pipeline is implemented in `training/eda_bilstm_quadrant.py`.
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
