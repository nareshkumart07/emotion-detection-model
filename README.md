# ASAP Brainwave Classification (Delivery Package)

EEG emotion classification project with:
- 4-class inference (`Happy`, `Stressed`, `Depressed`, `Calm`)
- binary EEGNet training (`valence`, `arousal`)
- Streamlit dashboard demo
- FastAPI inference API

## Clean Folder Structure

```text
ASAP_brainwave_classification/
├── app.py
├── predict.py
├── run.py                          # easiest launcher (train / streamlit / api / evaluate)
├── streamlit_app.py
├── requirements.txt
├── README.md
├── DREAMER.mat                     # dataset file for training (required)
├── inference/
├── training/
├── models/
│   ├── quadrant/
│   └── binary/
├── examples/
└── docs/
```

## For Normal User (Recommended)

Go to project folder first:

```bash
cd /Users/fudode/Project/Second.zip_1/ASAP_brainwave_classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model (single command, no preprocessing step needed):

```bash
python run.py train
```

Run Streamlit dashboard:

```bash
python run.py streamlit
```

Run API server:

```bash
python run.py api
```

Evaluate valence+arousal model pair:

```bash
python run.py evaluate
```

## Important Notes

- Manual preprocessing is not required. Training script handles it automatically.
- If GPU is available, `--device auto` uses GPU; otherwise CPU is used.
- Advanced commands are documented in [docs/TRAIN_AND_STREAMLIT.md](docs/TRAIN_AND_STREAMLIT.md).
- Python file roles are documented in [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md).
- Full technical implementation details are documented in [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md).
- Viva prep Q&A (PDF) is available at [docs/VIVA_QUESTIONS.pdf](docs/VIVA_QUESTIONS.pdf).
