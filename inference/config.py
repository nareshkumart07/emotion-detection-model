from pathlib import Path

# Project root (ASAP_brainwave_classification/)
ROOT = Path(__file__).resolve().parent.parent

SAMPLING_RATE = 128
NUM_CHANNELS = 14
NUM_CLASSES = 4
CHUNK_SIZE = 256
STEP_SIZE = 128

BAND_DICT = {
    'theta': [4, 8],
    'alpha': [8, 14],
    'beta': [14, 30],
    'gamma': [30, 47],
}

QUADRANT_NAMES = ['Happy', 'Stressed', 'Depressed', 'Calm']

DEFAULT_MODEL_PATH = ROOT / 'models' / 'quadrant' / 'model.pth'
DEFAULT_SCALER_PATH = ROOT / 'models' / 'quadrant' / 'scaler.pkl'
