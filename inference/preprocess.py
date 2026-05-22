import numpy as np
from torcheeg import transforms

from .config import BAND_DICT, CHUNK_SIZE, NUM_CHANNELS, SAMPLING_RATE


def _feature_transform():
    return transforms.Compose([
        transforms.BandPowerSpectralDensity(
            sampling_rate=SAMPLING_RATE,
            band_dict=BAND_DICT,
            apply_to_baseline=True,
        ),
        transforms.BaselineRemoval(),
    ])


_TRANSFORM = None


def get_feature_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = _feature_transform()
    return _TRANSFORM


def validate_eeg_window(eeg: np.ndarray, label: str = 'eeg') -> np.ndarray:
    arr = np.asarray(eeg, dtype=np.float64)
    if arr.shape != (NUM_CHANNELS, CHUNK_SIZE):
        raise ValueError(
            f'{label} must have shape ({NUM_CHANNELS}, {CHUNK_SIZE}), got {arr.shape}'
        )
    return arr


def validate_baseline(baseline: np.ndarray) -> np.ndarray:
    arr = np.asarray(baseline, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != NUM_CHANNELS:
        raise ValueError(
            f'baseline must be 2D with {NUM_CHANNELS} channels, got {arr.shape}'
        )
    if arr.shape[1] < SAMPLING_RATE:
        raise ValueError(
            f'baseline needs at least {SAMPLING_RATE} samples per channel, got {arr.shape[1]}'
        )
    return arr


def extract_features(eeg: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Baseline-corrected band PSD, shape (14, 4)."""
    eeg = validate_eeg_window(eeg)
    baseline = validate_baseline(baseline)
    result = get_feature_transform()(eeg=eeg, baseline=baseline)['eeg']
    return np.asarray(result, dtype=np.float32)
