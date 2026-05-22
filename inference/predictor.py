from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    NUM_CHANNELS,
    QUADRANT_NAMES,
)
from .model_arch import EEGBiLSTMClassifier
from .preprocess import extract_features


@dataclass
class PredictionResult:
    label_id: int
    label_name: str
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


class EmotionPredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        scaler_path: Union[str, Path] = DEFAULT_SCALER_PATH,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.scaler = joblib.load(scaler_path)
        self.model = EEGBiLSTMClassifier().to(self.device)
        try:
            state = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _scale(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape(1, -1)
        if (
            hasattr(self.scaler, 'n_features_in_')
            and int(self.scaler.n_features_in_) != flat.shape[1]
        ):
            raise ValueError(
                'Scaler feature mismatch: '
                f'expected {self.scaler.n_features_in_}, got {flat.shape[1]}.'
            )
        scaled = self.scaler.transform(flat).astype(np.float32)
        return scaled.reshape(NUM_CHANNELS, -1)

    @torch.no_grad()
    def predict_features(self, features: np.ndarray) -> PredictionResult:
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape != (NUM_CHANNELS, 4):
            raise ValueError(
                f'features must have shape ({NUM_CHANNELS}, 4), got {arr.shape}'
            )
        scaled = self._scale(arr)
        x = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        label_id = int(probs.argmax())
        return PredictionResult(
            label_id=label_id,
            label_name=QUADRANT_NAMES[label_id],
            probabilities={
                name: float(probs[i]) for i, name in enumerate(QUADRANT_NAMES)
            },
            confidence=float(probs[label_id]),
        )

    def predict_window(
        self, eeg: np.ndarray, baseline: np.ndarray
    ) -> PredictionResult:
        features = extract_features(eeg, baseline)
        return self.predict_features(features)

    def predict_trial(
        self,
        eeg_windows: List[np.ndarray],
        baseline: np.ndarray,
        strategy: str = 'vote',
    ) -> PredictionResult:
        if not eeg_windows:
            raise ValueError('eeg_windows must not be empty')
        results = [self.predict_window(w, baseline) for w in eeg_windows]
        if strategy == 'mean_prob':
            mean_probs = np.mean(
                [[r.probabilities[n] for n in QUADRANT_NAMES] for r in results],
                axis=0,
            )
            label_id = int(mean_probs.argmax())
            return PredictionResult(
                label_id=label_id,
                label_name=QUADRANT_NAMES[label_id],
                probabilities={
                    name: float(mean_probs[i])
                    for i, name in enumerate(QUADRANT_NAMES)
                },
                confidence=float(mean_probs[label_id]),
            )
        if strategy != 'vote':
            raise ValueError("strategy must be 'vote' or 'mean_prob'")
        avg_probs = {
            name: float(np.mean([r.probabilities[name] for r in results]))
            for name in QUADRANT_NAMES
        }
        ids = np.asarray([r.label_id for r in results], dtype=np.int64)
        counts = np.bincount(ids, minlength=len(QUADRANT_NAMES))
        top_count = int(counts.max())
        tied = np.flatnonzero(counts == top_count)
        if len(tied) == 1:
            label_id = int(tied[0])
        else:
            tie_scores = np.asarray(
                [avg_probs[QUADRANT_NAMES[idx]] for idx in tied], dtype=np.float64
            )
            label_id = int(tied[int(tie_scores.argmax())])
        return PredictionResult(
            label_id=label_id,
            label_name=QUADRANT_NAMES[label_id],
            probabilities=avg_probs,
            confidence=float(avg_probs[QUADRANT_NAMES[label_id]]),
        )


def load_predictor(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    scaler_path: Union[str, Path] = DEFAULT_SCALER_PATH,
    device: Optional[str] = None,
) -> EmotionPredictor:
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f'Model not found: {model_path}. Place/export model at models/quadrant/model.pth.'
        )
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f'Scaler not found: {scaler_path}. Place/export scaler at models/quadrant/scaler.pkl.'
        )
    return EmotionPredictor(model_path, scaler_path, device)
