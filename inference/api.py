"""FastAPI routes for EEG emotion quadrant inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import CHUNK_SIZE, NUM_CHANNELS
from .predictor import EmotionPredictor, load_predictor
from .torcheeg_patch import apply_lmdb_patch

apply_lmdb_patch()


@lru_cache
def get_predictor() -> EmotionPredictor:
    return load_predictor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title='Brainwave EEG Emotion API',
    description=(
        'Research prototype: 4-class valence–arousal quadrants from EEG band power. '
        'Not for clinical use.'
    ),
    version='0.1.0',
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictFeaturesRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description='Baseline-corrected PSD, shape (14, 4)',
    )

    @field_validator('features')
    @classmethod
    def check_features_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != NUM_CHANNELS:
            raise ValueError(f'features must have {NUM_CHANNELS} rows')
        if any(len(row) != 4 for row in v):
            raise ValueError('each row must have 4 band values')
        return v


class PredictWindowRequest(BaseModel):
    eeg: List[List[float]] = Field(
        ...,
        description=f'Stimulus EEG window, shape ({NUM_CHANNELS}, {CHUNK_SIZE})',
    )
    baseline: List[List[float]] = Field(
        ...,
        description='Resting baseline, shape (14, T) with T >= 128',
    )

    @field_validator('eeg')
    @classmethod
    def check_eeg(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != NUM_CHANNELS:
            raise ValueError(f'eeg must have {NUM_CHANNELS} channels')
        if any(len(row) != CHUNK_SIZE for row in v):
            raise ValueError(f'each channel must have {CHUNK_SIZE} samples')
        return v

    @field_validator('baseline')
    @classmethod
    def check_baseline(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != NUM_CHANNELS:
            raise ValueError(f'baseline must have {NUM_CHANNELS} channels')
        if any(len(row) < 128 for row in v):
            raise ValueError('each baseline channel must have at least 128 samples')
        return v


class PredictTrialRequest(BaseModel):
    eeg_windows: List[List[List[float]]] = Field(
        ...,
        description='List of stimulus windows, each (14, 256)',
    )
    baseline: List[List[float]]
    strategy: str = Field(default='vote', pattern='^(vote|mean_prob)$')

    @field_validator('eeg_windows')
    @classmethod
    def check_eeg_windows(
        cls, v: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        if not v:
            raise ValueError('eeg_windows must not be empty')
        for i, window in enumerate(v):
            if len(window) != NUM_CHANNELS:
                raise ValueError(f'eeg_windows[{i}] must have {NUM_CHANNELS} channels')
            if any(len(row) != CHUNK_SIZE for row in window):
                raise ValueError(
                    f'eeg_windows[{i}] channels must each have {CHUNK_SIZE} samples'
                )
        return v

    @field_validator('baseline')
    @classmethod
    def check_trial_baseline(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != NUM_CHANNELS:
            raise ValueError(f'baseline must have {NUM_CHANNELS} channels')
        if any(len(row) < 128 for row in v):
            raise ValueError('each baseline channel must have at least 128 samples')
        return v


class PredictionResponse(BaseModel):
    label_id: int
    label_name: str
    probabilities: dict[str, float]
    confidence: float


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        predictor = get_predictor()
        device = str(predictor.device)
        loaded = True
    except FileNotFoundError:
        device = 'unknown'
        loaded = False
    return HealthResponse(status='ok', model_loaded=loaded, device=device)


@app.post('/predict/features', response_model=PredictionResponse)
def predict_features(body: PredictFeaturesRequest) -> PredictionResponse:
    try:
        features = np.asarray(body.features, dtype=np.float32)
        result = get_predictor().predict_features(features)
        return PredictionResponse(**result.to_dict())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/predict/window', response_model=PredictionResponse)
def predict_window(body: PredictWindowRequest) -> PredictionResponse:
    try:
        eeg = np.asarray(body.eeg, dtype=np.float64)
        baseline = np.asarray(body.baseline, dtype=np.float64)
        result = get_predictor().predict_window(eeg, baseline)
        return PredictionResponse(**result.to_dict())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/predict/trial', response_model=PredictionResponse)
def predict_trial(body: PredictTrialRequest) -> PredictionResponse:
    try:
        windows = [np.asarray(w, dtype=np.float64) for w in body.eeg_windows]
        baseline = np.asarray(body.baseline, dtype=np.float64)
        result = get_predictor().predict_trial(
            windows, baseline, strategy=body.strategy
        )
        return PredictionResponse(**result.to_dict())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/preprocess/demo')
def preprocess_demo() -> dict:
    return {
        'eeg_window_shape': [NUM_CHANNELS, CHUNK_SIZE],
        'features_shape': [NUM_CHANNELS, 4],
        'baseline_min_samples': 128,
        'note': 'POST raw arrays to /predict/window or /predict/features',
    }
