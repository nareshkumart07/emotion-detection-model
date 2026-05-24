#!/usr/bin/env python3
"""Streamlit dashboard for testing EEG emotion inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import streamlit as st
import torch

from inference.config import (
    CHUNK_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    NUM_CHANNELS,
    QUADRANT_NAMES,
)
from inference.model_arch import EEGNet
from inference.predictor import EmotionPredictor, load_predictor


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / 'examples'
DEFAULT_EEGNET_SCALER_PATH = ROOT / 'models' / 'binary' / 'eegnet_scaler.pkl'
DEFAULT_EEGNET_VALENCE_PATH = ROOT / 'models' / 'binary' / 'eegnet_valence.pth'
DEFAULT_EEGNET_AROUSAL_PATH = ROOT / 'models' / 'binary' / 'eegnet_arousal.pth'
VALENCE_LABELS = ['Low Valence', 'High Valence']
AROUSAL_LABELS = ['Low Arousal', 'High Arousal']


@st.cache_resource(show_spinner=False)
def get_predictor(model_path: str, scaler_path: str, device: str) -> EmotionPredictor:
    device_arg = None if device == 'auto' else device
    return load_predictor(model_path=Path(model_path), scaler_path=Path(scaler_path), device=device_arg)


@st.cache_resource(show_spinner=False)
def get_binary_artifacts(
    model_path: str, scaler_path: str, device: str
) -> tuple[torch.nn.Module, Any, torch.device]:
    device_name = 'cuda' if device == 'auto' and torch.cuda.is_available() else device
    if device_name == 'auto':
        device_name = 'cpu'
    torch_device = torch.device(device_name)
    model = EEGNet(n_channels=NUM_CHANNELS, n_samples=CHUNK_SIZE, n_classes=2).to(
        torch_device
    )
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, torch_device


def load_example_payload(filename: str) -> Dict[str, Any]:
    path = EXAMPLES_DIR / filename
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _parse_payload(text: str) -> Dict[str, Any]:
    return json.loads(text)


def _render_result(result: Dict[str, Any]) -> None:
    st.success(f"Prediction: {result['label_name']} (class {result['label_id']})")
    st.metric('Confidence', f"{result['confidence']:.4f}")
    probs = result['probabilities']
    st.bar_chart({k: [float(v)] for k, v in probs.items()})
    st.json(result)


def _mode_features(predictor: EmotionPredictor) -> None:
    st.subheader('Predict from Features (14 x 4)')
    if st.button('Load Example Features'):
        st.session_state['features_payload'] = json.dumps(
            load_example_payload('dummy_features.json'), indent=2
        )

    default_payload = st.session_state.get(
        'features_payload',
        json.dumps(load_example_payload('dummy_features.json'), indent=2),
    )
    payload_text = st.text_area(
        'JSON payload',
        value=default_payload,
        height=280,
        key='features_text',
    )
    if st.button('Run Features Prediction'):
        payload = _parse_payload(payload_text)
        features = np.asarray(payload['features'], dtype=np.float32)
        result = predictor.predict_features(features).to_dict()
        _render_result(result)


def _mode_window(predictor: EmotionPredictor) -> None:
    st.subheader('Predict from Window + Baseline')
    st.caption(
        f'Expected shapes: eeg ({NUM_CHANNELS}, {CHUNK_SIZE}), baseline ({NUM_CHANNELS}, T>=128).'
    )
    if st.button('Load Example Window Payload'):
        st.session_state['window_payload'] = json.dumps(
            load_example_payload('dummy_window.json'), indent=2
        )

    default_payload = st.session_state.get(
        'window_payload',
        json.dumps(load_example_payload('dummy_window.json'), indent=2),
    )
    payload_text = st.text_area(
        'JSON payload',
        value=default_payload,
        height=280,
        key='window_text',
    )
    if st.button('Run Window Prediction'):
        payload = _parse_payload(payload_text)
        eeg = np.asarray(payload['eeg'], dtype=np.float64)
        baseline = np.asarray(payload['baseline'], dtype=np.float64)
        result = predictor.predict_window(eeg, baseline).to_dict()
        _render_result(result)


def _mode_trial(predictor: EmotionPredictor) -> None:
    st.subheader('Predict Trial (multiple windows)')
    strategy = st.selectbox('Aggregation strategy', options=['vote', 'mean_prob'])
    if st.button('Load Example Trial Payload'):
        st.session_state['trial_payload'] = json.dumps(
            load_example_payload('dummy_trial.json'), indent=2
        )

    default_payload = st.session_state.get(
        'trial_payload',
        json.dumps(load_example_payload('dummy_trial.json'), indent=2),
    )
    payload_text = st.text_area(
        'JSON payload',
        value=default_payload,
        height=280,
        key='trial_text',
    )
    if st.button('Run Trial Prediction'):
        payload = _parse_payload(payload_text)
        windows = [np.asarray(w, dtype=np.float64) for w in payload['eeg_windows']]
        baseline = np.asarray(payload['baseline'], dtype=np.float64)
        result = predictor.predict_trial(
            eeg_windows=windows,
            baseline=baseline,
            strategy=strategy,
        ).to_dict()
        _render_result(result)


def _predict_binary_window(
    model: torch.nn.Module,
    scaler: Any,
    device: torch.device,
    eeg: np.ndarray,
    task: str,
) -> Dict[str, Any]:
    arr = np.asarray(eeg, dtype=np.float32)
    if arr.shape != (NUM_CHANNELS, CHUNK_SIZE):
        raise ValueError(
            f'eeg must have shape ({NUM_CHANNELS}, {CHUNK_SIZE}), got {arr.shape}'
        )
    flat = arr.reshape(1, -1)
    if hasattr(scaler, 'n_features_in_') and int(scaler.n_features_in_) != flat.shape[1]:
        raise ValueError(
            'Scaler feature mismatch: '
            f'expected {scaler.n_features_in_}, got {flat.shape[1]}.'
        )
    scaled = scaler.transform(flat).reshape(arr.shape).astype(np.float32)
    x = torch.from_numpy(scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    label_id = int(np.argmax(probs))
    label_names = VALENCE_LABELS if task == 'valence' else AROUSAL_LABELS
    return {
        'label_id': label_id,
        'label_name': label_names[label_id],
        'confidence': float(probs[label_id]),
        'probabilities': {name: float(probs[i]) for i, name in enumerate(label_names)},
    }


def _mode_binary_window(
    model: torch.nn.Module, scaler: Any, torch_device: torch.device, task: str
) -> None:
    st.subheader(f'Binary EEGNet Prediction ({task})')
    st.caption(
        f'Expected shape: eeg ({NUM_CHANNELS}, {CHUNK_SIZE}). Baseline is ignored for EEGNet binary path.'
    )
    if st.button('Load Example Window Payload'):
        st.session_state['binary_window_payload'] = json.dumps(
            load_example_payload('dummy_window.json'), indent=2
        )
    default_payload = st.session_state.get(
        'binary_window_payload',
        json.dumps(load_example_payload('dummy_window.json'), indent=2),
    )
    payload_text = st.text_area(
        'JSON payload',
        value=default_payload,
        height=280,
        key='binary_window_text',
    )
    if st.button('Run Binary Prediction'):
        payload = _parse_payload(payload_text)
        result = _predict_binary_window(
            model=model,
            scaler=scaler,
            device=torch_device,
            eeg=np.asarray(payload['eeg'], dtype=np.float32),
            task=task,
        )
        _render_result(result)


def main() -> None:
    st.set_page_config(page_title='EEG Emotion Dashboard', layout='wide')
    st.title('EEG Emotion Classification Dashboard')
    st.caption('Research demo for testing model predictions.')

    with st.sidebar:
        st.header('Model Settings')
        model_family = st.radio(
            'Model family',
            options=[
                'Quadrant BiLSTM (4-class)',
                'Binary EEGNet (2-class)',
            ],
        )
        task = 'valence'
        if model_family == 'Binary EEGNet (2-class)':
            task = st.selectbox('Binary task', options=['valence', 'arousal'])
            default_model = (
                DEFAULT_EEGNET_VALENCE_PATH
                if task == 'valence'
                else DEFAULT_EEGNET_AROUSAL_PATH
            )
            default_scaler = DEFAULT_EEGNET_SCALER_PATH
        else:
            default_model = DEFAULT_MODEL_PATH
            default_scaler = DEFAULT_SCALER_PATH

        model_path = st.text_input('Model path', value=str(default_model))
        scaler_path = st.text_input('Scaler path', value=str(default_scaler))
        device = st.selectbox('Device', options=['auto', 'cpu', 'cuda'], index=0)
        if st.button('Reload Model'):
            get_predictor.clear()
            get_binary_artifacts.clear()
        st.markdown('---')
        st.write('Classes')
        if model_family == 'Binary EEGNet (2-class)':
            st.write(', '.join(VALENCE_LABELS if task == 'valence' else AROUSAL_LABELS))
        else:
            st.write(', '.join(QUADRANT_NAMES))

    try:
        if model_family == 'Binary EEGNet (2-class)':
            model, scaler, torch_device = get_binary_artifacts(
                model_path=model_path, scaler_path=scaler_path, device=device
            )
            _mode_binary_window(
                model=model, scaler=scaler, torch_device=torch_device, task=task
            )
        else:
            predictor = get_predictor(
                model_path=model_path, scaler_path=scaler_path, device=device
            )
            mode = st.radio(
                'Input Mode',
                options=[
                    'Features (14x4)',
                    'Window + Baseline',
                    'Trial (multi-window)',
                ],
                horizontal=True,
            )
            if mode == 'Features (14x4)':
                _mode_features(predictor)
            elif mode == 'Window + Baseline':
                _mode_window(predictor)
            else:
                _mode_trial(predictor)
    except Exception as exc:  # noqa: BLE001
        st.error(f'Prediction failed: {exc}')


if __name__ == '__main__':
    main()
  