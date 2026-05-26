"""Lightweight compatibility shim for torcheeg imports on Windows.

This project uses a subset of TorchEEG transforms. TorchEEG imports
`aryule` from the external `spectrum` package at module import time.
On some Windows setups, installing `spectrum` requires C++ build tools.

This local shim provides a minimal `aryule` implementation so TorchEEG
can import and run for the transforms used in this repo.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _autocorr(x: np.ndarray, order: int, norm: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError('Input signal must be non-empty.')
    full = np.correlate(x, x, mode='full')
    mid = n - 1
    r = full[mid : mid + order + 1].astype(np.float64)
    if norm == 'biased':
        r /= float(n)
    elif norm == 'unbiased':
        denom = np.arange(n, n - order - 1, -1, dtype=np.float64)
        r /= denom
    else:
        raise ValueError("norm must be 'biased' or 'unbiased'")
    return r


def aryule(
    x: np.ndarray,
    order: int = 4,
    norm: str = 'biased',
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Estimate AR coefficients with Yule-Walker + Levinson-Durbin.

    Returns `(a, e, k)` to mirror `spectrum.aryule` signature:
    - `a`: AR polynomial coefficients, with `a[0] == 1`
    - `e`: final prediction error variance
    - `k`: reflection coefficients
    """

    if order < 1:
        raise ValueError('order must be >= 1')

    r = _autocorr(x, order=order, norm=norm)
    a = np.zeros(order + 1, dtype=np.float64)
    k = np.zeros(order, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0])
    if e <= 0:
        return a, 0.0, k

    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        ki = -acc / e
        k[i - 1] = ki
        a_prev = a.copy()
        a[1:i] = a_prev[1:i] + ki * a_prev[i - 1 : 0 : -1]
        a[i] = ki
        e *= 1.0 - ki * ki
        if e <= 1e-12:
            e = 1e-12
            break

    return a, float(e), k
