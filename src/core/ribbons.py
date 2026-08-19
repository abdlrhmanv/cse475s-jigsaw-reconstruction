"""Pack Lab colour strips and signed profiles into a 4-channel ribbon tensor."""

from __future__ import annotations

import numpy as np

RIBBON_LEN = 32


def resample(arr: np.ndarray, n: int = RIBBON_LEN) -> np.ndarray:
    m = len(arr)
    if m == n:
        return arr.copy()
    if m == 0:
        return np.zeros((n,) + arr.shape[1:])
    indices = np.linspace(0, m - 1, n)
    floor_idx = np.floor(indices).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, m - 1)
    frac = indices - floor_idx
    if arr.ndim == 1:
        return arr[floor_idx] * (1 - frac) + arr[ceil_idx] * frac
    frac = frac[:, np.newaxis]
    return arr[floor_idx] * (1 - frac) + arr[ceil_idx] * frac


def pack_ribbon(colour: np.ndarray, profile: np.ndarray, n: int = RIBBON_LEN) -> np.ndarray:
    """Return (4, n) float32: Lab channels + signed profile, all in roughly [-1, 1] / [0, 1].

    Channel 0–2: Lab (L/100, a/128, b/128). Channel 3: profile / max(|profile|, 1).
    """
    lab = resample(colour, n).astype(np.float64)
    if lab.ndim == 1:
        lab = np.stack([lab, np.zeros(n), np.zeros(n)], axis=-1)
    if lab.shape[-1] < 3:
        pad = np.zeros((n, 3 - lab.shape[-1]))
        lab = np.concatenate([lab, pad], axis=-1)
    lab = lab[:, :3]
    lab_n = np.stack([lab[:, 0] / 100.0, lab[:, 1] / 128.0, lab[:, 2] / 128.0], axis=0)

    prof = resample(profile, n).astype(np.float64)
    scale = max(float(np.max(np.abs(prof))), 1.0)
    prof_n = (prof / scale)[np.newaxis, :]
    return np.concatenate([lab_n, prof_n], axis=0).astype(np.float32)
