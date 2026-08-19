"""Shared image utility helpers."""

from __future__ import annotations

import numpy as np


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel if needed (luminance weights)."""
    if image.ndim == 2:
        return image
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
