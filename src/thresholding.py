"""Foreground masks for CCL (Phase 1). Swap implementations via `Thresholder`, not if-else in the pipeline."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Thresholder


class GlobalThreshold(Thresholder):
    def __init__(self, t: float) -> None:
        self.t = t

    def threshold(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: global threshold B = I > T.")


class OtsuThreshold(Thresholder):
    """T maximising between-class variance; default when illumination is even."""

    def threshold(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: Otsu between-class variance.")


class AdaptiveThreshold(Thresholder):
    """Local T = window mean (or Gaussian-weighted mean) minus `c`. Use when lighting varies across the board."""

    def __init__(self, w: int, c: float, kind: str = "mean") -> None:
        self.w = w
        self.c = c
        self.kind = kind

    def threshold(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: adaptive mean or Gaussian threshold.")
