"""Foreground masks (Phase 1). Swap implementations via `Thresholder`, not if-else."""

from __future__ import annotations

import numpy as np

from src.core.convolution import ConvolutionEngine
from src.core.protocols import Thresholder
from src.core.image_utils import _ensure_gray
from src.enhancement import _ensure_float


class GlobalThreshold(Thresholder):
    """Binary: B(x,y) = 255 if I(x,y) > T else 0."""

    def __init__(self, t: float) -> None:
        self.t = t

    def threshold(self, image: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)
        return np.where(gray > self.t, 255.0, 0.0)


class OtsuThreshold(Thresholder):
    """Maximises between-class variance σ²_B to find optimal T.

    σ²_B(T) = w0(T) * w1(T) * (μ0(T) - μ1(T))²
    """

    def threshold(self, image: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)
        arr = np.clip(gray, 0, 255).astype(np.uint8)

        hist = np.zeros(256, dtype=np.float64)
        for v in arr.ravel():
            hist[v] += 1
        hist /= arr.size

        best_t, best_var = 0, -1.0
        w0 = 0.0
        mu0_accum = 0.0
        mu_total = sum(i * hist[i] for i in range(256))

        for t in range(256):
            w0 += hist[t]
            if w0 == 0:
                continue
            w1 = 1.0 - w0
            if w1 == 0:
                break
            mu0_accum += t * hist[t]
            mu0 = mu0_accum / w0
            mu1 = (mu_total - mu0_accum) / w1
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > best_var:
                best_var = var
                best_t = t

        return np.where(arr > best_t, 255.0, 0.0)


class AdaptiveThreshold(Thresholder):
    """Local T = local_mean(window) - c.

    kind="mean" uses box average; kind="gaussian" uses Gaussian-weighted average.
    """

    def __init__(self, w: int, c: float, kind: str = "mean") -> None:
        if w % 2 == 0:
            raise ValueError(f"w must be odd, got {w}")
        self.w = w
        self.c = c
        self.kind = kind
        self._engine = ConvolutionEngine()

    def threshold(self, image: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)

        if self.kind == "gaussian":
            from src.enhancement import GaussianFilter
            local_mean = GaussianFilter(self.w).apply(gray)
        else:
            kernel = np.ones((self.w, self.w), dtype=np.float64) / (self.w * self.w)
            local_mean = self._engine.convolve(gray, kernel)

        return np.where(gray > local_mean - self.c, 255.0, 0.0)
