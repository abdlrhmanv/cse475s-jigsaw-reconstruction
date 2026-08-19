"""Sobel, Prewitt, and Canny (Phase 2). Gradients go through ConvolutionEngine."""

from __future__ import annotations

import numpy as np

from src.core.convolution import ConvolutionEngine
from src.core.protocols import EdgeDetector, GradientOperator, ImageFilter
from src.core.types import EdgeResult
from src.enhancement import GaussianFilter, _ensure_float, _ensure_gray


class SobelOperator(GradientOperator):
    def __init__(self, engine: ConvolutionEngine | None = None) -> None:
        self.engine = engine or ConvolutionEngine()

    def gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)
        gx = self.engine.convolve(gray, kx)
        gy = self.engine.convolve(gray, ky)
        return gx, gy


class PrewittOperator(GradientOperator):
    def __init__(self, engine: ConvolutionEngine | None = None) -> None:
        self.engine = engine or ConvolutionEngine()

    def gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)
        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]], dtype=np.float64)
        gx = self.engine.convolve(gray, kx)
        gy = self.engine.convolve(gray, ky)
        return gx, gy


class CannyEdgeDetector(EdgeDetector):
    """Seven-stage Canny. `gradient` defaults to Sobel; inject Prewitt for ablation.

    Stages: 1) grayscale, 2) Gaussian smooth, 3) gradient magnitude+direction,
    4) non-max suppression, 5) double threshold, 6) hysteresis, 7) output.
    """

    def __init__(
        self,
        gradient: GradientOperator | None = None,
        gaussian: ImageFilter | None = None,
        t_low: float | None = None,
        t_high: float | None = None,
    ) -> None:
        self.gradient = gradient or SobelOperator()
        self.gaussian = gaussian or GaussianFilter(k=5, sigma=1.0)
        self.t_low = t_low
        self.t_high = t_high

    def detect(self, image: np.ndarray) -> EdgeResult:
        # 1 — grayscale
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)

        # 2 — Gaussian smooth
        smoothed = self.gaussian.apply(gray)

        # 3 — gradient magnitude & direction
        gx, gy = self.gradient.gradients(smoothed)
        magnitude = np.hypot(gx, gy)
        orientation = np.arctan2(gy, gx)  # radians

        # 4 — non-maximum suppression
        nms = self._non_max_suppression(magnitude, orientation)

        # 5 — double threshold (auto-compute if not given)
        t_high = self.t_high if self.t_high is not None else np.percentile(nms[nms > 0], 90) if np.any(nms > 0) else 1.0
        t_low = self.t_low if self.t_low is not None else t_high * 0.4

        strong = nms >= t_high
        weak = (nms >= t_low) & ~strong

        # 6 — hysteresis: promote weak pixels connected (8-way) to strong
        edges = self._hysteresis(strong, weak)

        return EdgeResult(
            magnitude=magnitude,
            orientation=orientation,
            edges=edges.astype(np.float64) * 255.0,
            extras={
                "smoothed": smoothed,
                "nms": nms,
                "strong": strong.astype(np.float64) * 255.0,
                "weak": weak.astype(np.float64) * 255.0,
            },
        )

    @staticmethod
    def _non_max_suppression(mag: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """Thin edges to 1-px width by suppressing non-maxima along gradient direction."""
        h, w = mag.shape
        out = np.zeros_like(mag)
        angle = np.degrees(orient) % 180  # map to [0, 180)

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                a = angle[r, c]
                m = mag[r, c]
                # Quantise to 4 directions
                if (0 <= a < 22.5) or (157.5 <= a < 180):
                    n1, n2 = mag[r, c - 1], mag[r, c + 1]
                elif 22.5 <= a < 67.5:
                    n1, n2 = mag[r - 1, c + 1], mag[r + 1, c - 1]
                elif 67.5 <= a < 112.5:
                    n1, n2 = mag[r - 1, c], mag[r + 1, c]
                else:
                    n1, n2 = mag[r - 1, c - 1], mag[r + 1, c + 1]

                if m >= n1 and m >= n2:
                    out[r, c] = m
        return out

    @staticmethod
    def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
        """BFS from strong pixels; promote reachable weak pixels."""
        edges = strong.copy()
        h, w = edges.shape
        # Seed queue with all strong pixels neighbouring a weak pixel
        queue: list[tuple[int, int]] = []
        rs, cs = np.nonzero(strong)
        for r, c in zip(rs, cs):
            queue.append((r, c))

        while queue:
            r, c = queue.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and weak[nr, nc] and not edges[nr, nc]:
                        edges[nr, nc] = True
                        queue.append((nr, nc))
        return edges
