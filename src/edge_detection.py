"""Sobel, Prewitt, and Canny (Phase 2). Gradients go through ConvolutionEngine."""

from __future__ import annotations

import numpy as np

from src.core.convolution import ConvolutionEngine
from src.core.protocols import EdgeDetector, GradientOperator, ImageFilter
from src.core.types import EdgeResult
from src.enhancement import GaussianFilter


class SobelOperator(GradientOperator):
    def __init__(self, engine: ConvolutionEngine | None = None) -> None:
        self.engine = engine or ConvolutionEngine()

    def gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Phase 2: Sobel Gx, Gy via ConvolutionEngine.")


class PrewittOperator(GradientOperator):
    def __init__(self, engine: ConvolutionEngine | None = None) -> None:
        self.engine = engine or ConvolutionEngine()

    def gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Phase 2: Prewitt Gx, Gy via ConvolutionEngine.")


class CannyEdgeDetector(EdgeDetector):
    """Seven-stage Canny. `gradient` defaults to Sobel; inject Prewitt for the report ablation."""

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
        raise NotImplementedError("Phase 2: full Canny (7 stages), no cv2.Canny.")
