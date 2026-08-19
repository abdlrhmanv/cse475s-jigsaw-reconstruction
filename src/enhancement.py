"""From-scratch enhancement operators (Phase 1). Bodies are stubs until that phase.

Linear filters hold a `ConvolutionEngine` (DIP). Median cannot: it is an
order statistic, not a kernel inner product — the course requires a justified loop.
"""

from __future__ import annotations

import numpy as np

from src.core.convolution import ConvolutionEngine
from src.core.protocols import ImageFilter


class MeanFilter(ImageFilter):
    """Uniform k×k kernel. `k` must be odd when implemented."""

    def __init__(self, k: int, engine: ConvolutionEngine | None = None) -> None:
        self.k = k
        self.engine = engine or ConvolutionEngine()

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: mean filter via ConvolutionEngine.")


class GaussianFilter(ImageFilter):
    """Low-pass kernel from odd `k` and `sigma`. If `sigma` is None, use σ = 0.3*((k-1)/2-1)+0.8."""

    def __init__(
        self,
        k: int,
        sigma: float | None = None,
        engine: ConvolutionEngine | None = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.engine = engine or ConvolutionEngine()

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: Gaussian filter via ConvolutionEngine.")


class MedianFilter(ImageFilter):
    def __init__(self, k: int) -> None:
        self.k = k

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Phase 1: median is nonlinear; nested loop is required, not convolution."
        )


class HistogramComputer:
    """Returns a 256-bin count vector, not an image — hence not an ImageFilter."""

    def compute(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: 256-bin histogram.")


class HistogramEqualizer(ImageFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: histogram equalization.")


class ContrastStretcher(ImageFilter):
    """Percentile stretch; 1st/99th ignore dust so min-max is not dominated by outliers."""

    def __init__(self, p_low: float = 1.0, p_high: float = 99.0) -> None:
        self.p_low = p_low
        self.p_high = p_high

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: percentile contrast stretching.")


class UnsharpMask(ImageFilter):
    """`orig + alpha * (orig - blur)`. Reuses GaussianFilter; do not copy blur code."""

    def __init__(
        self,
        k: int,
        sigma: float,
        alpha: float,
        gaussian: GaussianFilter | None = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.gaussian = gaussian or GaussianFilter(k, sigma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 1: unsharp masking composes GaussianFilter.")


class FilterChain(ImageFilter):
    """Composite: apply filters in order. Empty chain is identity."""

    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = image
        for filt in self.filters:
            out = filt.apply(out)
        return out
