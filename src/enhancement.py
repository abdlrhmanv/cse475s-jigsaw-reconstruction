"""From-scratch enhancement operators.

Linear filters hold a `ConvolutionEngine` (DIP). Median cannot: it is an
order statistic, not a kernel inner product — the course requires a justified loop.
"""

from __future__ import annotations

import numpy as np

from src.core.convolution import ConvolutionEngine
from src.core.protocols import ImageFilter


def _ensure_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) if image.dtype != np.float64 else image


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel if needed (luminance weights)."""
    if image.ndim == 2:
        return image
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Linear filters
# ---------------------------------------------------------------------------


class MeanFilter(ImageFilter):
    """Uniform k×k kernel K_ij = 1/k². `k` must be odd."""

    def __init__(self, k: int, engine: ConvolutionEngine | None = None) -> None:
        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")
        self.k = k
        self.engine = engine or ConvolutionEngine()

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.k, self.k), dtype=np.float64) / (self.k * self.k)
        return self.engine.convolve(_ensure_float(image), kernel)


class GaussianFilter(ImageFilter):
    """Low-pass kernel from odd `k` and `sigma`.

    If `sigma` is None: σ = 0.3 * ((k-1)/2 - 1) + 0.8  (OpenCV convention).
    """

    def __init__(
        self,
        k: int,
        sigma: float | None = None,
        engine: ConvolutionEngine | None = None,
    ) -> None:
        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")
        self.k = k
        self.sigma = sigma if sigma is not None else 0.3 * ((k - 1) / 2 - 1) + 0.8
        self.engine = engine or ConvolutionEngine()

    def kernel(self) -> np.ndarray:
        """Build the 2-D Gaussian kernel, normalised so sum = 1."""
        half = self.k // 2
        ax = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(ax, ax)
        g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * self.sigma ** 2))
        return g / g.sum()

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self.engine.convolve(_ensure_float(image), self.kernel())


# ---------------------------------------------------------------------------
# Non-linear filter
# ---------------------------------------------------------------------------


class MedianFilter(ImageFilter):
    """Order-statistic filter: replaces each pixel with the median of its k×k window.

    Unlike mean/Gaussian, median is **non-linear** — it cannot be expressed as a
    kernel inner-product, so convolution cannot implement it. A per-pixel loop
    over the sorted window is the correct (and course-required) approach.
    """

    def __init__(self, k: int) -> None:
        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")
        self.k = k

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_float(image)
        if img.ndim == 3:
            return np.stack(
                [self._apply_2d(img[:, :, c]) for c in range(img.shape[2])],
                axis=-1,
            )
        return self._apply_2d(img)

    def _apply_2d(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        pad = self.k // 2
        padded = np.pad(img, pad, mode="reflect")
        out = np.empty_like(img)
        mid = (self.k * self.k) // 2
        for r in range(h):
            for c in range(w):
                window = padded[r : r + self.k, c : c + self.k].ravel()
                window.sort()
                out[r, c] = window[mid]
        return out


# ---------------------------------------------------------------------------
# Histogram operations
# ---------------------------------------------------------------------------


class HistogramComputer:
    """Returns a 256-bin count vector, not an image — hence not an ImageFilter."""

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute histogram for a grayscale (or already-gray float) image.

        Quantises float images to [0, 255] uint8 before counting.
        """
        gray = _ensure_gray(image) if image.ndim == 3 else image
        arr = _to_uint8(gray) if gray.dtype != np.uint8 else gray
        hist = np.zeros(256, dtype=np.int64)
        for val in arr.ravel():
            hist[val] += 1
        return hist


class HistogramEqualizer(ImageFilter):
    """CDF-based equalisation: s = (L-1) * (cdf[v] - cdf_min) / (N - cdf_min)."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(image) if image.ndim == 3 else _ensure_float(image)
        arr = _to_uint8(gray)
        hist = HistogramComputer().compute(arr)
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        n = arr.size
        lut = np.zeros(256, dtype=np.uint8)
        denom = n - cdf_min
        if denom > 0:
            lut = np.clip(((cdf - cdf_min) * 255.0 / denom + 0.5), 0, 255).astype(np.uint8)
        return lut[arr].astype(np.float64)


class ContrastStretcher(ImageFilter):
    """Percentile stretch; p_low/p_high ignore outliers so min-max isn't skewed by dust."""

    def __init__(self, p_low: float = 1.0, p_high: float = 99.0) -> None:
        self.p_low = p_low
        self.p_high = p_high

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = _ensure_float(image)
        v_low = np.percentile(img, self.p_low)
        v_high = np.percentile(img, self.p_high)
        if v_high - v_low < 1e-6:
            return img
        stretched = (img - v_low) / (v_high - v_low) * 255.0
        return np.clip(stretched, 0, 255)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------


class UnsharpMask(ImageFilter):
    """`sharp = orig + alpha * (orig - blur)`. Reuses GaussianFilter; no blur code copied."""

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
        img = _ensure_float(image)
        blurred = self.gaussian.apply(img)
        sharp = img + self.alpha * (img - blurred)
        return np.clip(sharp, 0, 255)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class FilterChain(ImageFilter):
    """Composite: apply filters in order. Empty chain is identity."""

    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters

    def apply(self, image: np.ndarray) -> np.ndarray:
        out = image
        for filt in self.filters:
            out = filt.apply(out)
        return out
