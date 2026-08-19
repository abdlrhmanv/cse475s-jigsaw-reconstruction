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


from src.core.image_utils import _ensure_gray as _ensure_gray  # noqa: F811 — canonical home is core


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
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
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
    over the sorted window is the course-required approach (not OpenCV). On this
    photo set that is ~17 s per board; accepted for the submission.
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
        if img.dtype == np.uint8 or (np.issubdtype(img.dtype, np.floating) and float(img.min()) >= 0 and float(img.max()) <= 255):
            return self._apply_2d_hist(np.clip(img, 0, 255).astype(np.uint8)).astype(img.dtype)
        return self._apply_2d_sort(img)

    def _apply_2d_sort(self, img: np.ndarray) -> np.ndarray:
        """General fallback: sort-based median."""
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

    def _apply_2d_hist(self, img: np.ndarray) -> np.ndarray:
        """O(H*W*k) histogram-median for uint8 images."""
        h, w = img.shape
        pad = self.k // 2
        padded = np.pad(img, pad, mode="reflect")
        out = np.empty_like(img)
        mid = (self.k * self.k) // 2
        hist = np.zeros(256, dtype=np.int32)
        for r in range(h):
            hist[:] = 0
            for dr in range(self.k):
                for dc in range(self.k):
                    hist[padded[r + dr, dc]] += 1
            out[r, 0] = self._median_from_hist(hist, mid)
            for c in range(1, w):
                for dr in range(self.k):
                    hist[padded[r + dr, c - 1]] -= 1
                    hist[padded[r + dr, c + self.k - 1]] += 1
                out[r, c] = self._median_from_hist(hist, mid)
        return out

    @staticmethod
    def _median_from_hist(hist: np.ndarray, mid: int) -> int:
        count = 0
        for v in range(256):
            count += hist[v]
            if count > mid:
                return v
        return 255


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
# Binary morphology
# ---------------------------------------------------------------------------


class BinaryCloser(ImageFilter):
    """Morphological close (dilate then erode) with an odd square kernel.

    Closes small cracks so hole-filling can treat printed texture as interior.
    """

    def __init__(self, k: int = 7) -> None:
        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")
        self.k = k

    def apply(self, image: np.ndarray) -> np.ndarray:
        fg = (image if image.ndim == 2 else _ensure_gray(image)) > 0
        pad = self.k // 2
        dilated = self._dilate(fg, pad)
        closed = self._erode(dilated, pad)
        return closed.astype(np.float64) * 255.0

    @staticmethod
    def _dilate(fg: np.ndarray, pad: int) -> np.ndarray:
        h, w = fg.shape
        padded = np.pad(fg, pad, constant_values=False)
        out = np.zeros_like(fg)
        for i in range(2 * pad + 1):
            for j in range(2 * pad + 1):
                out |= padded[i : i + h, j : j + w]
        return out

    @staticmethod
    def _erode(fg: np.ndarray, pad: int) -> np.ndarray:
        h, w = fg.shape
        padded = np.pad(fg, pad, constant_values=True)
        out = np.ones_like(fg)
        for i in range(2 * pad + 1):
            for j in range(2 * pad + 1):
                out &= padded[i : i + h, j : j + w]
        return out


class BinaryHoleFiller(ImageFilter):
    """Fill enclosed 0-regions in a binary mask (texture holes inside pieces).

    Flood-fills background from the image border; leftover background pixels
    are holes and are set to 255. Does not use OpenCV morphology.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        from collections import deque

        gray = image if image.ndim == 2 else _ensure_gray(image)
        fg = gray > 0
        h, w = fg.shape
        reachable = np.zeros((h, w), dtype=bool)
        q: deque[tuple[int, int]] = deque()

        def try_seed(r: int, c: int) -> None:
            if not fg[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

        for r in range(h):
            try_seed(r, 0)
            try_seed(r, w - 1)
        for c in range(w):
            try_seed(0, c)
            try_seed(h - 1, c)

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not fg[nr, nc]:
                    reachable[nr, nc] = True
                    q.append((nr, nc))

        filled = fg | ~reachable
        return filled.astype(np.float64) * 255.0


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
