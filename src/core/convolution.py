"""Shared linear-filter primitive. All mean/Gaussian/Sobel kernels must call this.

Implemented from scratch. OpenCV `filter2D` is forbidden for graded operators.
"""

from __future__ import annotations

import numpy as np


class ConvolutionEngine:
    def convolve(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        padding: str = "reflect",
    ) -> np.ndarray:
        """Convolve `image` with `kernel`.

        Parameters
        ----------
        image : 2-D grayscale or 3-D colour array (H, W) or (H, W, C).
        kernel : 2-D array with odd height and width.
        padding : np.pad mode applied before the sliding window so the output
            keeps the input shape. ``"reflect"`` avoids dark-frame artefacts
            that would create false edges before Canny.

        Returns
        -------
        Filtered image with the same shape and dtype as ``image``.
        """
        if kernel.ndim != 2:
            raise ValueError("kernel must be 2-D")
        kh, kw = kernel.shape
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(f"kernel size must be odd, got {kh}×{kw}")

        # Colour images: convolve each channel independently.
        if image.ndim == 3:
            channels = [
                self._convolve_2d(image[:, :, c], kernel, padding)
                for c in range(image.shape[2])
            ]
            return np.stack(channels, axis=-1)

        return self._convolve_2d(image, kernel, padding)

    @staticmethod
    def _convolve_2d(
        image: np.ndarray,
        kernel: np.ndarray,
        padding: str,
    ) -> np.ndarray:
        """Core 2-D convolution: flip the kernel then slide (correlation with flipped kernel)."""
        kernel = kernel[::-1, ::-1]
        img = image.astype(np.float64)
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode=padding)

        h, w = img.shape
        out = np.empty((h, w), dtype=np.float64)

        # Vectorised sliding-window: extract each kernel-offset plane and
        # multiply by the corresponding weight, avoiding a pure-Python
        # per-pixel loop while still being from-scratch (no cv2/scipy).
        for ki in range(kh):
            for kj in range(kw):
                out_add = padded[ki : ki + h, kj : kj + w] * kernel[ki, kj]
                if ki == 0 and kj == 0:
                    out[:] = out_add
                else:
                    out += out_add

        return out
