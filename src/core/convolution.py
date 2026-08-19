"""Shared linear-filter primitive. All mean/Gaussian/Sobel kernels must call this.

Implemented in Phase 1 from scratch. OpenCV `filter2D` is forbidden for graded operators.
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

        `padding="reflect"` avoids a dark frame that would create false edges
        before Canny. Kernel size must be odd (enforced when implemented).
        """
        raise NotImplementedError(
            "Phase 1: implement convolution from scratch; do not use cv2.filter2D."
        )
