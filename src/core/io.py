"""Image load/save via Pillow. Not used for filters, edges, CCL, or Canny (course rule)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ImageStore:
    def load(self, path: str | Path) -> np.ndarray:
        """Load as a NumPy array. RGBA is flattened to RGB so Lab strips stay 3-channel."""
        image = Image.open(path)
        array = np.asarray(image)
        if array.ndim == 2:
            return array
        if array.shape[-1] == 4:
            return np.asarray(image.convert("RGB"))
        return array

    def save(self, path: str | Path, image: np.ndarray) -> None:
        """Write `image`. Floats in [0, 1] are scaled to 8-bit; other floats are clipped."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = image
        if array.dtype != np.uint8:
            finite = np.nan_to_num(array, nan=0.0)
            if finite.max() <= 1.0:
                finite = finite * 255.0
            array = np.clip(finite, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(path)
