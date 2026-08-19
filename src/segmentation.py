"""Connected components from scratch (Phase 2). Unique IDs are stable in raster-scan order."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Labeler


class ConnectedComponentLabeler(Labeler):
    def __init__(self, min_area: int = 50, max_area: int | None = None) -> None:
        # Dust vs merged-piece guards; tune on validation images, not test.
        self.min_area = min_area
        self.max_area = max_area

    def label(self, binary: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Phase 2: two-pass 8-connected CCL from scratch; no cv2.connectedComponents."
        )
