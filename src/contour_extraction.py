"""Moore contour trace and piece crops (Phase 2). Do not call cv2.findContours on the graded path."""

from __future__ import annotations

import numpy as np

from src.core.protocols import ContourTracer, PieceExtractor
from src.core.types import Piece


class MooreContourTracer(ContourTracer):
    def trace(self, blob_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 2: Moore-neighbour trace; no cv2.findContours.")


class PieceExtractorImpl(PieceExtractor):
    def __init__(self, pad: int = 5, tracer: ContourTracer | None = None) -> None:
        # Padding keeps tabs from being clipped at the bbox; 5 px is the plan default.
        self.pad = pad
        self.tracer = tracer or MooreContourTracer()

    def extract(self, image: np.ndarray, labels: np.ndarray) -> list[Piece]:
        raise NotImplementedError("Phase 2: bbox crop, mask, contour, PCA orientation.")
