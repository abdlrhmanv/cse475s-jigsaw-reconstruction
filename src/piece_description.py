"""Four corners, tab/blank/flat class, Lab strips (Phase 3) — highest classical risk."""

from __future__ import annotations

import numpy as np

from src.core.protocols import CornerFinder, PieceDescriptor
from src.core.types import Piece


class HybridCornerFinder(CornerFinder):
    """Hull → DP → 4-point subset. Primary: true corners are convex; tab tips are not selected."""

    def find(self, contour: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 3: hull + DP + 4-point subset.")


class CurvatureCornerFinder(CornerFinder):
    """Fallback when the hybrid subset has bad interior angles."""

    def find(self, contour: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 3: curvature-extrema fallback.")


class PieceDescriptorImpl(PieceDescriptor):
    def __init__(self, corner_finder: CornerFinder | None = None) -> None:
        self.corner_finder = corner_finder or HybridCornerFinder()

    def describe(self, piece: Piece) -> Piece:
        raise NotImplementedError("Phase 3: corners, tab/blank/flat, Lab strips.")
