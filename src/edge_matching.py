"""Classical side-pair dissimilarity (Phase 4). Output is lower-better D, not a probability."""

from __future__ import annotations

from src.core.protocols import CompatibilityMatcher
from src.core.types import CompatibilityTensor, Piece


class ClassicalCompatibilityMatcher(CompatibilityMatcher):
    def __init__(self, ws: float = 0.3, wc: float = 0.7) -> None:
        # Shape vs colour weights; must sum to 1 in the report. Tune on val only.
        self.ws = ws
        self.wc = wc

    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        raise NotImplementedError("Phase 4: shape + colour D tensor.")
