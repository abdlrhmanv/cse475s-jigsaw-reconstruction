"""Operator contracts. The pipeline depends on these ABCs, never on a named matcher.

Keep each protocol to one method (ISP). Adding Siamese or GNN must be a new
`CompatibilityMatcher` class, not a branch inside the assembler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import (
    AssemblyState,
    CompatibilityTensor,
    EdgeResult,
    GroundTruth,
    Piece,
    Puzzle,
)


class ImageFilter(ABC):
    """Pointwise or neighbourhood transform that returns an image of the same rank."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Thresholder(ABC):
    """Gray image → binary foreground mask used by CCL."""

    @abstractmethod
    def threshold(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientOperator(ABC):
    """Returns (Gx, Gy). Injected into Canny so Sobel vs Prewitt is a swap, not a fork."""

    @abstractmethod
    def gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class EdgeDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> EdgeResult:
        raise NotImplementedError


class Labeler(ABC):
    """Assign unique positive integer IDs; 0 is background."""

    @abstractmethod
    def label(self, binary: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ContourTracer(ABC):
    @abstractmethod
    def trace(self, blob_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PieceExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray, labels: np.ndarray) -> list[Piece]:
        raise NotImplementedError


class CornerFinder(ABC):
    """Return four (x, y) corners. Tabs must not be treated as corners."""

    @abstractmethod
    def find(self, contour: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PieceDescriptor(ABC):
    """Fill `piece.sides` (class, profile, Lab strip) in the piece-local frame."""

    @abstractmethod
    def describe(self, piece: Piece) -> Piece:
        raise NotImplementedError


class CompatibilityMatcher(ABC):
    """Build the (N, 4, N, 4) dissimilarity tensor. Classical, Siamese, and GNN all implement this."""

    @abstractmethod
    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        raise NotImplementedError


class Assembler(ABC):
    """Place each piece at most once. Must return the best partial grid if assembly stalls."""

    @abstractmethod
    def assemble(
        self,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        rows: int,
        cols: int,
    ) -> AssemblyState:
        raise NotImplementedError


class ImageReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, puzzle: Puzzle, state: AssemblyState) -> np.ndarray:
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: AssemblyState, gt: GroundTruth) -> dict:
        raise NotImplementedError
