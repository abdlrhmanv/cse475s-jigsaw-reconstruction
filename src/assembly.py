"""Greedy best-first assembly (Phase 4). Official table always uses this class, not a fancier solver."""

from __future__ import annotations

from src.core.protocols import Assembler, ImageReconstructor
from src.core.types import AssemblyState, CompatibilityTensor, Piece, Puzzle


class GreedyBestFirstAssembler(Assembler):
    def __init__(self, beam_k: int = 3) -> None:
        # Small beam bounds RAM while still recovering from an early dead-end.
        self.beam_k = beam_k

    def assemble(
        self,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        rows: int,
        cols: int,
    ) -> AssemblyState:
        raise NotImplementedError("Phase 4: greedy best-first with rotations and incomplete return.")


class CanvasReconstructor(ImageReconstructor):
    def reconstruct(self, puzzle: Puzzle, state: AssemblyState) -> np.ndarray:
        raise NotImplementedError("Phase 4: paste rotated crops onto a canvas.")
