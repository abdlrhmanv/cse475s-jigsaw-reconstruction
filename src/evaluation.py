"""Shared metrics for classical vs ML (Phase 4). Align the predicted board to GT before scoring."""

from __future__ import annotations

from src.core.protocols import Evaluator
from src.core.types import AssemblyState, GroundTruth


class ReconstructionEvaluator(Evaluator):
    def evaluate(self, state: AssemblyState, gt: GroundTruth) -> dict:
        raise NotImplementedError("Phase 4: position, orientation, edge, Q.")
