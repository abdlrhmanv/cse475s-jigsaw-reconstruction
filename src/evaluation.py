"""Shared metrics for classical vs ML (Phase 4). Evaluate assembly against ground truth."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Evaluator
from src.core.types import AssemblyState, GroundTruth


class ReconstructionEvaluator(Evaluator):
    """Compute position accuracy, orientation accuracy, neighbour-edge accuracy, and Q.

    Metrics (all in [0, 1], higher is better):
    - position_accuracy: fraction of pieces placed at correct (row, col)
    - orientation_accuracy: fraction of correctly-placed pieces also at correct rotation
    - edge_accuracy: fraction of adjacent pairs whose relative placement matches GT
    - Q: harmonic mean of the above three
    """

    def evaluate(self, state: AssemblyState, gt: GroundTruth) -> dict:
        total = 0
        pos_correct = 0
        orient_correct = 0
        edge_total = 0
        edge_correct = 0

        rows = len(state.grid)
        cols = len(state.grid[0]) if rows else 0

        for r in range(rows):
            for c in range(cols):
                placement = state.grid[r][c]
                if placement is None:
                    continue
                pid = placement.piece_id
                if pid not in gt.pieces:
                    continue
                total += 1
                gt_info = gt.pieces[pid]
                gt_r, gt_c = gt_info["row"], gt_info["col"]
                gt_rot = gt_info.get("rot", 0)

                if placement.row == gt_r and placement.col == gt_c:
                    pos_correct += 1
                    if placement.rot == gt_rot:
                        orient_correct += 1

        # Edge accuracy: check each horizontal and vertical adjacency
        for r in range(rows):
            for c in range(cols):
                p = state.grid[r][c]
                if p is None:
                    continue
                # Right neighbour
                if c + 1 < cols and state.grid[r][c + 1] is not None:
                    edge_total += 1
                    q = state.grid[r][c + 1]
                    if self._gt_adjacent(gt, p.piece_id, q.piece_id, "right"):
                        edge_correct += 1
                # Bottom neighbour
                if r + 1 < rows and state.grid[r + 1][c] is not None:
                    edge_total += 1
                    q = state.grid[r + 1][c]
                    if self._gt_adjacent(gt, p.piece_id, q.piece_id, "below"):
                        edge_correct += 1

        pos_acc = pos_correct / total if total > 0 else 0.0
        orient_acc = orient_correct / total if total > 0 else 0.0
        edge_acc = edge_correct / edge_total if edge_total > 0 else 0.0

        # Q: harmonic mean (avoid division by zero)
        vals = [v for v in (pos_acc, orient_acc, edge_acc) if v > 0]
        q_score = len(vals) / sum(1.0 / v for v in vals) if vals else 0.0

        return {
            "position_accuracy": pos_acc,
            "orientation_accuracy": orient_acc,
            "edge_accuracy": edge_acc,
            "Q": q_score,
            "total_pieces": total,
            "total_dissim": state.total_dissim,
        }

    @staticmethod
    def _gt_adjacent(gt: GroundTruth, pid_a: int, pid_b: int, direction: str) -> bool:
        """Check if pid_a and pid_b are adjacent in the given direction in GT."""
        if pid_a not in gt.pieces or pid_b not in gt.pieces:
            return False
        ga = gt.pieces[pid_a]
        gb = gt.pieces[pid_b]
        if direction == "right":
            return ga["row"] == gb["row"] and ga["col"] + 1 == gb["col"]
        if direction == "below":
            return ga["row"] + 1 == gb["row"] and ga["col"] == gb["col"]
        return False
