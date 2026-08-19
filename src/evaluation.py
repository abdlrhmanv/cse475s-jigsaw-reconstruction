"""Shared metrics for classical vs ML. No assembled-image GT exists.

Headline numbers (do not lead with Q alone):
    identity_neighbour_accuracy, complete_reconstruction, then Q.

Canonical quality formula (every code path, the report, and the notebook):
    Q = 0.5 * position_accuracy + 0.3 * orientation_accuracy + 0.2 * edge_accuracy
Missing components are 0, so a run without pose GT has Q ≤ 0.2.
Completeness and border-flat accuracy are diagnostic only — they are not Q.
Do not invent SSIM against a solved photo.
"""

from __future__ import annotations

import math

import numpy as np

from src.core.protocols import Evaluator
from src.core.types import AssemblyState, GroundTruth, Piece


def canonical_Q(
    pos_acc: float | None,
    ori_acc: float | None,
    edge_acc: float | None,
) -> float:
    """Q = 0.5 * position + 0.3 * orientation + 0.2 * edge.  None → 0."""
    return (
        0.5 * (pos_acc or 0.0)
        + 0.3 * (ori_acc or 0.0)
        + 0.2 * (edge_acc or 0.0)
    )


class ReconstructionEvaluator(Evaluator):
    """Position / neighbour accuracy when a compact identity pose exists.

    Default reconstruction Q on this dataset is identity-neighbour mixed with
    geometry Q. Orientation uses identity+flats on border pieces, not photo rot.
    """

    def evaluate(self, state: AssemblyState, gt: GroundTruth) -> dict:
        rows = len(state.grid)
        cols = len(state.grid[0]) if rows else 0
        n_gt = max(len(gt.pieces), 1)

        pos_correct = 0
        orient_correct = 0
        placed_in_gt = 0

        for r in range(rows):
            for c in range(cols):
                placement = state.grid[r][c]
                if placement is None:
                    continue
                pid = placement.piece_id
                if pid not in gt.pieces:
                    continue
                placed_in_gt += 1
                gt_info = gt.pieces[pid]
                gt_r, gt_c = gt_info["row"], gt_info["col"]
                gt_rot = gt_info.get("rot", 0)
                if placement.row == gt_r and placement.col == gt_c:
                    pos_correct += 1
                    if placement.rot == gt_rot:
                        orient_correct += 1

        edge_total = 0
        edge_correct = 0
        for r in range(rows):
            for c in range(cols):
                p = state.grid[r][c]
                if p is None:
                    continue
                if c + 1 < cols and state.grid[r][c + 1] is not None:
                    edge_total += 1
                    q = state.grid[r][c + 1]
                    if self._gt_adjacent(gt, p.piece_id, q.piece_id, "right"):
                        edge_correct += 1
                if r + 1 < rows and state.grid[r + 1][c] is not None:
                    edge_total += 1
                    q = state.grid[r + 1][c]
                    if self._gt_adjacent(gt, p.piece_id, q.piece_id, "below"):
                        edge_correct += 1

        pos_acc = pos_correct / n_gt
        orient_acc = orient_correct / n_gt
        edge_acc = edge_correct / edge_total if edge_total > 0 else 0.0
        complete = 1.0 if pos_correct == n_gt and orient_correct == n_gt else 0.0
        q_score = canonical_Q(pos_acc, orient_acc, edge_acc)

        return {
            "position_accuracy": pos_acc,
            "orientation_accuracy": orient_acc,
            "edge_accuracy": edge_acc,
            "neighbour_accuracy": edge_acc,
            "complete_reconstruction": complete,
            "Q": q_score,
            "quality_kind": "ground_truth",
            "total_pieces": placed_in_gt,
            "total_dissim": _finite_or_none(state.total_dissim),
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


def geometry_quality(
    state: AssemblyState,
    pieces: list[Piece],
    rows: int,
    cols: int,
) -> dict:
    """Numerical quality when pose GT is unavailable (YOLO boxes are not poses)."""
    n = max(len(pieces), 1)
    by_id = {p.id: p for p in pieces}
    by_index = {i: p for i, p in enumerate(pieces)}

    def piece_of(pid: int) -> Piece | None:
        return by_id.get(pid, by_index.get(pid))

    placed: list = []
    for r in range(rows):
        for c in range(cols):
            if state.grid[r][c] is not None:
                placed.append(state.grid[r][c])
    completeness = min(len(placed) / n, 1.0)

    border_ok = border_n = 0
    legal_ok = legal_n = 0
    for r in range(rows):
        for c in range(cols):
            pl = state.grid[r][c]
            if pl is None:
                continue
            piece = piece_of(pl.piece_id)
            if piece is None or not piece.sides:
                continue
            rot = pl.rot
            for my_dir, dr, dc in ((0, -1, 0), (1, 0, 1), (2, 1, 0), (3, 0, -1)):
                cls = piece.sides[(my_dir - rot) % 4].cls
                nr, nc = r + dr, c + dc
                outside = nr < 0 or nc < 0 or nr >= rows or nc >= cols
                if outside:
                    border_n += 1
                    if cls == "flat":
                        border_ok += 1
                    continue
                if my_dir not in (1, 2):
                    continue
                nbr = state.grid[nr][nc]
                if nbr is None:
                    continue
                other = piece_of(nbr.piece_id)
                if other is None or not other.sides:
                    continue
                nbr_dir = 3 if my_dir == 1 else 0
                ocls = other.sides[(nbr_dir - nbr.rot) % 4].cls
                legal_n += 1
                if {cls, ocls} == {"tab", "blank"}:
                    legal_ok += 1

    border_acc = border_ok / border_n if border_n else 0.0
    match_acc = legal_ok / legal_n if legal_n else 0.0
    q_score = canonical_Q(None, None, match_acc)
    return {
        "position_accuracy": None,
        "orientation_accuracy": None,
        "edge_accuracy": match_acc,
        "neighbour_accuracy": match_acc,
        # Filling the grid is not a correct reconstruction. Complete=1 only after
        # identity/pose checks in the pipeline.
        "complete_reconstruction": 0.0,
        "Q": q_score,
        "quality_kind": "geometry",
        "completeness": completeness,
        "border_flat_accuracy": border_acc,
        "tab_blank_accuracy": match_acc,
        "total_pieces": len(placed),
        "total_dissim": _finite_or_none(state.total_dissim),
    }


def identity_neighbour_quality(
    state: AssemblyState,
    pieces: list[Piece],
    names: list[str] | None = None,
) -> dict:
    """Neighbour accuracy from piece *identities*, not photo poses.

    YOLO classes are printed piece numbers on the completed 7×5 puzzle.
    If two placed pieces share an edge, this checks whether those numbers are
    4-adjacent on that layout. Rotation is not scored. Works for scattered
    subsets (no compact rectangle required).
    """
    from src.ml.pose_gt import canonical_neighbors, load_piece_names, piece_number

    names = names if names is not None else load_piece_names()
    number_of: dict[int, int] = {}
    for piece in pieces:
        if piece.class_id is None:
            continue
        number = piece_number(piece.class_id, names)
        if number is not None:
            number_of[piece.id] = number
    if len(number_of) < 2:
        return {}

    rows = len(state.grid)
    cols = len(state.grid[0]) if rows else 0
    edge_n = 0
    edge_ok = 0
    for r in range(rows):
        for c in range(cols):
            p = state.grid[r][c]
            if p is None:
                continue
            na = number_of.get(p.piece_id)
            if na is None:
                continue
            nbrs = canonical_neighbors(na)
            if c + 1 < cols and state.grid[r][c + 1] is not None:
                nb = number_of.get(state.grid[r][c + 1].piece_id)
                if nb is not None:
                    edge_n += 1
                    if nb in nbrs:
                        edge_ok += 1
            if r + 1 < rows and state.grid[r + 1][c] is not None:
                nb = number_of.get(state.grid[r + 1][c].piece_id)
                if nb is not None:
                    edge_n += 1
                    if nb in nbrs:
                        edge_ok += 1
    acc = edge_ok / edge_n if edge_n else 0.0
    return {
        "identity_neighbour_accuracy": acc,
        "identity_neighbour_n": edge_n,
    }


def identity_orientation_quality(
    state: AssemblyState,
    pieces: list[Piece],
    names: list[str] | None = None,
) -> dict:
    """Orientation accuracy from identity + observed flats, not photo labels.

    A unique rot exists when a piece's flats map onto the 7×5 border directions
    of its printed number (corners and edges). Interior pieces are skipped.
    """
    from src.ml.pose_gt import identity_rotation

    by_id = {piece.id: piece for piece in pieces}
    ok = 0
    n = 0
    for row in state.grid:
        for placement in row:
            if placement is None:
                continue
            piece = by_id.get(placement.piece_id)
            if piece is None:
                continue
            gt_rot = identity_rotation(piece, names=names)
            if gt_rot is None:
                continue
            n += 1
            if int(placement.rot) % 4 == gt_rot:
                ok += 1
    if n == 0:
        return {}
    return {
        "orientation_accuracy": ok / n,
        "orientation_n": n,
        "orientation_note": "identity_flats; interior unscored",
    }


def _finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)
