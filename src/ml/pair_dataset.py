"""Positive/negative side-pair generation for Siamese training (Phase 5).

A positive pair: two sides from adjacent pieces that are true neighbours in GT.
A negative pair: two sides from non-adjacent pieces (or wrong side combo).

Pairs are generated from already-described pieces (Phase 3 output) using
ground-truth adjacency. Train-only augmentation is applied via RibbonAugmenter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Piece, Side
from src.ml.augmentation import RibbonAugmenter


@dataclass
class SidePair:
    """One training/eval pair for the Siamese network."""
    ribbon_a: np.ndarray   # (N, 3) Lab colour strip or ribbon tensor
    ribbon_b: np.ndarray
    profile_a: np.ndarray  # shape profile
    profile_b: np.ndarray
    label: int             # 1 = positive (true neighbours), 0 = negative
    piece_id_a: int
    side_idx_a: int
    piece_id_b: int
    side_idx_b: int


class PairGenerator:
    """Generate positive and negative pairs from described pieces + GT adjacency.

    Parameters
    ----------
    neg_ratio : Number of negative pairs per positive pair.
    augmenter : Applied only when `augment=True` (train split).
    """

    def __init__(
        self,
        neg_ratio: int = 3,
        augmenter: RibbonAugmenter | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.neg_ratio = neg_ratio
        self.augmenter = augmenter
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        pieces: list[Piece],
        adjacency: dict[tuple[int, int], tuple[int, int]],
        augment: bool = False,
    ) -> list[SidePair]:
        """Generate pairs from described pieces.

        Parameters
        ----------
        pieces : List of pieces with `sides` filled.
        adjacency : Maps (piece_id_a, side_idx_a) → (piece_id_b, side_idx_b)
            for true GT neighbours.
        augment : If True, apply augmentation to ribbons (train only).

        Returns
        -------
        List of SidePair instances (positives + sampled negatives).
        """
        piece_map = {p.id: p for p in pieces}
        positives: list[SidePair] = []
        all_sides: list[tuple[int, int]] = []

        for p in pieces:
            for s in p.sides:
                if s.cls != "flat":
                    all_sides.append((p.id, s.index))

        # Positive pairs
        for (pid_a, si_a), (pid_b, si_b) in adjacency.items():
            if pid_a not in piece_map or pid_b not in piece_map:
                continue
            pa, pb = piece_map[pid_a], piece_map[pid_b]
            if si_a >= len(pa.sides) or si_b >= len(pb.sides):
                continue
            sa, sb = pa.sides[si_a], pb.sides[si_b]
            ra = self._maybe_augment(sa.colour, augment)
            rb = self._maybe_augment(sb.colour, augment)
            positives.append(SidePair(
                ribbon_a=ra, ribbon_b=rb,
                profile_a=sa.profile, profile_b=sb.profile,
                label=1,
                piece_id_a=pid_a, side_idx_a=si_a,
                piece_id_b=pid_b, side_idx_b=si_b,
            ))

        # Negative pairs: sample random non-adjacent side combos
        adj_set = set(adjacency.keys()) | {(v[0], v[1]) for v in adjacency.values()}
        negatives: list[SidePair] = []
        target_neg = len(positives) * self.neg_ratio
        attempts = 0
        max_attempts = target_neg * 10

        while len(negatives) < target_neg and attempts < max_attempts:
            attempts += 1
            idx_a = self.rng.integers(0, len(all_sides))
            idx_b = self.rng.integers(0, len(all_sides))
            pid_a, si_a = all_sides[idx_a]
            pid_b, si_b = all_sides[idx_b]
            if pid_a == pid_b:
                continue
            if (pid_a, si_a) in adj_set and (pid_b, si_b) in adj_set:
                # Check if this specific pair is adjacent
                if adjacency.get((pid_a, si_a)) == (pid_b, si_b):
                    continue
                if adjacency.get((pid_b, si_b)) == (pid_a, si_a):
                    continue

            pa, pb = piece_map[pid_a], piece_map[pid_b]
            sa, sb = pa.sides[si_a], pb.sides[si_b]
            ra = self._maybe_augment(sa.colour, augment)
            rb = self._maybe_augment(sb.colour, augment)
            negatives.append(SidePair(
                ribbon_a=ra, ribbon_b=rb,
                profile_a=sa.profile, profile_b=sb.profile,
                label=0,
                piece_id_a=pid_a, side_idx_a=si_a,
                piece_id_b=pid_b, side_idx_b=si_b,
            ))

        return positives + negatives

    def _maybe_augment(self, colour: np.ndarray, augment: bool) -> np.ndarray:
        if augment and self.augmenter is not None:
            return self.augmenter(colour)
        return colour.copy()
