"""Classical side-pair dissimilarity (Phase 4). Output is lower-better D, not a probability."""

from __future__ import annotations

import numpy as np

from src.core.protocols import CompatibilityMatcher
from src.core.types import CompatibilityTensor, Piece


def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    """Resample a 1-D or 2-D array to length n along axis 0 via linear interpolation."""
    m = len(arr)
    if m == n:
        return arr.copy()
    if m == 0:
        return np.zeros((n,) + arr.shape[1:])
    indices = np.linspace(0, m - 1, n)
    floor_idx = np.floor(indices).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, m - 1)
    frac = indices - floor_idx
    if arr.ndim == 1:
        return arr[floor_idx] * (1 - frac) + arr[ceil_idx] * frac
    frac = frac[:, np.newaxis]
    return arr[floor_idx] * (1 - frac) + arr[ceil_idx] * frac


def _shape_dissimilarity(profile_a: np.ndarray, profile_b: np.ndarray, n: int = 64) -> float:
    """Shape dissimilarity: sum of (p_a + reversed p_b)².

    Complementary sides should sum to ~0 (tab fills blank).
    """
    pa = _resample(profile_a, n)
    pb = _resample(profile_b, n)[::-1]
    diff = pa + pb  # complementary = 0
    return float(np.mean(diff ** 2))


def _colour_dissimilarity(colour_a: np.ndarray, colour_b: np.ndarray, n: int = 32) -> float:
    """Mean squared Lab distance between two colour strips (reversed alignment)."""
    ca = _resample(colour_a, n)
    cb = _resample(colour_b, n)[::-1]
    valid = (np.abs(ca).sum(axis=1) > 1e-6) & (np.abs(cb).sum(axis=1) > 1e-6)
    if not np.any(valid):
        return 0.0
    diff = ca[valid] - cb[valid]
    return float(np.mean(np.sum(diff ** 2, axis=1)))


ILLEGAL_COST = 1.0e6


class ClassicalCompatibilityMatcher(CompatibilityMatcher):
    """Build (N, 4, N, 4) dissimilarity tensor using shape + colour.

    D(i, si, j, sj) = ws * shape_dissim + wc * colour_dissim.
    Self-pairs stay inf. Same-class (tab-tab / blank-blank) and flat sides
    get a large finite penalty so the assembler can still fill every cell.
    """

    def __init__(self, ws: float = 0.3, wc: float = 0.7) -> None:
        self.ws = ws
        self.wc = wc

    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        n = len(pieces)
        dissim = np.full((n, 4, n, 4), np.inf, dtype=np.float64)

        for i in range(n):
            if not pieces[i].sides:
                continue
            for si in range(4):
                side_a = pieces[i].sides[si]
                for j in range(n):
                    if i == j:
                        continue
                    if not pieces[j].sides:
                        continue
                    for sj in range(4):
                        side_b = pieces[j].sides[sj]
                        ds = _shape_dissimilarity(side_a.profile, side_b.profile)
                        dc = _colour_dissimilarity(side_a.colour, side_b.colour)
                        cost = self.ws * ds + self.wc * dc
                        if side_a.cls == "flat" or side_b.cls == "flat":
                            cost += ILLEGAL_COST
                        elif side_a.cls == side_b.cls:
                            cost += ILLEGAL_COST
                        dissim[i, si, j, sj] = cost

        return CompatibilityTensor(dissim=dissim)
