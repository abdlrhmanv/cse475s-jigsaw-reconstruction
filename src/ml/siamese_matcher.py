"""Siamese CompatibilityMatcher. Same assembler as classical; only D changes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.core.protocols import CompatibilityMatcher
from src.core.ribbons import pack_ribbon
from src.core.types import CompatibilityTensor, Piece
from src.edge_matching import ILLEGAL_COST
from src.ml.siamese import SiameseNet


class SiameseCompatibilityMatcher(CompatibilityMatcher):
    """dissim = −log(p + ε). Flat / same-class pairs still get ILLEGAL_COST."""

    def __init__(
        self,
        weights: str | Path | None = "checkpoints/siamese.pt",
        device: str | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.model = SiameseNet().to(self.device)
        self.model.eval()
        if weights is not None and Path(weights).exists():
            blob = torch.load(weights, map_location=self.device, weights_only=True)
            state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            self.model.load_state_dict(state)

    def _ribbon(self, side) -> np.ndarray:
        if side.ribbon is not None and getattr(side.ribbon, "ndim", 0) == 2 and side.ribbon.shape[0] == 4:
            return side.ribbon.astype(np.float32)
        return pack_ribbon(side.colour, side.profile)

    @torch.no_grad()
    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        n = len(pieces)
        dissim = np.full((n, 4, n, 4), np.inf, dtype=np.float64)
        ribbons: list[list[np.ndarray | None]] = []
        for p in pieces:
            row: list[np.ndarray | None] = [None] * 4
            if p.sides:
                for s in p.sides:
                    row[s.index] = self._ribbon(s)
            ribbons.append(row)

        batch_a: list[np.ndarray] = []
        batch_b: list[np.ndarray] = []
        extras: list[float] = []
        index: list[tuple[int, int, int, int]] = []
        for i in range(n):
            if not pieces[i].sides:
                continue
            for si, sa in enumerate(pieces[i].sides):
                ra = ribbons[i][si]
                if ra is None:
                    continue
                for j in range(n):
                    if i == j or not pieces[j].sides:
                        continue
                    for sj, sb in enumerate(pieces[j].sides):
                        rb = ribbons[j][sj]
                        if rb is None:
                            continue
                        extra = 0.0
                        if sa.cls == "flat" or sb.cls == "flat" or sa.cls == sb.cls:
                            extra = ILLEGAL_COST
                        batch_a.append(ra)
                        batch_b.append(rb)
                        extras.append(extra)
                        index.append((i, si, j, sj))

        if not batch_a:
            return CompatibilityTensor(dissim=dissim)

        a = torch.from_numpy(np.stack(batch_a)).to(self.device)
        b = torch.from_numpy(np.stack(batch_b)).to(self.device)
        p, _, _, _ = self.model(a, b)
        p_np = p.detach().cpu().numpy()
        for k, (i, si, j, sj) in enumerate(index):
            dissim[i, si, j, sj] = float(-np.log(p_np[k] + self.eps)) + extras[k]
        return CompatibilityTensor(dissim=dissim)
