"""Siamese CompatibilityMatcher. Same assembler as classical; only D changes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.core.protocols import CompatibilityMatcher
from src.core.ribbons import pack_ribbon
from src.core.types import CompatibilityTensor, Piece
from src.ml.siamese import SiameseNet


class SiameseCompatibilityMatcher(CompatibilityMatcher):
    """dissim = −log(p + ε). Illegal class pairs stay +inf (never chosen).

    Relative orientation from the 4-way head is compared to the side-index
    implication ``(si - sj) % 4``; a mismatch down-weights neighbour p.
    """

    def __init__(
        self,
        weights: str | Path | None = "checkpoints/siamese.pt",
        device: str | None = None,
        eps: float = 1e-6,
        require_weights: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.model = SiameseNet().to(self.device)
        self.model.eval()
        if weights is None:
            if require_weights:
                raise FileNotFoundError(
                    "Siamese weights=None. Pass a checkpoint or require_weights=False for tests."
                )
            return
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Siamese checkpoint {path}. Train with: python main.py train-siamese"
            )
        blob = torch.load(path, map_location=self.device, weights_only=True)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        self.model.load_state_dict(state)

    def _ribbon(self, side) -> np.ndarray:
        if side.ribbon is not None and getattr(side.ribbon, "ndim", 0) == 2 and side.ribbon.shape[0] == 4:
            return np.nan_to_num(side.ribbon.astype(np.float32), nan=0.0)
        colour = np.nan_to_num(side.colour, nan=0.0)
        return pack_ribbon(colour, side.profile)

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
                        if sa.cls == "flat" or sb.cls == "flat" or sa.cls == sb.cls:
                            continue
                        batch_a.append(ra)
                        batch_b.append(rb)
                        index.append((i, si, j, sj))

        if not batch_a:
            return CompatibilityTensor(dissim=dissim)

        a = torch.from_numpy(np.stack(batch_a)).to(self.device)
        b = torch.from_numpy(np.stack(batch_b)).to(self.device)
        p, rot_logits, _, _ = self.model(a, b)
        p_np = p.detach().cpu().numpy()
        pred_rot = rot_logits.argmax(dim=-1).detach().cpu().numpy()
        for k, (i, si, j, sj) in enumerate(index):
            pk = float(p_np[k])
            implied = (si - sj) % 4
            if int(pred_rot[k]) != implied:
                pk *= 0.25
            dissim[i, si, j, sj] = float(-np.log(pk + self.eps))
        return CompatibilityTensor(dissim=dissim)
