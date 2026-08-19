"""GNN CompatibilityMatcher. Same assembler; D comes from inter-edge probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.core.protocols import CompatibilityMatcher
from src.core.types import CompatibilityTensor, Piece
from src.edge_matching import ILLEGAL_COST
from src.ml.gnn import PuzzleGNN
from src.ml.gnn_graph import build_side_graph
from src.ml.train_gnn import graph_to_tensors


class GNNCompatibilityMatcher(CompatibilityMatcher):
    """dissim = −log(p + ε) on inter-piece GNN edges. Unscored pairs stay inf."""

    def __init__(
        self,
        weights: str | Path | None = "checkpoints/gnn.pt",
        device: str | None = None,
        top_k: int = 8,
        eps: float = 1e-6,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.eps = eps
        self.model = PuzzleGNN().to(self.device)
        self.model.eval()
        if weights is not None and Path(weights).exists():
            blob = torch.load(weights, map_location=self.device, weights_only=True)
            state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            self.model.load_state_dict(state)

    @torch.no_grad()
    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        n = len(pieces)
        dissim = np.full((n, 4, n, 4), np.inf, dtype=np.float64)
        if n == 0:
            return CompatibilityTensor(dissim=dissim)

        graph = build_side_graph(pieces, adjacency=None, top_k=self.top_k)
        ribbons, src, dst, et, _, inter = graph_to_tensors(graph, self.device)
        p, _ = self.model(ribbons, src, dst, et)
        p_np = p.detach().cpu().numpy()
        src_np = graph.edge_src
        dst_np = graph.edge_dst
        types = graph.edge_type

        for e in range(len(src_np)):
            if types[e] != 1:
                continue
            u, v = int(src_np[e]), int(dst_np[e])
            i, si = u // 4, u % 4
            j, sj = v // 4, v % 4
            d = float(-np.log(p_np[e] + self.eps))
            sa = pieces[i].sides[si]
            sb = pieces[j].sides[sj]
            if sa.cls == "flat" or sb.cls == "flat" or sa.cls == sb.cls:
                d += ILLEGAL_COST
            dissim[i, si, j, sj] = d
        return CompatibilityTensor(dissim=dissim)
