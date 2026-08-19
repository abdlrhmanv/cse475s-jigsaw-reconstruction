"""GNN CompatibilityMatcher. Same assembler; D comes from inter-edge probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.core.protocols import CompatibilityMatcher
from src.core.types import CompatibilityTensor, Piece
from src.ml.gnn import PuzzleGNN
from src.ml.gnn_graph import build_side_graph
from src.ml.train_gnn import graph_to_tensors


class GNNCompatibilityMatcher(CompatibilityMatcher):
    """dissim = −log(p + ε) on class-legal inter-piece GNN edges.

    Relative orientation is implied by the matched side indices:
    ``rel_orient = (si - sj) % 4``. Unscored (illegal) pairs stay +inf.
    """

    def __init__(
        self,
        weights: str | Path | None = "checkpoints/gnn.pt",
        device: str | None = None,
        top_k: int = 8,
        eps: float = 1e-6,
        require_weights: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.eps = eps
        self.model = PuzzleGNN().to(self.device)
        self.model.eval()
        if weights is None:
            if require_weights:
                raise FileNotFoundError(
                    "GNN weights=None. Pass a checkpoint or require_weights=False for tests."
                )
            return
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing GNN checkpoint {path}. Train with: python main.py train-gnn"
            )
        blob = torch.load(path, map_location=self.device, weights_only=True)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        self.model.load_state_dict(state)

    @torch.no_grad()
    def build(self, pieces: list[Piece]) -> CompatibilityTensor:
        n = len(pieces)
        dissim = np.full((n, 4, n, 4), np.inf, dtype=np.float64)
        if n == 0:
            return CompatibilityTensor(dissim=dissim)

        graph = build_side_graph(pieces, adjacency=None, top_k=self.top_k)
        ribbons, src, dst, et, _, _inter = graph_to_tensors(graph, self.device)
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
            dissim[i, si, j, sj] = float(-np.log(float(p_np[e]) + self.eps))
        return CompatibilityTensor(dissim=dissim)
