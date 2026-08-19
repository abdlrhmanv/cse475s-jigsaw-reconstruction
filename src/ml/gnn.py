"""Side-node GraphSAGE (Phase 7). Relational: messages mix intra-piece and inter-piece edges.

This is not a Siamese net with extra convs. Each side's embedding is updated from
its other three sides and from competing candidate matches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.siamese import RibbonEncoder


class SageLayer(nn.Module):
    """Mean-aggregate neighbours, separate weights for intra vs inter edge types."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_intra = nn.Linear(dim, dim)
        self.w_inter = nn.Linear(dim, dim)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        n, dim = h.shape
        intra = edge_type == 0
        inter = edge_type == 1
        agg_intra = self._mean_agg(h, src[intra], dst[intra], n, dim)
        agg_inter = self._mean_agg(h, src[inter], dst[inter], n, dim)
        out = self.w_self(h) + self.w_intra(agg_intra) + self.w_inter(agg_inter)
        return F.relu(out)

    @staticmethod
    def _mean_agg(
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n: int,
        dim: int,
    ) -> torch.Tensor:
        agg = torch.zeros(n, dim, device=h.device, dtype=h.dtype)
        deg = torch.zeros(n, 1, device=h.device, dtype=h.dtype)
        if src.numel() == 0:
            return agg
        agg.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones(src.shape[0], 1, device=h.device, dtype=h.dtype))
        return agg / deg.clamp(min=1.0)


class PuzzleGNN(nn.Module):
    """Encode ribbons → 2 Sage layers → inter-edge neighbour probability."""

    def __init__(self, in_ch: int = 4, dim: int = 64, n_layers: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = RibbonEncoder(in_ch=in_ch, emb=dim)
        self.layers = nn.ModuleList([SageLayer(dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def encode_nodes(self, ribbons: torch.Tensor) -> torch.Tensor:
        return self.encoder(ribbons)

    def propagate(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = self.drop(layer(h, src, dst, edge_type))
        return h

    def edge_prob(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        hu, hv = h[src], h[dst]
        feat = torch.cat(
            [hu, hv, torch.abs(hu - hv), edge_type.float().unsqueeze(-1)],
            dim=-1,
        )
        return torch.sigmoid(self.edge_mlp(feat).squeeze(-1))

    def forward(
        self,
        ribbons: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.propagate(self.encode_nodes(ribbons), src, dst, edge_type)
        p = self.edge_prob(h, src, dst, edge_type)
        return p, h
