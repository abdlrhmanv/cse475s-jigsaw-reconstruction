"""Shared-weight Siamese CNN on 4-channel edge ribbons (Phase 6).

No pretrained backbone. Input is (B, 4, L) Lab+profile, not a full-piece ImageNet crop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RibbonEncoder(nn.Module):
    """Conv(4,32,3)-BN-ReLU-MaxPool ×3 → GAP → FC 128."""

    def __init__(self, in_ch: int = 4, emb: int = 128) -> None:
        super().__init__()
        ch = in_ch
        blocks: list[nn.Module] = []
        for out_ch in (32, 32, 32):
            blocks.extend(
                [
                    nn.Conv1d(ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
            )
            ch = out_ch
        self.conv = nn.Sequential(*blocks)
        self.fc = nn.Linear(ch, emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.fc(h)


class SiameseNet(nn.Module):
    """Twin encoder. Neighbour probability + 4-way relative orientation."""

    def __init__(self, in_ch: int = 4, emb: int = 128) -> None:
        super().__init__()
        self.encoder = RibbonEncoder(in_ch=in_ch, emb=emb)
        self.neighbour = nn.Linear(emb, 1)
        self.orient = nn.Linear(emb, 4)

    def embed(self, ribbon: torch.Tensor) -> torch.Tensor:
        return self.encoder(ribbon)

    def forward(
        self,
        ribbon_a: torch.Tensor,
        ribbon_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.embed(ribbon_a)
        e2 = self.embed(ribbon_b)
        h = torch.abs(e1 - e2)
        logit = self.neighbour(h).squeeze(-1)
        p = torch.sigmoid(logit)
        rot_logits = self.orient(h)
        return p, rot_logits, e1, e2
