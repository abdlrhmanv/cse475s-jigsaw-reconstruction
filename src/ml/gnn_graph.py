"""Build a side-node graph from described pieces (GNN Formulation A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.ribbons import pack_ribbon
from src.core.types import Piece


@dataclass
class SideGraph:
    """One puzzle as a graph: 4N side-nodes, intra-piece cycles + inter candidates."""

    n_pieces: int
    ribbons: np.ndarray          # (4N, 4, L)
    edge_src: np.ndarray         # (E,)
    edge_dst: np.ndarray
    edge_type: np.ndarray        # 0 = intra-piece, 1 = inter-piece
    inter_mask: np.ndarray       # bool (E,) True if inter
    labels: np.ndarray | None    # (E,) 1 if true neighbour, else 0; None at inference
    node_piece: np.ndarray       # (4N,) piece index
    node_side: np.ndarray        # (4N,) side 0..3


def _ribbon(side) -> np.ndarray:
    if side.ribbon is not None and getattr(side.ribbon, "ndim", 0) == 2 and side.ribbon.shape[0] == 4:
        out = side.ribbon.astype(np.float32)
    else:
        out = pack_ribbon(side.colour, side.profile)
    return np.nan_to_num(out, nan=0.0)


def _colour_ssd(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b), 32)
    if n == 0:
        return 1e9
    ca, cb = a[:n], b[:n][::-1]
    return float(np.mean((ca - cb) ** 2))


def build_side_graph(
    pieces: list[Piece],
    adjacency: dict[tuple[int, int], tuple[int, int]] | None = None,
    top_k: int = 8,
) -> SideGraph:
    """Nodes = piece sides. Intra edges form 4-cycles. Inter edges are class-legal top-K.

    `adjacency` maps (piece_id, side) → neighbour (piece_id, side) for training labels.
    Piece ids are assumed to match list indices 0..N-1.
    """
    n = len(pieces)
    n_nodes = n * 4
    ribbons = np.zeros((n_nodes, 4, 32), dtype=np.float32)
    node_piece = np.repeat(np.arange(n), 4)
    node_side = np.tile(np.arange(4), n)
    classes: list[str] = ["tab"] * n_nodes

    for i, p in enumerate(pieces):
        for s in p.sides:
            nid = i * 4 + s.index
            ribbons[nid] = _ribbon(s)
            classes[nid] = s.cls

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []
    labels: list[int] = []

    # Intra-piece undirected cycle
    for i in range(n):
        for s in range(4):
            u = i * 4 + s
            v = i * 4 + ((s + 1) % 4)
            src.extend([u, v])
            dst.extend([v, u])
            etype.extend([0, 0])
            labels.extend([0, 0])

    pos_set: set[tuple[int, int]] = set()
    if adjacency:
        for (pi, si), (pj, sj) in adjacency.items():
            if 0 <= pi < n and 0 <= pj < n:
                pos_set.add((pi * 4 + si, pj * 4 + sj))
                pos_set.add((pj * 4 + sj, pi * 4 + si))

    # Inter-piece candidates: all class-legal pairs (top_k<=0) or colour top-K
    for i in range(n):
        for si in range(4):
            u = i * 4 + si
            if classes[u] == "flat":
                continue
            scored: list[tuple[float, int]] = []
            for j in range(n):
                if j == i:
                    continue
                for sj in range(4):
                    v = j * 4 + sj
                    if classes[v] == "flat" or classes[v] == classes[u]:
                        continue
                    sa = pieces[i].sides[si].colour if si < len(pieces[i].sides) else np.zeros((1, 3))
                    sb = pieces[j].sides[sj].colour if sj < len(pieces[j].sides) else np.zeros((1, 3))
                    scored.append((_colour_ssd(sa, sb), v))
            scored.sort()
            chosen = scored if top_k is None or top_k <= 0 else scored[:top_k]
            for _, v in chosen:
                src.append(u)
                dst.append(v)
                etype.append(1)
                labels.append(1 if (u, v) in pos_set else 0)

    return SideGraph(
        n_pieces=n,
        ribbons=ribbons,
        edge_src=np.array(src, dtype=np.int64),
        edge_dst=np.array(dst, dtype=np.int64),
        edge_type=np.array(etype, dtype=np.int64),
        inter_mask=np.array(etype, dtype=np.int64) == 1,
        labels=np.array(labels, dtype=np.float32) if adjacency is not None else None,
        node_piece=node_piece,
        node_side=node_side,
    )
