import numpy as np
import torch

from src.core.protocols import CompatibilityMatcher
from src.ml.gnn import PuzzleGNN
from src.ml.gnn_graph import build_side_graph
from src.ml.gnn_matcher import GNNCompatibilityMatcher
from src.ml.train_gnn import graph_to_tensors, make_synthetic_grid


def test_synthetic_grid_adjacency():
    pieces, adj = make_synthetic_grid(2, 2, rng=np.random.default_rng(0))
    assert len(pieces) == 4
    # 2x2: 4 internal undirected edges recorded once each direction in adj keys
    assert len(adj) == 4
    assert pieces[0].is_corner or sum(s.cls == "flat" for s in pieces[0].sides) == 2


def test_graph_has_intra_and_inter():
    pieces, adj = make_synthetic_grid(3, 3, rng=np.random.default_rng(1))
    g = build_side_graph(pieces, adjacency=adj, top_k=4)
    assert g.ribbons.shape[0] == 36
    assert np.any(g.edge_type == 0)
    assert np.any(g.edge_type == 1)
    assert g.labels is not None
    assert g.labels[g.inter_mask].sum() >= 1


def test_gnn_forward_shapes():
    pieces, adj = make_synthetic_grid(2, 2, rng=np.random.default_rng(2))
    g = build_side_graph(pieces, adjacency=adj, top_k=4)
    model = PuzzleGNN(dim=32)
    ribbons, src, dst, et, y, inter = graph_to_tensors(g, "cpu")
    p, h = model(ribbons, src, dst, et)
    assert p.shape[0] == src.shape[0]
    assert h.shape == (16, 32)
    assert torch.all((p >= 0) & (p <= 1))


def test_gnn_matcher_protocol():
    assert issubclass(GNNCompatibilityMatcher, CompatibilityMatcher)
    pieces, _ = make_synthetic_grid(2, 2, rng=np.random.default_rng(3))
    tensor = GNNCompatibilityMatcher(weights=None, device="cpu", top_k=4, require_weights=False).build(pieces)
    assert tensor.dissim.shape == (4, 4, 4, 4)
    # some inter pair should be finite
    finite = np.isfinite(tensor.dissim)
    assert finite.any()
    # self pairs remain inf
    for i in range(4):
        assert not np.isfinite(tensor.dissim[i, :, i, :]).any()
