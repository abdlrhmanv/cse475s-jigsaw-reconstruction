import numpy as np
import pytest
import torch

from src.core.protocols import CompatibilityMatcher
from src.core.ribbons import pack_ribbon
from src.core.types import Piece, Side
from src.ml.siamese import SiameseNet
from src.ml.siamese_matcher import SiameseCompatibilityMatcher
from src.ml.train_siamese import make_synthetic_pairs, side_pairs_to_tensors
from src.ml.train_metrics import average_precision, binary_pr_metrics
from src.piece_description import PieceDescriptorImpl


def test_siamese_forward_shapes() -> None:
    net = SiameseNet()
    a = torch.randn(8, 4, 32)
    b = torch.randn(8, 4, 32)
    p, rot, e1, e2 = net(a, b)
    assert p.shape == (8,)
    assert rot.shape == (8, 4)
    assert e1.shape == (8, 128)
    assert e2.shape == (8, 128)
    assert torch.all((p >= 0) & (p <= 1))


def test_pack_ribbon_shape() -> None:
    colour = np.random.rand(20, 3) * 50
    profile = np.linspace(-5, 5, 40)
    r = pack_ribbon(colour, profile)
    assert r.shape == (4, 32)
    assert r.dtype == np.float32


def test_descriptor_fills_ribbon() -> None:
    contour = []
    for x in range(5, 35):
        contour.append((x, 5))
    for y in range(5, 35):
        contour.append((35, y))
    for x in range(35, 5, -1):
        contour.append((x, 35))
    for y in range(35, 5, -1):
        contour.append((5, y))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    piece = Piece(
        id=0,
        image=np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8),
        mask=mask,
        contour=np.array(contour, dtype=np.int32),
        bbox=(5, 5, 35, 35),
        pca_theta=0.0,
        corners=np.empty((4, 2)),
    )
    out = PieceDescriptorImpl().describe(piece)
    assert len(out.sides) == 4
    for s in out.sides:
        assert s.ribbon.shape == (4, 32)


def _piece(pid: int) -> Piece:
    sides = [
        Side(
            index=i,
            cls="tab" if i % 2 == 0 else "blank",
            profile=np.ones(20) * (1 if i % 2 == 0 else -1),
            colour=np.zeros((32, 3)),
            ribbon=pack_ribbon(np.zeros((32, 3)), np.ones(20) * (1 if i % 2 == 0 else -1)),
            contour_pts=np.zeros((10, 2)),
        )
        for i in range(4)
    ]
    return Piece(
        id=pid,
        image=np.zeros((10, 10, 3)),
        mask=np.zeros((10, 10), dtype=np.uint8),
        contour=np.zeros((10, 2), dtype=np.int32),
        bbox=(0, 0, 10, 10),
        pca_theta=0.0,
        corners=np.zeros((4, 2)),
        sides=sides,
    )


def test_siamese_matcher_protocol_and_tensor() -> None:
    assert issubclass(SiameseCompatibilityMatcher, CompatibilityMatcher)
    matcher = SiameseCompatibilityMatcher(weights=None, device="cpu", require_weights=False)
    tensor = matcher.build([_piece(0), _piece(1)])
    assert tensor.dissim.shape == (2, 4, 2, 4)
    assert np.isfinite(tensor.pair(0, 0, 1, 1))  # tab-blank
    assert tensor.pair(0, 0, 0, 1) == np.inf     # self


def test_siamese_missing_checkpoint_fails():
    with pytest.raises(FileNotFoundError, match="Missing Siamese checkpoint"):
        SiameseCompatibilityMatcher(weights="checkpoints/does_not_exist.pt", device="cpu")


def test_synthetic_pairs_balanced() -> None:
    pairs = make_synthetic_pairs(n_pos=10, n_neg=30, rng=np.random.default_rng(0))
    labels = [p[2] for p in pairs]
    assert labels.count(1) == 10
    assert labels.count(0) == 30
    assert pairs[0][0].shape == (4, 32)


def test_side_pairs_label_relative_ori() -> None:
    from src.ml.pair_dataset import SidePair

    pos = SidePair(
        ribbon_a=np.zeros((4, 32), dtype=np.float32),
        ribbon_b=np.zeros((4, 32), dtype=np.float32),
        profile_a=np.ones(8),
        profile_b=np.ones(8),
        label=1,
        piece_id_a=0,
        side_idx_a=1,
        piece_id_b=1,
        side_idx_b=3,
        rel_orient=2,
    )
    neg = SidePair(
        ribbon_a=np.zeros((4, 32), dtype=np.float32),
        ribbon_b=np.zeros((4, 32), dtype=np.float32),
        profile_a=np.ones(8),
        profile_b=np.ones(8),
        label=0,
        piece_id_a=0,
        side_idx_a=1,
        piece_id_b=2,
        side_idx_b=0,
        rel_orient=-1,
    )
    tensors = side_pairs_to_tensors([pos, neg], rot_known=True)
    assert tensors[0][2] == 1 and tensors[0][3] == 2
    assert tensors[1][2] == 0 and tensors[1][3] == 0
    skipped = side_pairs_to_tensors([pos], rot_known=False)
    assert skipped[0][3] == -1


def test_average_precision_perfect_and_random() -> None:
    y = np.array([1, 1, 0, 0], dtype=np.float64)
    assert average_precision(np.array([0.9, 0.8, 0.2, 0.1]), y) > 0.99
    metrics = binary_pr_metrics(np.array([0.2, 0.2, 0.2, 0.2]), y)
    assert metrics["f1"] == 0.0
    assert metrics["p_pos"] == pytest.approx(0.2)
    assert metrics["p_neg"] == pytest.approx(0.2)
