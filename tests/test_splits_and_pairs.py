import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.types import Piece, Side
from src.ml.augmentation import RibbonAugmenter
from src.ml.pair_dataset import PairGenerator, SidePair
from src.ml.splits import check_no_leakage, freeze_splits, load_splits


# -- Splits ------------------------------------------------------------------

def test_freeze_and_load(tmp_path):
    """freeze_splits should scan folders and produce a loadable JSON."""
    for split in ("train", "val", "test"):
        d = tmp_path / "input" / split
        d.mkdir(parents=True)
        for i in range(3 if split == "train" else 1):
            (d / f"img_{split}_{i}.jpg").touch()

    out = tmp_path / "splits.json"
    splits = freeze_splits(tmp_path, out)
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1

    loaded = load_splits(out)
    assert loaded == splits


def test_no_leakage_clean():
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    check_no_leakage(splits)  # should not raise


def test_leakage_detected():
    splits = {"train": ["a", "b"], "val": ["b", "c"], "test": ["d"]}
    with pytest.raises(ValueError, match="Leakage"):
        check_no_leakage(splits)


def test_real_data_no_leakage():
    """Verify the actual dataset has no split leakage."""
    splits_path = Path("data/splits.json")
    if not splits_path.exists():
        data_dir = Path("data")
        if (data_dir / "input" / "train").exists():
            freeze_splits(data_dir, splits_path)
        else:
            pytest.skip("No data directory found")

    splits = load_splits(splits_path)
    check_no_leakage(splits)


# -- Augmentation ------------------------------------------------------------

def test_augmenter_preserves_shape():
    rng = np.random.default_rng(42)
    aug = RibbonAugmenter(rng=rng)
    ribbon = np.random.rand(32, 3) * 255
    result = aug(ribbon)
    assert result.shape == ribbon.shape


def test_augmenter_clips_to_valid_range():
    aug = RibbonAugmenter(jitter_range=300, noise_std=100, rng=np.random.default_rng(0))
    ribbon = np.full((32, 3), 128.0)
    result = aug(ribbon)
    assert result.min() >= 0
    assert result.max() <= 255


# -- Pair generation ---------------------------------------------------------

def _make_piece(pid: int, n_sides: int = 4) -> Piece:
    sides = [
        Side(
            index=i,
            cls="tab" if i % 2 == 0 else "blank",
            profile=np.ones(20) * (1 if i % 2 == 0 else -1),
            colour=np.random.rand(32, 3) * 255,
            ribbon=np.empty(0),
            contour_pts=np.zeros((10, 2)),
        )
        for i in range(n_sides)
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


def test_pair_generator_counts():
    pieces = [_make_piece(i) for i in range(4)]
    adjacency = {
        (0, 1): (1, 3),  # piece 0 side 1 neighbours piece 1 side 3
        (2, 0): (3, 2),
    }
    gen = PairGenerator(neg_ratio=2, rng=np.random.default_rng(42))
    pairs = gen.generate(pieces, adjacency, augment=False)
    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]
    assert len(positives) == 2
    assert len(negatives) == 4  # 2 * neg_ratio


def test_pair_generator_with_augment():
    pieces = [_make_piece(i) for i in range(4)]
    adjacency = {(0, 1): (1, 3)}
    aug = RibbonAugmenter(rng=np.random.default_rng(0))
    gen = PairGenerator(neg_ratio=1, augmenter=aug, rng=np.random.default_rng(0))
    pairs = gen.generate(pieces, adjacency, augment=True)
    assert len(pairs) > 0
    # Augmented ribbons should differ from originals
    pos = [p for p in pairs if p.label == 1][0]
    orig = pieces[0].sides[1].colour
    assert not np.array_equal(pos.ribbon_a, orig)


def test_pair_no_self_pairing():
    pieces = [_make_piece(0)]
    adjacency = {}
    gen = PairGenerator(neg_ratio=5, rng=np.random.default_rng(0))
    pairs = gen.generate(pieces, adjacency, augment=False)
    for p in pairs:
        assert p.piece_id_a != p.piece_id_b or p.label == 1
