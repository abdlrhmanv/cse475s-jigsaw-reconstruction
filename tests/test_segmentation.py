from pathlib import Path

import numpy as np

from src.core.protocols import ContourTracer, Labeler, PieceExtractor
from src.contour_extraction import (
    MooreContourTracer,
    PieceExtractorImpl,
    YoloBoxExtractor,
    gt_label_path,
)
from src.segmentation import ConnectedComponentLabeler


def test_ccl_is_labeler() -> None:
    assert issubclass(ConnectedComponentLabeler, Labeler)


def test_contour_tracer_protocol() -> None:
    assert issubclass(MooreContourTracer, ContourTracer)


def test_piece_extractor_protocol() -> None:
    assert issubclass(PieceExtractorImpl, PieceExtractor)


def test_ccl_two_blobs():
    """Two separated rectangles should get two distinct labels."""
    binary = np.zeros((30, 60), dtype=np.float64)
    binary[5:15, 5:15] = 255
    binary[5:15, 40:50] = 255
    labels = ConnectedComponentLabeler(min_area=10).label(binary)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 2


def test_ccl_filters_dust():
    binary = np.zeros((30, 30), dtype=np.float64)
    binary[10:20, 10:20] = 255  # 100 px area
    binary[2, 2] = 255           # 1 px dust
    labels = ConnectedComponentLabeler(min_area=5).label(binary)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 1  # dust removed


def test_ccl_drops_thin_blob():
    binary = np.zeros((40, 40), dtype=np.float64)
    binary[5:15, 5:15] = 255       # 10×10 square
    binary[2, 2:38] = 255          # 1×36 bar
    labels = ConnectedComponentLabeler(min_area=5, max_aspect=4.0).label(binary)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 1


def test_ccl_keep_n_largest():
    binary = np.zeros((40, 80), dtype=np.float64)
    binary[2:12, 2:12] = 255       # 100
    binary[2:10, 30:38] = 255      # 64
    binary[2:6, 50:54] = 255       # 16
    labels = ConnectedComponentLabeler(min_area=1, keep_n=2).label(binary)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 2


def test_ccl_drops_low_solidity():
    binary = np.zeros((40, 40), dtype=np.float64)
    binary[2:16, 2:16] = 255  # solid square
    for i in range(12):
        binary[i, 25 + i] = 255  # disconnected diagonal, low solidity
    labels = ConnectedComponentLabeler(min_area=1, min_solidity=0.4).label(binary)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) == 1


def test_moore_traces_square():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    contour = MooreContourTracer().trace(mask)
    assert contour.shape[1] == 2
    assert len(contour) > 0
    # All contour points should be on the boundary of the square
    for x, y in contour:
        assert mask[y, x] > 0


def test_moore_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    contour = MooreContourTracer().trace(mask)
    assert len(contour) == 0


def test_piece_extractor_returns_pieces():
    img = np.random.rand(40, 80, 3) * 255
    binary = np.zeros((40, 80), dtype=np.float64)
    binary[5:15, 5:15] = 255
    binary[5:15, 50:65] = 255
    labels = ConnectedComponentLabeler(min_area=10).label(binary)
    pieces = PieceExtractorImpl(pad=2).extract(img, labels)
    assert len(pieces) == 2
    for p in pieces:
        assert p.image.ndim == 3
        assert p.mask.max() == 255
        assert p.contour.shape[1] == 2


def test_overlay_contours_paints_red(tmp_path):
    from src.core.viz import StageVisualizer

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    binary = np.zeros((20, 20), dtype=np.float64)
    binary[4:12, 4:12] = 255
    labels = ConnectedComponentLabeler(min_area=4).label(binary)
    pieces = PieceExtractorImpl(pad=0).extract(img, labels)
    overlay = StageVisualizer().overlay_contours(img, pieces)
    assert overlay[..., 0].max() == 255
    StageVisualizer().save_json(tmp_path / "metrics.json", {"n": 1})
    assert (tmp_path / "metrics.json").exists()


def test_yolo_box_extractor(tmp_path: Path):
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    img[5:15, 5:15] = 220
    img[5:15, 50:65] = 220
    labels = tmp_path / "boxes.txt"
    # YOLO: class cx cy w h  (normalized)
    labels.write_text(
        "0 0.125 0.25 0.125 0.25\n"
        "1 0.71875 0.25 0.1875 0.25\n"
    )
    pieces = YoloBoxExtractor(pad=1).extract(img, labels)
    assert len(pieces) == 2
    for p in pieces:
        assert p.mask.max() == 255
        assert len(p.contour) >= 4
        assert p.id in (0, 1)


def test_gt_label_path(tmp_path: Path):
    img = tmp_path / "input" / "test" / "board.jpg"
    lbl = tmp_path / "ground_truth" / "test" / "board.txt"
    img.parent.mkdir(parents=True)
    lbl.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    lbl.write_text("0 0.5 0.5 0.1 0.1\n")
    found = gt_label_path(img)
    assert found == lbl
