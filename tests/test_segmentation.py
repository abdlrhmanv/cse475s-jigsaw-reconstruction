import numpy as np

from src.core.protocols import ContourTracer, Labeler, PieceExtractor
from src.contour_extraction import MooreContourTracer, PieceExtractorImpl
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
