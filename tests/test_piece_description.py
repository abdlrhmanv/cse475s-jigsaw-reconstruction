import numpy as np

from src.core.protocols import CornerFinder, PieceDescriptor
from src.core.types import Piece
from src.piece_description import (
    CurvatureCornerFinder,
    HybridCornerFinder,
    PieceDescriptorImpl,
    _classify_side,
    _signed_profile,
)


def test_description_protocols() -> None:
    assert issubclass(HybridCornerFinder, CornerFinder)
    assert issubclass(CurvatureCornerFinder, CornerFinder)
    assert issubclass(PieceDescriptorImpl, PieceDescriptor)


def _make_square_contour(x0: int, y0: int, size: int) -> np.ndarray:
    """Generate a clockwise square contour as (M, 2) array."""
    pts: list[tuple[int, int]] = []
    for x in range(x0, x0 + size):
        pts.append((x, y0))
    for y in range(y0, y0 + size):
        pts.append((x0 + size, y))
    for x in range(x0 + size, x0, -1):
        pts.append((x, y0 + size))
    for y in range(y0 + size, y0, -1):
        pts.append((x0, y))
    return np.array(pts, dtype=np.int32)


def test_hybrid_finds_four_corners():
    contour = _make_square_contour(10, 10, 40)
    corners = HybridCornerFinder().find(contour)
    assert corners.shape == (4, 2)


def test_curvature_finds_four_corners():
    contour = _make_square_contour(10, 10, 40)
    corners = CurvatureCornerFinder().find(contour)
    assert corners.shape == (4, 2)


def test_signed_profile_flat_side():
    side = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.int32)
    profile = _signed_profile(side)
    np.testing.assert_allclose(profile, 0.0, atol=1e-10)


def test_classify_flat():
    profile = np.zeros(20)
    assert _classify_side(profile) == "flat"


def test_classify_tab():
    profile = np.full(20, 10.0)
    assert _classify_side(profile) == "tab"


def test_classify_blank():
    profile = np.full(20, -10.0)
    assert _classify_side(profile) == "blank"


def _make_dummy_piece() -> Piece:
    contour = _make_square_contour(5, 5, 30)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    image = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    return Piece(
        id=1,
        image=image,
        mask=mask,
        contour=contour,
        bbox=(5, 5, 35, 35),
        pca_theta=0.0,
        corners=np.empty((4, 2)),
    )


def test_descriptor_fills_four_sides():
    piece = _make_dummy_piece()
    desc = PieceDescriptorImpl()
    result = desc.describe(piece)
    assert len(result.sides) == 4
    for side in result.sides:
        assert side.cls in ("tab", "blank", "flat")
        assert side.colour.shape == (32, 3)


def test_descriptor_square_is_all_flat():
    """A perfect square contour should have all flat sides."""
    piece = _make_dummy_piece()
    desc = PieceDescriptorImpl()
    result = desc.describe(piece)
    flat_count = sum(1 for s in result.sides if s.cls == "flat")
    assert flat_count == 4
    assert result.is_corner is True


def test_deskew_uprights_tilted_rectangle():
    """A 30°-tilted rectangle should be axis-aligned after describe()."""
    from src.contour_extraction import MooreContourTracer, _rotate_nn, deskew_piece

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 25:55] = 255
    tilted = _rotate_nn(mask, np.deg2rad(30))
    contour = MooreContourTracer().trace(tilted)
    piece = Piece(
        id=0,
        image=np.stack([tilted, tilted, tilted], axis=-1),
        mask=tilted,
        contour=contour,
        bbox=(0, 0, 79, 79),
        pca_theta=0.0,
        corners=np.empty((4, 2)),
    )
    out = deskew_piece(piece)
    ys, xs = np.nonzero(out.mask)
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    # Axis-aligned 30×40 rectangle: aspect closer to 30/40 than to a diamond.
    assert min(width, height) / max(width, height) < 0.85

