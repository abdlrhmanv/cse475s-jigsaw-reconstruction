import numpy as np

from src.core.protocols import CompatibilityMatcher
from src.core.types import Piece, Side
from src.edge_matching import ILLEGAL_COST, ClassicalCompatibilityMatcher


def test_classical_matcher_protocol() -> None:
    assert issubclass(ClassicalCompatibilityMatcher, CompatibilityMatcher)


def _make_piece(pid: int, profiles: list[np.ndarray], classes: list[str]) -> Piece:
    sides = []
    for i, (prof, cls) in enumerate(zip(profiles, classes)):
        sides.append(Side(
            index=i, cls=cls, profile=prof,
            colour=np.zeros((32, 3)), ribbon=np.empty(0),
            contour_pts=np.zeros((10, 2)),
        ))
    return Piece(
        id=pid, image=np.zeros((10, 10, 3)), mask=np.zeros((10, 10), dtype=np.uint8),
        contour=np.zeros((10, 2), dtype=np.int32), bbox=(0, 0, 10, 10),
        pca_theta=0.0, corners=np.zeros((4, 2)), sides=sides,
    )


def test_self_pairing_is_inf():
    p = _make_piece(0, [np.ones(20)] * 4, ["tab", "blank", "tab", "blank"])
    tensor = ClassicalCompatibilityMatcher().build([p])
    assert tensor.pair(0, 0, 0, 1) == np.inf


def test_tab_blank_finite():
    p0 = _make_piece(0, [np.ones(20), -np.ones(20), np.ones(20), -np.ones(20)],
                      ["tab", "blank", "tab", "blank"])
    p1 = _make_piece(1, [-np.ones(20), np.ones(20), -np.ones(20), np.ones(20)],
                      ["blank", "tab", "blank", "tab"])
    tensor = ClassicalCompatibilityMatcher().build([p0, p1])
    # tab(0,0) vs blank(1,0) should be finite
    assert tensor.pair(0, 0, 1, 0) < np.inf


def test_same_class_is_penalised():
    p0 = _make_piece(0, [np.ones(20)] * 4, ["tab"] * 4)
    p1 = _make_piece(1, [np.ones(20)] * 4, ["tab"] * 4)
    tensor = ClassicalCompatibilityMatcher().build([p0, p1])
    assert tensor.pair(0, 0, 1, 0) >= ILLEGAL_COST  # tab-tab


def test_flat_sides_are_penalised():
    p0 = _make_piece(0, [np.zeros(20)] * 4, ["flat", "tab", "flat", "blank"])
    p1 = _make_piece(1, [np.zeros(20)] * 4, ["flat", "blank", "flat", "tab"])
    tensor = ClassicalCompatibilityMatcher().build([p0, p1])
    assert tensor.pair(0, 0, 1, 0) >= ILLEGAL_COST  # flat side
