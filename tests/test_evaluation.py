import numpy as np
import pytest

from src.core.types import AssemblyState, Placement, Piece, Side
from src.evaluation import canonical_Q, identity_neighbour_quality, identity_orientation_quality
from src.ml.pose_gt import weak_side_adjacency


def test_canonical_q_is_the_only_formula():
    """Lock Q = 0.5 pos + 0.3 ori + 0.2 edge; missing terms are 0."""
    assert canonical_Q(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert canonical_Q(None, None, 0.2) == pytest.approx(0.04)
    assert canonical_Q(0.0, 0.0, 1.0) == pytest.approx(0.2)
    assert canonical_Q(1.0, None, 0.2) == pytest.approx(0.54)
    # Completeness/border must not be smuggled into Q.
    assert canonical_Q(None, None, 1.0) < 0.21


NAMES = [
    "1", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "2", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "3", "30", "31", "32", "33", "34", "35", "4", "5", "6", "7", "8", "9",
]


def _side(index: int, cls: str, bump: float) -> Side:
    prof = np.ones(16) * bump
    return Side(
        index=index,
        cls=cls,  # type: ignore[arg-type]
        profile=prof,
        colour=np.zeros((8, 3)),
        ribbon=np.empty(0),
        contour_pts=np.zeros((4, 2)),
    )


def _piece(pid: int, class_id: int, sides: list[Side]) -> Piece:
    return Piece(
        id=pid,
        image=np.zeros((4, 4, 3)),
        mask=np.zeros((4, 4), dtype=np.uint8),
        contour=np.zeros((4, 2), dtype=np.int32),
        bbox=(0, 0, 4, 4),
        pca_theta=0.0,
        corners=np.zeros((4, 2)),
        sides=sides,
        class_id=class_id,
    )


def test_identity_neighbour_true_and_false():
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    a = _piece(0, class_of[1], [_side(i, "flat", 0) for i in range(4)])
    b = _piece(1, class_of[2], [_side(i, "flat", 0) for i in range(4)])
    grid = [
        [Placement(0, 0, 0, 0), Placement(1, 0, 1, 0)],
    ]
    state = AssemblyState(grid=grid, used={0, 1})
    metrics = identity_neighbour_quality(state, [a, b], names=NAMES)
    assert metrics["identity_neighbour_accuracy"] == 1.0

    far = _piece(1, class_of[35], [_side(i, "flat", 0) for i in range(4)])
    state.grid[0][1] = Placement(1, 0, 1, 0)
    metrics = identity_neighbour_quality(state, [a, far], names=NAMES)
    assert metrics["identity_neighbour_accuracy"] == 0.0


def test_weak_adjacency_unique_legal_pair():
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    east = [_side(0, "flat", 0), _side(1, "tab", 4.0), _side(2, "flat", 0), _side(3, "flat", 0)]
    west = [_side(0, "flat", 0), _side(1, "flat", 0), _side(2, "flat", 0), _side(3, "blank", -4.0)]
    pieces = [
        _piece(0, class_of[1], east),
        _piece(1, class_of[2], west),
    ]
    adj = weak_side_adjacency(pieces, names=NAMES)
    assert adj[(0, 1)] == (1, 3)


def test_weak_adjacency_prefers_identity_mapped_sides():
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    # Two legal tab↔blank pairs; identity+flats must pick E↔W (1, 3).
    a = _piece(
        0,
        class_of[1],
        [_side(0, "flat", 0), _side(1, "tab", 4), _side(2, "blank", -8), _side(3, "flat", 0)],
    )
    b = _piece(
        1,
        class_of[2],
        [_side(0, "flat", 0), _side(1, "tab", 8), _side(2, "blank", -4), _side(3, "blank", -1)],
    )
    adj = weak_side_adjacency([a, b], names=NAMES)
    assert adj[(0, 1)] == (1, 3)


def test_identity_orientation_border_piece():
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    # Canonical NW flats: piece 1 wants rot=0, piece 5 wants rot=1.
    tl = _piece(0, class_of[1], [_side(0, "flat", 0), _side(1, "tab", 4), _side(2, "blank", -4), _side(3, "flat", 0)])
    tr = _piece(1, class_of[5], [_side(0, "flat", 0), _side(1, "tab", 4), _side(2, "blank", -4), _side(3, "flat", 0)])
    grid = [[Placement(0, 0, 0, 0), Placement(1, 0, 1, 1)]]
    state = AssemblyState(grid=grid, used={0, 1})
    metrics = identity_orientation_quality(state, [tl, tr], names=NAMES)
    assert metrics["orientation_n"] == 2
    assert metrics["orientation_accuracy"] == 1.0

    state.grid[0][1] = Placement(1, 0, 1, 0)
    metrics = identity_orientation_quality(state, [tl, tr], names=NAMES)
    assert metrics["orientation_accuracy"] == 0.5
