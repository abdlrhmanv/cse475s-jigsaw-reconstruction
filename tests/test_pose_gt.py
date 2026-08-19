import numpy as np

from src.core.types import Piece
from src.ml.pose_gt import (
    attach_class_ids,
    bbox_iou,
    canonical_cell,
    canonical_flat_dirs,
    canonical_neighbors,
    identity_local_sides,
    identity_rotation,
    piece_number,
    pose_from_pieces,
    relative_orient,
)


NAMES = [
    "1", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "2", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "3", "30", "31", "32", "33", "34", "35", "4", "5", "6", "7", "8", "9",
]


def _piece(pid: int, class_id: int | None, bbox=(0, 0, 10, 10)) -> Piece:
    return Piece(
        id=pid,
        image=np.zeros((4, 4, 3)),
        mask=np.zeros((4, 4), dtype=np.uint8),
        contour=np.zeros((4, 2), dtype=np.int32),
        bbox=bbox,
        pca_theta=0.0,
        corners=np.zeros((4, 2)),
        class_id=class_id,
    )


def test_piece_number_maps_yaml_order():
    assert piece_number(0, NAMES) == 1
    assert piece_number(1, NAMES) == 10
    assert piece_number(11, NAMES) == 2
    assert piece_number(28, NAMES) == 35


def test_canonical_cell_row_major_7x5():
    assert canonical_cell(1) == (0, 0)
    assert canonical_cell(5) == (0, 4)
    assert canonical_cell(6) == (1, 0)
    assert canonical_cell(35) == (6, 4)


def test_canonical_neighbors_interior():
    # Piece 7 is row 1, col 1
    nbrs = canonical_neighbors(7)
    assert nbrs[2] == (0, 2)   # north of 7 is piece 2
    assert nbrs[8] == (1, 3)   # east
    assert nbrs[12] == (2, 0)  # south
    assert nbrs[6] == (3, 1)   # west


def test_pose_compacts_filled_3x3_block():
    # Pieces 1–3, 6–8, 11–13 occupy the top-left 3×3 of the 7×5 board.
    ids = [1, 2, 3, 6, 7, 8, 11, 12, 13]
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    pieces = [_piece(i, class_of[n]) for i, n in enumerate(ids)]
    gt, rows, cols = pose_from_pieces(pieces, names=NAMES)
    assert (rows, cols) == (3, 3)
    assert gt is not None
    assert gt.pieces[0] == {"row": 0, "col": 0, "rot": 0, "pose_note": "canonical_compact"}
    assert gt.pieces[8]["row"] == 2 and gt.pieces[8]["col"] == 2


def test_pose_uncompacted_when_scattered():
    class_of = {int(name): i for i, name in enumerate(NAMES)}
    pieces = [_piece(0, class_of[1]), _piece(1, class_of[35])]
    gt, rows, cols = pose_from_pieces(pieces, names=NAMES)
    assert (rows, cols) == (7, 5)
    assert gt is not None
    assert gt.pieces[0]["pose_note"] == "canonical_7x5_uncompacted"


def test_attach_class_ids_by_iou():
    ccl = [_piece(0, None, bbox=(0, 0, 10, 10)), _piece(1, None, bbox=(20, 20, 30, 30))]
    boxed = [_piece(10, 4, bbox=(1, 1, 11, 11)), _piece(11, 7, bbox=(19, 19, 31, 31))]
    attach_class_ids(ccl, boxed)
    assert ccl[0].class_id == 4
    assert ccl[1].class_id == 7


def test_bbox_iou_identical():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_canonical_flat_dirs():
    assert canonical_flat_dirs(1) == {0, 3}
    assert canonical_flat_dirs(5) == {0, 1}
    assert canonical_flat_dirs(31) == {2, 3}
    assert canonical_flat_dirs(35) == {1, 2}
    assert canonical_flat_dirs(3) == {0}
    assert canonical_flat_dirs(7) == set()


def test_identity_rotation_corners_after_canonical_frame():
    from src.core.types import Side

    def sides(*classes: str):
        return [
            Side(
                index=i,
                cls=cls,  # type: ignore[arg-type]
                profile=np.zeros(4),
                colour=np.zeros((2, 3)),
                ribbon=np.empty(0),
                contour_pts=np.zeros((2, 2)),
            )
            for i, cls in enumerate(classes)
        ]

    class_of = {int(name): i for i, name in enumerate(NAMES)}
    tl = _piece(0, class_of[1])
    tl.sides = sides("flat", "tab", "blank", "flat")
    assert identity_rotation(tl, names=NAMES) == 0

    tr = _piece(1, class_of[5])
    tr.sides = sides("flat", "tab", "blank", "flat")
    assert identity_rotation(tr, names=NAMES) == 1

    interior = _piece(2, class_of[7])
    interior.sides = sides("tab", "blank", "tab", "blank")
    assert identity_rotation(interior, names=NAMES) is None


def test_relative_orient_from_side_indices():
    assert relative_orient(1, 3) == 2
    assert relative_orient(0, 0) == 0
    assert relative_orient(0, 1) == 3


def test_identity_local_sides_top_edge_neighbours():
    from src.core.types import Side

    def sides(*classes: str):
        return [
            Side(
                index=i,
                cls=cls,  # type: ignore[arg-type]
                profile=np.zeros(4),
                colour=np.zeros((2, 3)),
                ribbon=np.empty(0),
                contour_pts=np.zeros((2, 2)),
            )
            for i, cls in enumerate(classes)
        ]

    class_of = {int(name): i for i, name in enumerate(NAMES)}
    a = _piece(0, class_of[1])
    a.sides = sides("flat", "tab", "blank", "flat")
    b = _piece(1, class_of[2])
    b.sides = sides("flat", "tab", "blank", "blank")
    mapped = identity_local_sides(a, b, 1, 3, names=NAMES)
    assert mapped == (1, 3)
    assert relative_orient(*mapped) == 2
