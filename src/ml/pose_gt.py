"""Canonical pose from YOLO piece IDs (1–35 on a 7×5 board).

The Roboflow classes are physical piece numbers, not (row, col, rot) in the
scrambled photo. Position GT is only claimed when the photographed pieces
occupy a filled rectangle on the completed 7×5 layout (then compacted).
Rotation in the photo is unknown. Placement orientation for border pieces is
recovered from printed identity plus observed flats (unique 90° that maps those
flats onto the 7×5 border). Interior pieces have no orientation GT.
Neighbour *identities* follow the 7×5 numbering. Matching *sides* are labelled
only when a unique tab–blank pair exists; otherwise shape complementarity
(E_shape) is a weak tie-break — not classical D.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.types import GroundTruth, Piece
from src.edge_matching import _shape_dissimilarity

CANON_ROWS = 7
CANON_COLS = 5
DATA_YAML = Path("data/data.yaml")


def load_piece_names(path: str | Path = DATA_YAML) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    names = payload.get("names") or []
    return [str(n) for n in names]


def piece_number(class_id: int, names: list[str] | None = None) -> int | None:
    """YOLO class index → printed piece id 1..35."""
    names = names if names is not None else load_piece_names()
    if class_id < 0 or class_id >= len(names):
        return None
    try:
        value = int(names[class_id])
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 35 else None


def canonical_cell(piece_num: int, rows: int = CANON_ROWS, cols: int = CANON_COLS) -> tuple[int, int]:
    """Piece 1 at (0, 0), row-major on the completed board."""
    idx = piece_num - 1
    return idx // cols, idx % cols


def pose_from_pieces(
    pieces: list[Piece],
    names: list[str] | None = None,
) -> tuple[GroundTruth | None, int, int]:
    """Build pose GT keyed by extraction id. Compact to the occupied block when possible."""
    names = names if names is not None else load_piece_names()
    cells: dict[int, tuple[int, int]] = {}
    for piece in pieces:
        if piece.class_id is None:
            continue
        number = piece_number(piece.class_id, names)
        if number is None:
            continue
        cells[piece.id] = canonical_cell(number)
    if len(cells) < 2:
        return None, 0, 0

    rows = [r for r, _c in cells.values()]
    cols = [c for _r, c in cells.values()]
    r0, c0 = min(rows), min(cols)
    shifted = {pid: (r - r0, c - c0) for pid, (r, c) in cells.items()}
    height = max(r for r, _c in shifted.values()) + 1
    width = max(c for _r, c in shifted.values()) + 1
    occupied = set(shifted.values())
    compact = height * width == len(shifted) and len(occupied) == len(shifted)
    if not compact:
        gt = GroundTruth(pieces={
            pid: {"row": r, "col": c, "rot": 0, "pose_note": "canonical_7x5_uncompacted"}
            for pid, (r, c) in cells.items()
        })
        return gt, CANON_ROWS, CANON_COLS

    gt = GroundTruth(pieces={
        pid: {"row": r, "col": c, "rot": 0, "pose_note": "canonical_compact"}
        for pid, (r, c) in shifted.items()
    })
    return gt, height, width


def canonical_flat_dirs(piece_num: int, rows: int = CANON_ROWS, cols: int = CANON_COLS) -> set[int]:
    """Board directions that must be flat for this piece on the completed 7×5."""
    r, c = canonical_cell(piece_num, rows=rows, cols=cols)
    dirs: set[int] = set()
    if r == 0:
        dirs.add(0)
    if c == cols - 1:
        dirs.add(1)
    if r == rows - 1:
        dirs.add(2)
    if c == 0:
        dirs.add(3)
    return dirs


def identity_rotation(
    piece: Piece,
    names: list[str] | None = None,
) -> int | None:
    """Unique placement rot that maps observed flats onto canonical border dirs.

    Interior pieces (no canonical flats) return None — photo rotation is not
    labelled. After ``flat_frame_rotation``, a top-left corner is rot=0, a
    top-right corner is rot=1, and so on.
    """
    if piece.class_id is None or not piece.sides:
        return None
    names = names if names is not None else load_piece_names()
    number = piece_number(piece.class_id, names)
    if number is None:
        return None
    want = canonical_flat_dirs(number)
    if not want:
        return None
    have = {int(s.index) % 4 for s in piece.sides if s.cls == "flat"}
    matches: list[int] = []
    for rot in range(4):
        predicted = {d for d in range(4) if (d - rot) % 4 in have}
        if predicted == want:
            matches.append(rot)
    return matches[0] if len(matches) == 1 else None


def relative_orient(si: int, sj: int) -> int:
    """Clockwise steps of piece j so local side sj faces local side si.

    Same convention as ``SiameseCompatibilityMatcher`` / ``Placement.rot``.
    This is implied by the matching side indices; it is not photo rotation.
    """
    return (int(si) - int(sj)) % 4


def identity_local_sides(
    piece_a: Piece,
    piece_b: Piece,
    si_canon: int,
    sj_canon: int,
    names: list[str] | None = None,
) -> tuple[int, int] | None:
    """Map canonical neighbour dirs to piece-local sides via identity+flats.

    Returns None unless both pieces have a unique identity rotation and the
    mapped sides are a class-legal tab↔blank pair.
    """
    rot_a = identity_rotation(piece_a, names)
    rot_b = identity_rotation(piece_b, names)
    if rot_a is None or rot_b is None:
        return None
    if not piece_a.sides or not piece_b.sides:
        return None
    si = (int(si_canon) - rot_a) % 4
    sj = (int(sj_canon) - rot_b) % 4
    if si >= len(piece_a.sides) or sj >= len(piece_b.sides):
        return None
    sa, sb = piece_a.sides[si], piece_b.sides[sj]
    if sa.cls == "flat" or sb.cls == "flat" or sa.cls == sb.cls:
        return None
    return si, sj


def canonical_neighbors(piece_num: int) -> dict[int, tuple[int, int]]:
    """Map neighbour piece number → (my_canonical_side, their_canonical_side).

    Sides: 0=N, 1=E, 2=S, 3=W on the completed board.
    """
    r, c = canonical_cell(piece_num)
    out: dict[int, tuple[int, int]] = {}
    if r > 0:
        out[(r - 1) * CANON_COLS + c + 1] = (0, 2)
    if c + 1 < CANON_COLS:
        out[r * CANON_COLS + (c + 1) + 1] = (1, 3)
    if r + 1 < CANON_ROWS:
        out[(r + 1) * CANON_COLS + c + 1] = (2, 0)
    if c > 0:
        out[r * CANON_COLS + (c - 1) + 1] = (3, 1)
    return out


def weak_side_adjacency(
    pieces: list[Piece],
    names: list[str] | None = None,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Neighbour identities from piece numbers; sides without using classical D.

    If both pieces have a unique identity+flats rotation, canonical neighbour
    directions are mapped to local sides (stronger than E_shape). Otherwise a
    unique tab↔blank pair is used; if several are legal, lowest E_shape is a
    weak tie-break. Colour/full D is not used, so Siamese/GNN labels are not a
    copy of the classical matcher. Relative orientation is ``(si − sj) mod 4``.
    """
    names = names if names is not None else load_piece_names()
    number_of: dict[int, int] = {}
    for piece in pieces:
        if piece.class_id is None or not piece.sides:
            continue
        number = piece_number(piece.class_id, names)
        if number is not None:
            number_of[piece.id] = number
    by_number = {n: pid for pid, n in number_of.items()}
    if len(by_number) < 2:
        return {}

    index_of = {p.id: i for i, p in enumerate(pieces)}
    adjacency: dict[tuple[int, int], tuple[int, int]] = {}
    seen: set[tuple[int, int]] = set()
    for pid, number in number_of.items():
        for other_num, (si_canon, sj_canon) in canonical_neighbors(number).items():
            if other_num not in by_number:
                continue
            qid = by_number[other_num]
            key = tuple(sorted((pid, qid)))
            if key in seen:
                continue
            seen.add(key)
            i, j = index_of[pid], index_of[qid]
            mapped = identity_local_sides(
                pieces[i], pieces[j], si_canon, sj_canon, names
            )
            if mapped is not None:
                si, sj = mapped
                adjacency[(pid, si)] = (qid, sj)
                continue
            legal: list[tuple[float, int, int]] = []
            for si, sa in enumerate(pieces[i].sides):
                for sj, sb in enumerate(pieces[j].sides):
                    if sa.cls == "flat" or sb.cls == "flat" or sa.cls == sb.cls:
                        continue
                    legal.append((_shape_dissimilarity(sa.profile, sb.profile), si, sj))
            if not legal:
                continue
            legal.sort()
            _cost, si, sj = legal[0]
            adjacency[(pid, si)] = (qid, sj)
    return adjacency


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = max(0, ax1 - ax0) * max(0, ay1 - ay0) + max(0, bx1 - bx0) * max(0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def attach_class_ids(pieces: list[Piece], boxed: list[Piece], min_iou: float = 0.3) -> None:
    """Copy YOLO class ids onto CCL pieces by maximum bounding-box IoU."""
    used: set[int] = set()
    for piece in pieces:
        best: tuple[float, Piece] | None = None
        for box in boxed:
            if box.id in used or box.class_id is None:
                continue
            iou = bbox_iou(piece.bbox, box.bbox)
            if best is None or iou > best[0]:
                best = (iou, box)
        if best is not None and best[0] >= min_iou:
            piece.class_id = best[1].class_id
            used.add(best[1].id)
