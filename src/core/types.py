"""Domain objects shared by classical matching and the Milestone 2 models.

These dataclasses hold data only. Behaviour lives in `src.core.protocols`
implementations so classical, Siamese, and GNN never fork reconstruction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SideClass = Literal["tab", "blank", "flat"]


@dataclass
class Side:
    """One of four piece sides in the piece-local frame (0=top, 1=right, 2=bottom, 3=left)."""

    index: int
    cls: SideClass
    # Signed offsets along the side; positive = exterior tab, negative = blank.
    profile: np.ndarray
    # Inward Lab samples used for photometric matching (N, 3).
    colour: np.ndarray
    # Edge-band tensor for Siamese/GNN; unused until Milestone 2.
    ribbon: np.ndarray
    contour_pts: np.ndarray


@dataclass
class Piece:
    """One extracted jigsaw piece. `sides` is empty until `PieceDescriptor.describe`."""

    id: int
    image: np.ndarray  # bbox crop; Pillow RGB (this project does not use OpenCV BGR)
    mask: np.ndarray
    contour: np.ndarray  # (M, 2) in crop coordinates
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in the full scrambled image
    # Principal-axis angle of the mask (pre-deskew). After describe(), the crop
    # is axis-aligned and flats are snapped: corners N+W, edges N.
    pca_theta: float
    corners: np.ndarray  # (4, 2), clockwise from top-left after PCA-align
    sides: list[Side] = field(default_factory=list)
    is_corner: bool = False
    is_border: bool = False
    class_id: int | None = None  # YOLO class index when boxes are used; else None

    def side_toward(self, rot: int, board_dir: int) -> Side:
        """Side that faces `board_dir` after `rot` 90° clockwise placements.

        Sides are stored in the unrotated piece frame. Clockwise rotation of the
        piece maps board direction D (0=N, 1=E, 2=S, 3=W) to piece side
        ``(D - rot) % 4``.
        """
        if not self.sides:
            raise ValueError(f"piece {self.id} has no described sides")
        index = (board_dir - rot) % 4
        return self.sides[index]


@dataclass
class CompatibilityScore:
    """One candidate pairing of piece i/side si with piece j/side sj."""

    i: int
    si: int
    j: int
    sj: int
    dissim: float  # inf means geometrically illegal; never chosen by the assembler
    e_shape: float
    e_colour: float
    p_neighbour: float | None  # filled by ML matchers; None for classical
    rel_orient: int  # 0..3 clockwise steps of j relative to i


@dataclass
class CompatibilityTensor:
    """Pairwise side dissimilarity, shape (N, 4, N, 4).

    Lower is better. The assembler minimises total edge dissimilarity, so
    Siamese/GNN probabilities must be converted (e.g. -log p) before insertion.
    """

    dissim: np.ndarray

    def pair(self, i: int, si: int, j: int, sj: int) -> float:
        return float(self.dissim[i, si, j, sj])


@dataclass
class Placement:
    """One cell of the reconstructed grid. `rot` is 90° clockwise steps."""

    piece_id: int
    row: int
    col: int
    rot: int


@dataclass
class AssemblyState:
    grid: list[list[Placement | None]]
    used: set[int]
    total_dissim: float = 0.0


@dataclass
class Puzzle:
    image: np.ndarray
    pieces: list[Piece]
    rows: int
    cols: int
    # Unfiltered colour copy; photometric strips must not see equalization/blur.
    raw_colour: np.ndarray | None = None


@dataclass
class EdgeResult:
    """Canny/Sobel intermediates for reports. `extras` holds per-stage maps."""

    magnitude: np.ndarray
    orientation: np.ndarray
    edges: np.ndarray
    extras: dict = field(default_factory=dict)


@dataclass
class GroundTruth:
    """Per-piece grid pose. Keys are extraction IDs after mask-centroid correspondence."""

    pieces: dict[int, dict]


@dataclass
class ReconstructionResult:
    state: AssemblyState
    image: np.ndarray | None
    quality: float | None = None
    metrics: dict = field(default_factory=dict)
