import numpy as np

from src.assembly import CanvasReconstructor, GreedyBestFirstAssembler
from src.core.protocols import Assembler, ImageReconstructor
from src.core.types import (
    AssemblyState,
    CompatibilityTensor,
    GroundTruth,
    Piece,
    Placement,
    Puzzle,
    Side,
)
from src.evaluation import ReconstructionEvaluator


def test_assembler_protocols() -> None:
    assert issubclass(GreedyBestFirstAssembler, Assembler)
    assert issubclass(CanvasReconstructor, ImageReconstructor)


def _make_piece(pid: int) -> Piece:
    sides = [
        Side(index=i, cls="tab" if i % 2 == 0 else "blank",
             profile=np.ones(20) * (1 if i % 2 == 0 else -1),
             colour=np.zeros((32, 3)), ribbon=np.empty(0),
             contour_pts=np.zeros((10, 2)))
        for i in range(4)
    ]
    return Piece(
        id=pid, image=np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8),
        mask=np.ones((20, 20), dtype=np.uint8) * 255,
        contour=np.zeros((10, 2), dtype=np.int32), bbox=(0, 0, 20, 20),
        pca_theta=0.0, corners=np.zeros((4, 2)), sides=sides,
    )


def test_assembler_produces_grid():
    pieces = [_make_piece(i) for i in range(4)]
    tensor = CompatibilityTensor(dissim=np.ones((4, 4, 4, 4)))
    state = GreedyBestFirstAssembler(beam_k=1).assemble(pieces, tensor, 2, 2)
    assert len(state.grid) == 2
    assert len(state.grid[0]) == 2
    placed = sum(1 for row in state.grid for cell in row if cell is not None)
    assert placed == 4


def test_assembler_empty():
    state = GreedyBestFirstAssembler().assemble([], CompatibilityTensor(dissim=np.empty((0, 4, 0, 4))), 2, 2)
    assert len(state.used) == 0


def test_canvas_rotates_clockwise():
    """rot=1 is 90° clockwise, so a red top row becomes the right column."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[0, :, 0] = 255
    mask = np.ones((10, 20), dtype=np.uint8) * 255
    piece = Piece(
        id=0, image=img, mask=mask,
        contour=np.zeros((4, 2), dtype=np.int32), bbox=(0, 0, 20, 10),
        pca_theta=0.0, corners=np.zeros((4, 2)),
    )
    puzzle = Puzzle(image=np.zeros((10, 20, 3)), pieces=[piece], rows=1, cols=1)
    state = AssemblyState(grid=[[Placement(0, 0, 0, 1)]], used={0})
    canvas = CanvasReconstructor().reconstruct(puzzle, state)
    xs = np.where(canvas[:, :, 0] > 128)[1]
    assert len(xs) > 0
    assert float(xs.mean()) > canvas.shape[1] * 0.5


def test_assembler_grows_confident_neighbour():
    """After the seed, the next cell should be the zero-cost adjacent piece."""
    def piece_with(pid: int, classes: list[str]) -> Piece:
        sides = [
            Side(index=i, cls=classes[i],  # type: ignore[arg-type]
                 profile=np.ones(8), colour=np.zeros((8, 3)),
                 ribbon=np.empty(0), contour_pts=np.zeros((4, 2)))
            for i in range(4)
        ]
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        return Piece(
            id=pid, image=img, mask=np.ones((8, 8), dtype=np.uint8) * 255,
            contour=np.zeros((4, 2), dtype=np.int32), bbox=(0, 0, 8, 8),
            pca_theta=0.0, corners=np.zeros((4, 2)), sides=sides,
            is_corner=classes.count("flat") >= 2,
        )

    # 0=TL (N+W flat), 1=TR (N+E flat), 2=BL (S+W flat), 3=BR (S+E flat)
    pieces = [
        piece_with(0, ["flat", "tab", "tab", "flat"]),
        piece_with(1, ["flat", "flat", "tab", "blank"]),
        piece_with(2, ["blank", "tab", "flat", "flat"]),
        piece_with(3, ["blank", "flat", "flat", "blank"]),
    ]
    dissim = np.full((4, 4, 4, 4), 1.0e6)
    # piece 0 east (side 1) matches piece 1 west (side 3)
    dissim[0, 1, 1, 3] = 0.0
    dissim[1, 3, 0, 1] = 0.0
    tensor = CompatibilityTensor(dissim=dissim)
    state = GreedyBestFirstAssembler(beam_k=1).assemble(pieces, tensor, 2, 2)
    assert state.grid[0][0] is not None
    assert state.grid[0][1] is not None
    assert state.grid[0][0].piece_id == 0
    assert state.grid[0][1].piece_id == 1
    placed = sum(1 for row in state.grid for cell in row if cell is not None)
    assert placed == 4


def test_canvas_reconstructor_shape():
    pieces = [_make_piece(i) for i in range(4)]
    puzzle = Puzzle(image=np.zeros((40, 40, 3)), pieces=pieces, rows=2, cols=2)
    grid = [[Placement(0, 0, 0, 0), Placement(1, 0, 1, 0)],
            [Placement(2, 1, 0, 0), Placement(3, 1, 1, 0)]]
    state = AssemblyState(grid=grid, used={0, 1, 2, 3})
    canvas = CanvasReconstructor().reconstruct(puzzle, state)
    assert canvas.ndim == 3
    assert canvas.shape[0] > 0 and canvas.shape[1] > 0


def test_evaluator_perfect_score():
    grid = [[Placement(0, 0, 0, 0), Placement(1, 0, 1, 0)],
            [Placement(2, 1, 0, 0), Placement(3, 1, 1, 0)]]
    state = AssemblyState(grid=grid, used={0, 1, 2, 3})
    gt = GroundTruth(pieces={
        0: {"row": 0, "col": 0, "rot": 0},
        1: {"row": 0, "col": 1, "rot": 0},
        2: {"row": 1, "col": 0, "rot": 0},
        3: {"row": 1, "col": 1, "rot": 0},
    })
    metrics = ReconstructionEvaluator().evaluate(state, gt)
    assert metrics["position_accuracy"] == 1.0
    assert metrics["orientation_accuracy"] == 1.0
    assert metrics["edge_accuracy"] == 1.0
    assert metrics["Q"] == 1.0
