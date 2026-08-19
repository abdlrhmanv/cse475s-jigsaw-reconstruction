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
