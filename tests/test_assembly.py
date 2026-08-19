from src.assembly import CanvasReconstructor, GreedyBestFirstAssembler
from src.core.protocols import Assembler, ImageReconstructor


def test_assembler_protocols() -> None:
    assert issubclass(GreedyBestFirstAssembler, Assembler)
    assert issubclass(CanvasReconstructor, ImageReconstructor)
