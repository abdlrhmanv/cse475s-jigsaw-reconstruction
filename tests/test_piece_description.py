from src.core.protocols import CornerFinder, PieceDescriptor
from src.piece_description import HybridCornerFinder, PieceDescriptorImpl


def test_description_protocols() -> None:
    assert issubclass(HybridCornerFinder, CornerFinder)
    assert issubclass(PieceDescriptorImpl, PieceDescriptor)
