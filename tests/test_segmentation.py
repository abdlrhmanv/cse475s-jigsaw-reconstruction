from src.core.protocols import Labeler
from src.segmentation import ConnectedComponentLabeler


def test_ccl_is_labeler() -> None:
    assert issubclass(ConnectedComponentLabeler, Labeler)
