from src.core.protocols import EdgeDetector, GradientOperator
from src.edge_detection import CannyEdgeDetector, PrewittOperator, SobelOperator


def test_gradient_and_canny_protocols() -> None:
    assert issubclass(SobelOperator, GradientOperator)
    assert issubclass(PrewittOperator, GradientOperator)
    assert issubclass(CannyEdgeDetector, EdgeDetector)
