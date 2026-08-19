import numpy as np

from src.core.protocols import EdgeDetector, GradientOperator
from src.edge_detection import CannyEdgeDetector, PrewittOperator, SobelOperator


def test_gradient_and_canny_protocols() -> None:
    assert issubclass(SobelOperator, GradientOperator)
    assert issubclass(PrewittOperator, GradientOperator)
    assert issubclass(CannyEdgeDetector, EdgeDetector)


def test_sobel_horizontal_edge():
    """A sharp horizontal step should produce strong Gy."""
    img = np.zeros((20, 20), dtype=np.float64)
    img[10:, :] = 200.0
    gx, gy = SobelOperator().gradients(img)
    # Gy should peak at row 10 (the transition)
    assert np.abs(gy[10, 10]) > np.abs(gy[5, 10])


def test_prewitt_horizontal_edge():
    img = np.zeros((20, 20), dtype=np.float64)
    img[10:, :] = 200.0
    gx, gy = PrewittOperator().gradients(img)
    assert np.abs(gy[10, 10]) > np.abs(gy[5, 10])


def test_sobel_constant_image_zero_gradient():
    img = np.full((15, 15), 100.0)
    gx, gy = SobelOperator().gradients(img)
    np.testing.assert_allclose(gx, 0.0, atol=1e-10)
    np.testing.assert_allclose(gy, 0.0, atol=1e-10)


def test_canny_detects_rectangle():
    """Canny should find edges around a bright rectangle on dark background."""
    img = np.zeros((40, 40), dtype=np.float64)
    img[10:30, 10:30] = 200.0
    result = CannyEdgeDetector(t_low=20, t_high=50).detect(img)
    assert result.edges.shape == (40, 40)
    # Edges should be non-zero somewhere near the rectangle border
    border_region = result.edges[9:31, 9:31]
    assert border_region.max() > 0


def test_canny_result_fields():
    img = np.random.rand(20, 20) * 255
    result = CannyEdgeDetector().detect(img)
    assert result.magnitude.shape == (20, 20)
    assert result.orientation.shape == (20, 20)
    assert result.edges.shape == (20, 20)
    assert "nms" in result.extras
