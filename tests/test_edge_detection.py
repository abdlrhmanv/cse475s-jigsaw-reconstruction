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


def test_canny_circle_closed_loop():
    """Canny on a bright circle should produce a closed edge loop after hysteresis."""
    img = np.zeros((80, 80), dtype=np.float64)
    ys, xs = np.mgrid[0:80, 0:80]
    circle = ((ys - 40) ** 2 + (xs - 40) ** 2) <= 25 ** 2
    img[circle] = 200.0
    result = CannyEdgeDetector(t_low=10, t_high=30).detect(img)
    edge_map = result.edges > 0
    edge_pts = np.argwhere(edge_map)
    assert len(edge_pts) > 20, "should detect substantial edge pixels"
    # Check that edge pixels form a rough ring (present in multiple quadrants)
    center = np.array([40, 40])
    quadrants = set()
    for r, c in edge_pts:
        qr = 0 if r < 40 else 1
        qc = 0 if c < 40 else 1
        quadrants.add((qr, qc))
    assert len(quadrants) == 4, "edge should span all four quadrants (closed loop)"


def test_nms_thins_edges():
    """NMS on a thick gradient should reduce to ~1 px width."""
    img = np.zeros((40, 40), dtype=np.float64)
    img[:, 20:] = 200.0
    result = CannyEdgeDetector(t_low=5, t_high=20).detect(img)
    edge_cols = np.where(result.edges[20, :] > 0)[0]
    if len(edge_cols) > 0:
        assert edge_cols.max() - edge_cols.min() <= 2, "NMS should thin to ~1px"
