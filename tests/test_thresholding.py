import numpy as np

from src.core.protocols import Thresholder
from src.thresholding import AdaptiveThreshold, GlobalThreshold, OtsuThreshold


def test_thresholders_implement_protocol() -> None:
    assert issubclass(GlobalThreshold, Thresholder)
    assert issubclass(OtsuThreshold, Thresholder)
    assert issubclass(AdaptiveThreshold, Thresholder)


def test_global_threshold_binary():
    img = np.array([[100, 200], [50, 150]], dtype=np.float64)
    result = GlobalThreshold(120).threshold(img)
    expected = np.array([[0, 255], [0, 255]], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_otsu_bimodal():
    """Otsu should separate a clearly bimodal image."""
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:10, :] = 30
    img[10:, :] = 220
    result = OtsuThreshold().threshold(img.astype(np.float64))
    assert result[:10, :].max() == 0
    assert result[10:, :].min() == 255


def test_adaptive_threshold_shape():
    img = np.random.rand(30, 30) * 255
    result = AdaptiveThreshold(11, 5.0).threshold(img)
    assert result.shape == (30, 30)
    assert set(np.unique(result)).issubset({0.0, 255.0})
