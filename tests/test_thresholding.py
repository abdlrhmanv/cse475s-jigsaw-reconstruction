from src.core.protocols import Thresholder
from src.thresholding import AdaptiveThreshold, GlobalThreshold, OtsuThreshold


def test_thresholders_implement_protocol() -> None:
    assert issubclass(GlobalThreshold, Thresholder)
    assert issubclass(OtsuThreshold, Thresholder)
    assert issubclass(AdaptiveThreshold, Thresholder)
