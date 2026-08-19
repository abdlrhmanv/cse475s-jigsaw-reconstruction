from src.core.protocols import CompatibilityMatcher
from src.edge_matching import ClassicalCompatibilityMatcher


def test_classical_matcher_protocol() -> None:
    assert issubclass(ClassicalCompatibilityMatcher, CompatibilityMatcher)
