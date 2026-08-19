from src.core.protocols import ImageFilter
from src.enhancement import FilterChain, GaussianFilter, MeanFilter, MedianFilter


def test_mean_filter_is_image_filter() -> None:
    assert issubclass(MeanFilter, ImageFilter)


def test_gaussian_and_median_are_image_filters() -> None:
    assert issubclass(GaussianFilter, ImageFilter)
    assert issubclass(MedianFilter, ImageFilter)


def test_filter_chain_is_image_filter() -> None:
    assert issubclass(FilterChain, ImageFilter)
