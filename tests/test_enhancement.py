import numpy as np
import pytest

from src.core.convolution import ConvolutionEngine
from src.core.protocols import ImageFilter
from src.enhancement import (
    BinaryHoleFiller,
    ContrastStretcher,
    FilterChain,
    GaussianFilter,
    HistogramComputer,
    HistogramEqualizer,
    MeanFilter,
    MedianFilter,
    UnsharpMask,
)


# -- Protocol adherence ------------------------------------------------------

def test_mean_filter_is_image_filter() -> None:
    assert issubclass(MeanFilter, ImageFilter)


def test_gaussian_and_median_are_image_filters() -> None:
    assert issubclass(GaussianFilter, ImageFilter)
    assert issubclass(MedianFilter, ImageFilter)


def test_filter_chain_is_image_filter() -> None:
    assert issubclass(FilterChain, ImageFilter)


# -- ConvolutionEngine -------------------------------------------------------

def test_convolve_identity():
    """Convolution with delta kernel should return the image unchanged."""
    engine = ConvolutionEngine()
    img = np.random.rand(10, 10) * 255
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    result = engine.convolve(img, kernel)
    np.testing.assert_allclose(result, img, atol=1e-10)


def test_convolve_colour():
    engine = ConvolutionEngine()
    img = np.random.rand(8, 8, 3) * 255
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    result = engine.convolve(img, kernel)
    assert result.shape == img.shape
    np.testing.assert_allclose(result, img, atol=1e-10)


def test_convolve_even_kernel_raises():
    with pytest.raises(ValueError):
        ConvolutionEngine().convolve(np.zeros((5, 5)), np.zeros((2, 2)))


# -- MeanFilter --------------------------------------------------------------

def test_mean_filter_constant_image():
    img = np.full((10, 10), 128.0)
    result = MeanFilter(3).apply(img)
    np.testing.assert_allclose(result, 128.0, atol=1e-8)


def test_mean_filter_shape_preserved():
    img = np.random.rand(20, 30) * 255
    assert MeanFilter(5).apply(img).shape == (20, 30)


# -- GaussianFilter ----------------------------------------------------------

def test_gaussian_kernel_sums_to_one():
    g = GaussianFilter(5, sigma=1.0)
    np.testing.assert_allclose(g.kernel().sum(), 1.0, atol=1e-12)


def test_gaussian_constant_image():
    img = np.full((10, 10), 200.0)
    result = GaussianFilter(3, 1.0).apply(img)
    np.testing.assert_allclose(result, 200.0, atol=1e-6)


# -- MedianFilter ------------------------------------------------------------

def test_median_removes_salt_pepper():
    np.random.seed(42)
    img = np.full((20, 20), 128.0)
    noise_idx = np.random.choice(400, 20, replace=False)
    img.ravel()[noise_idx] = np.random.choice([0.0, 255.0], 20)
    result = MedianFilter(3).apply(img)
    # The median should push most outliers back toward 128
    assert np.abs(result - 128.0).mean() < 20.0


def test_median_constant_image():
    img = np.full((8, 8), 50.0)
    np.testing.assert_allclose(MedianFilter(3).apply(img), 50.0)


# -- Histogram ---------------------------------------------------------------

def test_histogram_256_bins():
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    hist = HistogramComputer().compute(img)
    assert hist.shape == (256,)
    assert hist.sum() == 100


def test_histogram_equalizer_range():
    img = np.random.randint(50, 200, (30, 30), dtype=np.uint8).astype(np.float64)
    result = HistogramEqualizer().apply(img)
    assert result.min() >= 0
    assert result.max() <= 255


# -- ContrastStretcher -------------------------------------------------------

def test_contrast_stretcher_full_range():
    img = np.random.rand(20, 20) * 100 + 50  # [50, 150]
    result = ContrastStretcher(0.0, 100.0).apply(img)
    assert result.min() >= 0
    assert result.max() <= 255


# -- UnsharpMask -------------------------------------------------------------

def test_unsharp_mask_enhances():
    img = np.full((20, 20), 128.0)
    img[8:12, 8:12] = 200.0
    sharp = UnsharpMask(3, 1.0, 1.5).apply(img)
    # Centre should become brighter (or at least stay same)
    assert sharp[10, 10] >= img[10, 10]


# -- FilterChain -------------------------------------------------------------

def test_empty_chain_is_identity():
    img = np.random.rand(5, 5) * 255
    np.testing.assert_array_equal(FilterChain([]).apply(img), img)


def test_hole_filler_closes_interior():
    img = np.zeros((20, 20), dtype=np.float64)
    img[4:16, 4:16] = 255
    img[8:12, 8:12] = 0
    filled = BinaryHoleFiller().apply(img)
    assert filled[10, 10] == 255
    assert filled[0, 0] == 0
    assert filled[5, 5] == 255


def test_binary_closer_seals_gap():
    from src.enhancement import BinaryCloser
    img = np.zeros((21, 21), dtype=np.float64)
    img[8:13, 2:9] = 255
    img[8:13, 12:19] = 255
    closed = BinaryCloser(k=5).apply(img)
    assert closed[10, 10] == 255
