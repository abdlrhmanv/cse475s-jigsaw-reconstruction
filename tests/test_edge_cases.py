"""Edge-case tests: empty images, NaN inputs, single-piece puzzle."""

import numpy as np
import pytest

from src.contour_extraction import MooreContourTracer, PieceExtractorImpl
from src.core.types import CompatibilityTensor, Piece, Side
from src.assembly import GreedyBestFirstAssembler
from src.enhancement import MeanFilter, GaussianFilter, MedianFilter
from src.segmentation import ConnectedComponentLabeler
from src.thresholding import OtsuThreshold, GlobalThreshold


class TestEmptyImages:
    def test_ccl_empty(self):
        binary = np.zeros((20, 20), dtype=np.float64)
        labels = ConnectedComponentLabeler(min_area=5).label(binary)
        assert labels.max() == 0

    def test_contour_trace_empty(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        contour = MooreContourTracer().trace(mask)
        assert len(contour) == 0

    def test_extractor_empty(self):
        img = np.zeros((20, 20, 3), dtype=np.float64)
        labels = np.zeros((20, 20), dtype=np.int32)
        pieces = PieceExtractorImpl(pad=2).extract(img, labels)
        assert len(pieces) == 0

    def test_assembler_empty(self):
        state = GreedyBestFirstAssembler().assemble(
            [], CompatibilityTensor(dissim=np.empty((0, 4, 0, 4))), 1, 1
        )
        assert len(state.used) == 0


class TestNaNInputs:
    def test_mean_filter_nan(self):
        img = np.full((10, 10), 100.0)
        img[5, 5] = np.nan
        result = MeanFilter(k=3).apply(img)
        assert np.isfinite(result).sum() >= 90

    def test_gaussian_filter_nan(self):
        img = np.full((10, 10), 100.0)
        img[5, 5] = np.nan
        result = GaussianFilter(k=3, sigma=1.0).apply(img)
        assert result.shape == (10, 10)

    def test_otsu_constant(self):
        img = np.full((20, 20), 128.0)
        result = OtsuThreshold().threshold(img)
        assert result.shape == (20, 20)


class TestSinglePiecePuzzle:
    def test_single_piece_assembly(self):
        sides = [
            Side(index=i, cls="flat", profile=np.zeros(8),
                 colour=np.zeros((8, 3)), ribbon=np.empty(0),
                 contour_pts=np.zeros((4, 2)))
            for i in range(4)
        ]
        piece = Piece(
            id=0, image=np.zeros((10, 10, 3), dtype=np.uint8),
            mask=np.ones((10, 10), dtype=np.uint8) * 255,
            contour=np.zeros((4, 2), dtype=np.int32), bbox=(0, 0, 10, 10),
            pca_theta=0.0, corners=np.zeros((4, 2)), sides=sides,
        )
        dissim = np.full((1, 4, 1, 4), np.inf)
        state = GreedyBestFirstAssembler().assemble(
            [piece], CompatibilityTensor(dissim=dissim), 1, 1
        )
        assert len(state.used) == 1

    def test_single_pixel_ccl(self):
        binary = np.zeros((10, 10), dtype=np.float64)
        binary[5, 5] = 255
        labels = ConnectedComponentLabeler(min_area=0).label(binary)
        assert labels[5, 5] > 0


class TestAllBlack:
    def test_global_threshold_black(self):
        img = np.zeros((20, 20), dtype=np.float64)
        result = GlobalThreshold(t=128).threshold(img)
        assert result.max() == 0

    def test_otsu_black(self):
        img = np.zeros((20, 20), dtype=np.float64)
        result = OtsuThreshold().threshold(img)
        assert result.shape == (20, 20)
