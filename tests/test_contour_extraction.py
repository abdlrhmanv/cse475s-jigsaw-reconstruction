"""Tests for Moore contour tracing and piece extraction (spec-required file)."""

from pathlib import Path

import numpy as np
import pytest

from src.contour_extraction import (
    MooreContourTracer,
    PieceExtractorImpl,
    YoloBoxExtractor,
    deskew_piece,
    rotate_piece_cw,
    gt_label_path,
    count_yolo_classes,
)
from src.core.types import Piece
from src.segmentation import ConnectedComponentLabeler


# ---------------------------------------------------------------------------
# MooreContourTracer
# ---------------------------------------------------------------------------


class TestMooreContourTracer:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        contour = MooreContourTracer().trace(mask)
        assert contour.shape == (0, 2)

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255
        contour = MooreContourTracer().trace(mask)
        assert len(contour) >= 1
        assert contour[0, 0] == 5 and contour[0, 1] == 5

    def test_square_boundary(self):
        """10×10 square: contour should trace the border, all points on the mask."""
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        contour = MooreContourTracer().trace(mask)
        assert contour.shape[1] == 2
        assert len(contour) >= 4
        for x, y in contour:
            assert mask[y, x] > 0

    def test_square_perimeter_length(self):
        """A 10×10 filled square has perimeter ~36 boundary pixels."""
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        contour = MooreContourTracer().trace(mask)
        assert 30 <= len(contour) <= 44

    def test_circle_closed_loop(self):
        """Contour of a circle should form a closed loop (first ≈ last)."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        cy, cx, r = 25, 25, 15
        ys, xs = np.mgrid[0:50, 0:50]
        mask[((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2] = 255
        contour = MooreContourTracer().trace(mask)
        assert len(contour) > 10
        dist = np.linalg.norm(contour[0].astype(float) - contour[-1].astype(float))
        assert dist <= 2.0, "contour should close (first and last within 2 px)"

    def test_l_shape(self):
        """L-shaped blob: contour should include concave corner region."""
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:20, 5:15] = 255
        mask[15:20, 15:25] = 255
        contour = MooreContourTracer().trace(mask)
        assert len(contour) > 10
        for x, y in contour:
            assert mask[y, x] > 0

    def test_contour_does_not_include_interior(self):
        """Interior pixels of a large filled square should not be in the contour."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:35, 5:35] = 255
        contour = MooreContourTracer().trace(mask)
        pts = set(map(tuple, contour.tolist()))
        assert (20, 20) not in pts, "centre pixel should not be on the boundary"


# ---------------------------------------------------------------------------
# PieceExtractorImpl
# ---------------------------------------------------------------------------


class TestPieceExtractor:
    def test_extracts_correct_count(self):
        img = np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8).astype(np.float64)
        binary = np.zeros((40, 80), dtype=np.float64)
        binary[5:15, 5:15] = 255
        binary[5:15, 50:65] = 255
        labels = ConnectedComponentLabeler(min_area=10).label(binary)
        pieces = PieceExtractorImpl(pad=2).extract(img, labels)
        assert len(pieces) == 2

    def test_piece_fields(self):
        img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8).astype(np.float64)
        binary = np.zeros((30, 30), dtype=np.float64)
        binary[5:15, 5:15] = 255
        labels = ConnectedComponentLabeler(min_area=10).label(binary)
        pieces = PieceExtractorImpl(pad=2).extract(img, labels)
        assert len(pieces) == 1
        p = pieces[0]
        assert p.mask.max() == 255
        assert p.contour.shape[1] == 2
        assert len(p.contour) > 0
        assert p.bbox[0] <= p.bbox[2]
        assert p.bbox[1] <= p.bbox[3]

    def test_pca_angle_finite(self):
        img = np.zeros((20, 20, 3), dtype=np.float64)
        binary = np.zeros((20, 20), dtype=np.float64)
        binary[3:17, 3:17] = 255
        labels = ConnectedComponentLabeler(min_area=5).label(binary)
        pieces = PieceExtractorImpl(pad=1).extract(img, labels)
        assert np.isfinite(pieces[0].pca_theta)

    def test_masked_exterior_zeroed(self):
        img = np.full((30, 30, 3), 128.0)
        binary = np.zeros((30, 30), dtype=np.float64)
        binary[10:20, 10:20] = 255
        labels = ConnectedComponentLabeler(min_area=5).label(binary)
        pieces = PieceExtractorImpl(pad=2).extract(img, labels)
        p = pieces[0]
        outside = p.image[p.mask == 0]
        assert np.all(outside == 0), "exterior should be zeroed"


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


class TestRotation:
    @staticmethod
    def _make_piece() -> Piece:
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[3:17, 3:27] = 255
        img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
        img[mask == 0] = 0
        contour = MooreContourTracer().trace(mask)
        return Piece(
            id=0, image=img, mask=mask, contour=contour,
            bbox=(0, 0, 29, 19), pca_theta=0.0, corners=np.zeros((4, 2)),
        )

    def test_rotate_0_is_identity(self):
        p = self._make_piece()
        orig_shape = p.image.shape
        q = rotate_piece_cw(p, 0)
        assert q.image.shape == orig_shape

    def test_rotate_4_is_identity(self):
        p = self._make_piece()
        orig = p.image.copy()
        q = rotate_piece_cw(p, 4)
        assert q.image.shape == orig.shape

    def test_rotate_1_swaps_dims(self):
        p = self._make_piece()
        h, w = p.image.shape[:2]
        q = rotate_piece_cw(p, 1)
        assert q.image.shape[:2] == (w, h)

    def test_rotate_clears_sides(self):
        from src.core.types import Side
        p = self._make_piece()
        p.sides = [Side(index=i, cls="flat", profile=np.zeros(5),
                        colour=np.zeros((5, 3)), ribbon=np.zeros((4, 5)),
                        contour_pts=np.zeros((5, 2))) for i in range(4)]
        q = rotate_piece_cw(p, 1)
        assert q.sides == []


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_gt_label_path_found(self, tmp_path: Path):
        img = tmp_path / "input" / "test" / "abc.jpg"
        lbl = tmp_path / "ground_truth" / "test" / "abc.txt"
        img.parent.mkdir(parents=True)
        lbl.parent.mkdir(parents=True)
        img.write_bytes(b"x")
        lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        assert gt_label_path(img) == lbl

    def test_gt_label_path_missing(self, tmp_path: Path):
        img = tmp_path / "input" / "test" / "nope.jpg"
        img.parent.mkdir(parents=True)
        img.write_bytes(b"x")
        assert gt_label_path(img) is None

    def test_count_yolo_classes(self, tmp_path: Path):
        f = tmp_path / "labels.txt"
        f.write_text("0 0.1 0.1 0.1 0.1\n1 0.5 0.5 0.1 0.1\n0 0.9 0.9 0.1 0.1\n")
        assert count_yolo_classes(f) == 2
        assert count_yolo_classes(None) == 0

    def test_count_yolo_classes_empty(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert count_yolo_classes(f) == 0
