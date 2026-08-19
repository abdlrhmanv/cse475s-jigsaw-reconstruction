"""Moore contour trace and piece extraction (Phase 2). No cv2.findContours."""

from __future__ import annotations

import numpy as np

from src.core.protocols import ContourTracer, PieceExtractor
from src.core.types import Piece


# 8-connected Moore neighbourhood, clockwise from east
_MOORE_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_MOORE_DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)


class MooreContourTracer(ContourTracer):
    """Moore-neighbour tracing on a single-object binary mask.

    Returns an (M, 2) array of (x, y) boundary coordinates, ordered clockwise.
    The mask must contain exactly one connected blob (pass individual label masks).
    """

    def trace(self, blob_mask: np.ndarray) -> np.ndarray:
        mask = blob_mask > 0
        h, w = mask.shape

        # Find the topmost-leftmost foreground pixel (start point)
        start = None
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    start = (c, r)  # (x, y)
                    break
            if start is not None:
                break

        if start is None:
            return np.empty((0, 2), dtype=np.int32)

        contour: list[tuple[int, int]] = []
        sx, sy = start
        # The backtrack direction: we entered from the left (west), so
        # the previous background pixel was at (sx-1, sy) → direction index 4 (west).
        backtrack_dir = 4
        cx, cy = sx, sy
        contour.append((cx, cy))

        while True:
            # Start scanning from (backtrack_dir + 1) % 8
            scan_start = (backtrack_dir + 1) % 8
            found = False
            for i in range(8):
                d = (scan_start + i) % 8
                nx, ny = cx + _MOORE_DX[d], cy + _MOORE_DY[d]
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                    # Record the direction we came from (opposite)
                    backtrack_dir = (d + 4) % 8
                    cx, cy = nx, ny
                    found = True
                    break

            if not found:
                break  # isolated single pixel

            if cx == sx and cy == sy:
                break  # closed the loop

            contour.append((cx, cy))

        return np.array(contour, dtype=np.int32)


class PieceExtractorImpl(PieceExtractor):
    """Extract individual `Piece` objects from a labelled image.

    For each label: compute bounding box, crop image+mask, trace contour,
    and estimate PCA orientation angle.
    """

    def __init__(self, pad: int = 5, tracer: ContourTracer | None = None) -> None:
        self.pad = pad
        self.tracer = tracer or MooreContourTracer()

    def extract(self, image: np.ndarray, labels: np.ndarray) -> list[Piece]:
        pieces: list[Piece] = []
        unique = np.unique(labels)

        for lbl in unique:
            if lbl == 0:
                continue

            blob = labels == lbl
            ys, xs = np.nonzero(blob)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            # Pad bbox, clamp to image bounds
            h, w = labels.shape[:2]
            x0p = max(x0 - self.pad, 0)
            y0p = max(y0 - self.pad, 0)
            x1p = min(x1 + self.pad, w - 1)
            y1p = min(y1 + self.pad, h - 1)

            crop_mask = blob[y0p : y1p + 1, x0p : x1p + 1].astype(np.uint8) * 255
            crop_img = image[y0p : y1p + 1, x0p : x1p + 1].copy()
            if crop_img.ndim == 3:
                crop_img[crop_mask == 0] = 0

            contour = self.tracer.trace(crop_mask)
            pca_theta = self._pca_angle(crop_mask)

            pieces.append(Piece(
                id=len(pieces),
                image=crop_img,
                mask=crop_mask,
                contour=contour,
                bbox=(x0p, y0p, x1p, y1p),
                pca_theta=pca_theta,
                corners=np.empty((4, 2)),
            ))

        return pieces

    @staticmethod
    def _pca_angle(mask: np.ndarray) -> float:
        """Principal axis angle from the binary mask's second-order central moments."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            return 0.0
        cx = xs.mean()
        cy = ys.mean()
        dx = xs - cx
        dy = ys - cy
        # Covariance matrix entries
        mxx = (dx * dx).mean()
        myy = (dy * dy).mean()
        mxy = (dx * dy).mean()
        theta = 0.5 * np.arctan2(2 * mxy, mxx - myy)
        return float(theta)
