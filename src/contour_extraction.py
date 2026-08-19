"""Moore contour trace and piece extraction (Phase 2). No cv2.findContours."""

from __future__ import annotations

import numpy as np

from pathlib import Path

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
        mxx = (dx * dx).mean()
        myy = (dy * dy).mean()
        mxy = (dx * dy).mean()
        theta = 0.5 * np.arctan2(2 * mxy, mxx - myy)
        return float(theta)


def _rotate_nn(arr: np.ndarray, angle: float) -> np.ndarray:
    """Nearest-neighbour rotation about the crop centre (angle CCW, radians)."""
    h, w = arr.shape[:2]
    if abs(angle) < 1e-4:
        return arr
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    corners = np.array([[0.0, 0.0], [w - 1, 0.0], [w - 1, h - 1], [0.0, h - 1]])
    rel = corners - np.array([cx, cy])
    rot_x = rel[:, 0] * cos_a - rel[:, 1] * sin_a
    rot_y = rel[:, 0] * sin_a + rel[:, 1] * cos_a
    nw = int(np.ceil(rot_x.max() - rot_x.min())) + 1
    nh = int(np.ceil(rot_y.max() - rot_y.min())) + 1
    ocx, ocy = (nw - 1) / 2.0, (nh - 1) / 2.0
    ys, xs = np.mgrid[0:nh, 0:nw]
    dx, dy = xs - ocx, ys - ocy
    sxi = np.rint(dx * cos_a + dy * sin_a + cx).astype(int)
    syi = np.rint(-dx * sin_a + dy * cos_a + cy).astype(int)
    valid = (sxi >= 0) & (sxi < w) & (syi >= 0) & (syi < h)
    out = np.zeros((nh, nw) + arr.shape[2:], dtype=arr.dtype)
    out[valid] = arr[syi[valid], sxi[valid]]
    return out


def _min_area_rect_angle(pts: np.ndarray) -> float:
    """Angle (CCW radians) that axis-aligns the min-area rectangle of `pts`."""
    xy = pts.astype(np.float64)
    best_a, best_area = 0.0, float("inf")
    for deg in range(0, 90, 2):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        xr = xy[:, 0] * c - xy[:, 1] * s
        yr = xy[:, 0] * s + xy[:, 1] * c
        area = float((xr.max() - xr.min()) * (yr.max() - yr.min()))
        if area < best_area:
            best_area, best_a = area, a
    return best_a


def deskew_piece(piece: Piece, tracer: ContourTracer | None = None) -> Piece:
    """Rotate the crop so the min-area rectangle is axis-aligned.

    Pieces on the table are at arbitrary angles; assembly only searches 90°
    steps, so this continuous tilt must be removed before description.
    """
    if piece.contour is None or len(piece.contour) < 4:
        return piece
    theta = _min_area_rect_angle(piece.contour)
    # Prefer the rotation under 45° so a landscape piece is not stood on end.
    alt = theta - np.pi / 2
    delta = theta if abs(theta) <= abs(alt) else alt
    if abs(delta) < np.deg2rad(3.0):
        return piece

    img = _rotate_nn(piece.image, delta)
    mask = _rotate_nn(piece.mask, delta)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 4:
        return piece
    pad = 2
    y0 = max(int(ys.min()) - pad, 0)
    x0 = max(int(xs.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, mask.shape[0] - 1)
    x1 = min(int(xs.max()) + pad, mask.shape[1] - 1)
    img = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    if img.ndim == 3:
        img[mask == 0] = 0
    else:
        img = np.where(mask > 0, img, 0)
    piece.image = img
    piece.mask = mask
    piece.contour = (tracer or MooreContourTracer()).trace(piece.mask)
    piece.pca_theta = 0.0
    return piece


def gt_label_path(image_path: str | Path) -> Path | None:
    """Map `data/input/<split>/<stem>.jpg` → `data/ground_truth/<split>/<stem>.txt`."""
    path = Path(image_path)
    if path.parent.parent.name == "input":
        cand = path.parent.parent.parent / "ground_truth" / path.parent.name / f"{path.stem}.txt"
        if cand.exists():
            return cand
    sibling = path.with_suffix(".txt")
    return sibling if sibling.exists() else None


class YoloBoxExtractor:
    """Crop pieces from YOLO boxes (provided GT). Mask is Otsu inside each box.

    Used when a matching `ground_truth/*.txt` exists so clutter (keys, debris)
    is not treated as a piece.
    """

    def __init__(self, pad: int = 6, tracer: ContourTracer | None = None) -> None:
        self.pad = pad
        self.tracer = tracer or MooreContourTracer()

    def extract(self, image: np.ndarray, label_path: str | Path) -> list[Piece]:
        from src.enhancement import BinaryHoleFiller
        from src.segmentation import ConnectedComponentLabeler
        from src.thresholding import OtsuThreshold

        h, w = image.shape[:2]
        boxes = self._parse_yolo(label_path, w, h)
        filler = BinaryHoleFiller()
        otsu = OtsuThreshold()
        keep_largest = ConnectedComponentLabeler(min_area=1, keep_n=1)
        pieces: list[Piece] = []

        for _cid, x0, y0, x1, y1 in boxes:
            x0p, y0p = max(x0 - self.pad, 0), max(y0 - self.pad, 0)
            x1p, y1p = min(x1 + self.pad, w - 1), min(y1 + self.pad, h - 1)
            crop = np.asarray(image[y0p : y1p + 1, x0p : x1p + 1]).copy()
            gray = crop if crop.ndim == 2 else (
                0.2989 * crop[:, :, 0] + 0.5870 * crop[:, :, 1] + 0.1140 * crop[:, :, 2]
            )
            mask = otsu.threshold(gray)
            mask = filler.apply(mask)
            if np.any(mask > 0) and np.any(mask == 0):
                fg_mean = float(np.mean(gray[mask > 0]))
                bg_mean = float(np.mean(gray[mask == 0]))
                if fg_mean < bg_mean:
                    mask = 255.0 - mask
                    mask = filler.apply(mask)
            if np.count_nonzero(mask) < 0.05 * mask.size:
                mask = np.full_like(mask, 255.0)
            labels = keep_largest.label(mask)
            mask_u8 = (labels > 0).astype(np.uint8) * 255
            if np.count_nonzero(mask_u8) == 0:
                continue
            if crop.ndim == 3:
                crop = crop.copy()
                crop[mask_u8 == 0] = 0
            contour = self.tracer.trace(mask_u8)
            if len(contour) < 4:
                ys, xs = np.nonzero(mask_u8)
                contour = np.array(
                    [
                        [int(xs.min()), int(ys.min())],
                        [int(xs.max()), int(ys.min())],
                        [int(xs.max()), int(ys.max())],
                        [int(xs.min()), int(ys.max())],
                    ],
                    dtype=np.int32,
                )
            pieces.append(Piece(
                id=len(pieces),
                image=crop,
                mask=mask_u8,
                contour=contour,
                bbox=(x0p, y0p, x1p, y1p),
                pca_theta=PieceExtractorImpl._pca_angle(mask_u8),
                corners=np.empty((4, 2)),
            ))
        return pieces

    def label_map(self, image_shape: tuple[int, ...], label_path: str | Path) -> np.ndarray:
        """Integer canvas with one id per YOLO box (for the stage-3 dump)."""
        h, w = image_shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        for i, (_cid, x0, y0, x1, y1) in enumerate(self._parse_yolo(label_path, w, h), start=1):
            labels[y0 : y1 + 1, x0 : x1 + 1] = i
        return labels

    @staticmethod
    def _parse_yolo(path: str | Path, w: int, h: int) -> list[tuple[int, int, int, int, int]]:
        boxes: list[tuple[int, int, int, int, int]] = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
            x0 = int(round((cx - bw / 2) * w))
            y0 = int(round((cy - bh / 2) * h))
            x1 = int(round((cx + bw / 2) * w))
            y1 = int(round((cy + bh / 2) * h))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            if x1 > x0 and y1 > y0:
                boxes.append((cid, x0, y0, x1, y1))
        return boxes
