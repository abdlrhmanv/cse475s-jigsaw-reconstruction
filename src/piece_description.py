"""Four corners, tab/blank/flat class, colour strips (Phase 3)."""

from __future__ import annotations

import numpy as np

from src.contour_extraction import MooreContourTracer, deskew_piece
from src.core.protocols import CornerFinder, PieceDescriptor
from src.core.types import Piece, Side, SideClass
from src.core.ribbons import pack_ribbon


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Gift-wrapping (Jarvis march) returning indices of hull vertices CCW."""
    n = len(pts)
    if n < 3:
        return np.arange(n)
    start = int(np.argmin(pts[:, 0] * 1_000_000 + pts[:, 1]))  # leftmost-bottom
    hull_idx: list[int] = []
    current = start
    while True:
        hull_idx.append(current)
        candidate = 0
        for i in range(n):
            if i == current:
                continue
            v_a = pts[candidate] - pts[current]
            v_b = pts[i] - pts[current]
            cross = float(v_a[0] * v_b[1] - v_a[1] * v_b[0])
            if candidate == current or cross > 0 or (
                cross == 0 and np.linalg.norm(pts[i] - pts[current])
                > np.linalg.norm(pts[candidate] - pts[current])
            ):
                candidate = i
        current = candidate
        if current == start:
            break
    return np.array(hull_idx, dtype=np.int32)


def _rdp(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification, returns simplified points."""
    if len(pts) <= 2:
        return pts
    d_max = 0.0
    idx = 0
    start, end = pts[0].astype(np.float64), pts[-1].astype(np.float64)
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    for i in range(1, len(pts) - 1):
        if line_len < 1e-12:
            d = float(np.linalg.norm(pts[i] - start))
        else:
            diff = start - pts[i].astype(np.float64)
            d = float(abs(line_vec[0] * diff[1] - line_vec[1] * diff[0]) / line_len)
        if d > d_max:
            d_max = d
            idx = i
    if d_max > epsilon:
        left = _rdp(pts[: idx + 1], epsilon)
        right = _rdp(pts[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.array([pts[0], pts[-1]])


def _interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by segments a→b and b→c, in degrees."""
    v1 = a.astype(np.float64) - b.astype(np.float64)
    v2 = c.astype(np.float64) - b.astype(np.float64)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def _order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """Sort 4 corners clockwise from top-left."""
    cx, cy = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
    order = np.argsort(angles)
    ordered = corners[order]
    sums = ordered[:, 0] + ordered[:, 1]
    shift = int(np.argmin(sums))
    return np.roll(ordered, -shift, axis=0)


def _pca_rect_corners(pts: np.ndarray) -> np.ndarray:
    """Four corners of the min-area PCA rectangle of `pts`, clockwise from TL."""
    xy = pts.astype(np.float64)
    mean = xy.mean(axis=0)
    centered = xy - mean
    if len(xy) < 3:
        return _order_corners_clockwise(xy[:4] if len(xy) else np.zeros((4, 2)))
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or np.linalg.det(cov) < 1e-12:
        mins, maxs = xy.min(axis=0), xy.max(axis=0)
        raw = np.array([
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ])
        return _order_corners_clockwise(raw)
    _, eigvecs = np.linalg.eigh(cov)
    proj = centered @ eigvecs
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    corners_pca = np.array([
        [lo[0], lo[1]],
        [hi[0], lo[1]],
        [hi[0], hi[1]],
        [lo[0], hi[1]],
    ])
    return _order_corners_clockwise(corners_pca @ eigvecs.T + mean)


def _snap_to_contour(corners: np.ndarray, contour: np.ndarray) -> np.ndarray:
    snapped = np.zeros_like(corners)
    used: set[int] = set()
    for i, c in enumerate(corners):
        dists = np.linalg.norm(contour.astype(np.float64) - c.astype(np.float64), axis=1)
        for idx in np.argsort(dists):
            j = int(idx)
            if j not in used:
                snapped[i] = contour[j]
                used.add(j)
                break
    return snapped


# ---------------------------------------------------------------------------
# Corner finders
# ---------------------------------------------------------------------------

class HybridCornerFinder(CornerFinder):
    """PCA rectangle snapped onto the contour.

    Tab tips sit on the convex hull, so hull+90° picking often returns tabs.
    The PCA rectangle tracks the piece body; nearest contour points are the
    true corners.
    """

    def __init__(self, epsilon_frac: float = 0.02) -> None:
        self.epsilon_frac = epsilon_frac

    def find(self, contour: np.ndarray) -> np.ndarray:
        if len(contour) < 4:
            return contour[:4] if len(contour) > 0 else np.zeros((4, 2), dtype=np.int32)
        rect = _pca_rect_corners(contour)
        snapped = _snap_to_contour(rect, contour)
        return _order_corners_clockwise(snapped)


class CurvatureCornerFinder(CornerFinder):
    """Curvature-based fallback: compute discrete curvature along the contour,
    pick the 4 highest-curvature points that are well-separated.
    """

    def __init__(self, smooth_k: int = 7, min_sep_frac: float = 0.1) -> None:
        self.smooth_k = smooth_k
        self.min_sep_frac = min_sep_frac

    def find(self, contour: np.ndarray) -> np.ndarray:
        n = len(contour)
        if n < 4:
            return contour[:4] if n > 0 else np.zeros((4, 2), dtype=np.int32)

        pts = contour.astype(np.float64)
        half = self.smooth_k // 2
        curvature = np.zeros(n)
        for i in range(n):
            a = pts[(i - half) % n]
            b = pts[i]
            c = pts[(i + half) % n]
            v1, v2 = a - b, c - b
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            dot = float(np.dot(v1, v2))
            curvature[i] = abs(np.arctan2(cross, dot))

        min_sep = int(n * self.min_sep_frac)
        order = np.argsort(-curvature)
        selected: list[int] = []
        for idx in order:
            if all(min((idx - s) % n, (s - idx) % n) >= min_sep for s in selected):
                selected.append(idx)
            if len(selected) == 4:
                break

        # Pad if we couldn't find 4
        while len(selected) < 4:
            selected.append(selected[-1] if selected else 0)

        corners = contour[selected]
        return _order_corners_clockwise(corners)


# ---------------------------------------------------------------------------
# Piece descriptor
# ---------------------------------------------------------------------------

def _split_contour_by_corners(
    contour: np.ndarray,
    corners: np.ndarray,
) -> list[np.ndarray]:
    """Split contour into 4 side segments following corner order (TL→TR→BR→BL).

    Walks forward along the clockwise contour; does not re-sort indices, so
    side 0 stays the top side.
    """
    n = len(contour)
    idxs: list[int] = []
    for c in corners:
        dists = np.linalg.norm(contour.astype(np.float64) - c.astype(np.float64), axis=1)
        idxs.append(int(np.argmin(dists)))

    sides: list[np.ndarray] = []
    for i in range(4):
        start = idxs[i]
        end = idxs[(i + 1) % 4]
        if start == end:
            sides.append(contour[start : start + 1])
            continue
        pts: list[np.ndarray] = []
        j = start
        guard = 0
        while guard <= n:
            pts.append(contour[j])
            if j == end and guard > 0:
                break
            j = (j + 1) % n
            guard += 1
        sides.append(np.asarray(pts))
    return sides


def _classify_side(
    profile: np.ndarray,
    side_pts: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
) -> SideClass:
    """Classify a side as tab, blank, or flat.

    Flat = peak deviation small vs side length.
    Tab vs blank uses the piece centroid when available (curve farther from
    centroid than the chord → tab), otherwise the sign of the profile mean.
    """
    if len(profile) == 0:
        return "flat"
    peak = float(np.max(np.abs(profile)))
    length = 1.0
    if side_pts is not None and len(side_pts) >= 2:
        length = float(np.linalg.norm(side_pts[-1].astype(np.float64) - side_pts[0].astype(np.float64)))
    if peak < max(4.0, 0.14 * length):
        return "flat"
    if side_pts is not None and centroid is not None and len(side_pts) >= 2:
        chord = 0.5 * (side_pts[0].astype(np.float64) + side_pts[-1].astype(np.float64))
        curve = side_pts.astype(np.float64).mean(axis=0)
        if np.linalg.norm(curve - centroid) >= np.linalg.norm(chord - centroid):
            return "tab"
        return "blank"
    return "tab" if float(np.mean(profile)) > 0 else "blank"


def _signed_profile(side_pts: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance of each point from the line through the endpoints."""
    if len(side_pts) < 2:
        return np.zeros(0)
    a = side_pts[0].astype(np.float64)
    b = side_pts[-1].astype(np.float64)
    ab = b - a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-12:
        return np.linalg.norm(side_pts.astype(np.float64) - a, axis=1)
    # Signed distance: cross(ab, ap) / |ab|
    ap = side_pts.astype(np.float64) - a
    raw = (ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / ab_len
    if len(raw) >= 5:
        kernel = np.ones(5) / 5.0
        raw = np.convolve(np.pad(raw, 2, mode="edge"), kernel, mode="valid")
    return raw


def _rgb_to_lab_approx(rgb: np.ndarray) -> np.ndarray:
    """Approximate RGB→Lab via linearised sRGB→XYZ→Lab. Good enough for matching."""
    rgb_lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB → XYZ (D65)
    x = rgb_lin[..., 0] * 0.4124564 + rgb_lin[..., 1] * 0.3575761 + rgb_lin[..., 2] * 0.1804375
    y = rgb_lin[..., 0] * 0.2126729 + rgb_lin[..., 1] * 0.7151522 + rgb_lin[..., 2] * 0.0721750
    z = rgb_lin[..., 0] * 0.0193339 + rgb_lin[..., 1] * 0.1191920 + rgb_lin[..., 2] * 0.9503041
    # Normalise by D65 white point
    x /= 0.95047
    z /= 1.08883
    epsilon = 0.008856
    kappa = 903.3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return np.stack([L, a, b_ch], axis=-1)


def _sample_colour_strip(
    image: np.ndarray,
    mask: np.ndarray,
    side_pts: np.ndarray,
    inward_px: int = 5,
    n_samples: int = 32,
) -> np.ndarray:
    """Sample Lab colour along the inward edge of a side.

    Returns (n_samples, 3) Lab array. Pixels outside the mask are skipped.
    """
    if len(side_pts) < 2 or image.ndim < 3:
        return np.zeros((n_samples, 3))

    # Compute inward normal direction (perpendicular to side baseline, pointing into piece)
    a = side_pts[0].astype(np.float64)
    b = side_pts[-1].astype(np.float64)
    ab = b - a
    normal = np.array([-ab[1], ab[0]], dtype=np.float64)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-12:
        return np.zeros((n_samples, 3))
    normal /= normal_len

    h, w = mask.shape[:2]
    indices = np.linspace(0, len(side_pts) - 1, n_samples).astype(int)
    samples: list[np.ndarray] = []
    img_f = image.astype(np.float64) / 255.0 if image.max() > 1.0 else image.astype(np.float64)

    # Flip the normal if the first sample would land outside the piece.
    probe = side_pts[len(side_pts) // 2].astype(np.float64) + normal * inward_px
    px, py = int(round(probe[0])), int(round(probe[1]))
    if not (0 <= px < w and 0 <= py < h and mask[py, px] > 0):
        normal = -normal

    for idx in indices:
        pt = side_pts[idx].astype(np.float64) + normal * inward_px
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            samples.append(_rgb_to_lab_approx(img_f[y, x].reshape(1, 3))[0])
        else:
            samples.append(np.zeros(3))
    return np.array(samples)


class PieceDescriptorImpl(PieceDescriptor):
    """Fill `piece.sides` with corner-based segmentation, profiles, classification, and colour."""

    def __init__(
        self,
        corner_finder: CornerFinder | None = None,
        inward_px: int = 5,
        n_colour_samples: int = 32,
    ) -> None:
        self.corner_finder = corner_finder or HybridCornerFinder()
        self.inward_px = inward_px
        self.n_colour_samples = n_colour_samples

    def describe(self, piece: Piece) -> Piece:
        piece = deskew_piece(piece, tracer=MooreContourTracer())
        if piece.contour is None or len(piece.contour) < 4:
            ys, xs = np.nonzero(piece.mask)
            if len(xs):
                piece.contour = np.array(
                    [
                        [int(xs.min()), int(ys.min())],
                        [int(xs.max()), int(ys.min())],
                        [int(xs.max()), int(ys.max())],
                        [int(xs.min()), int(ys.max())],
                    ],
                    dtype=np.int32,
                )
            else:
                h, w = piece.mask.shape[:2]
                piece.contour = np.array(
                    [[0, 0], [max(w - 1, 0), 0], [max(w - 1, 0), max(h - 1, 0)], [0, max(h - 1, 0)]],
                    dtype=np.int32,
                )
        corners = self.corner_finder.find(piece.contour)
        piece.corners = corners

        ys, xs = np.nonzero(piece.mask)
        centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64) if len(xs) else corners.mean(axis=0)

        side_segments = _split_contour_by_corners(piece.contour, corners)
        sides: list[Side] = []
        for i, seg in enumerate(side_segments):
            profile = _signed_profile(seg)
            cls = _classify_side(profile, seg, centroid)
            colour = _sample_colour_strip(
                piece.image, piece.mask, seg,
                inward_px=self.inward_px,
                n_samples=self.n_colour_samples,
            )
            sides.append(Side(
                index=i,
                cls=cls,
                profile=profile,
                colour=colour,
                ribbon=pack_ribbon(colour, profile),
                contour_pts=seg,
            ))
        piece.sides = sides

        # Classify border/corner pieces
        flat_count = sum(1 for s in sides if s.cls == "flat")
        piece.is_border = flat_count >= 1
        piece.is_corner = flat_count >= 2

        return piece
