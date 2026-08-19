"""Four corners, tab/blank/flat class, colour strips (Phase 3)."""

from __future__ import annotations

import numpy as np

from src.core.protocols import CornerFinder, PieceDescriptor
from src.core.types import Piece, Side, SideClass


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
    # Rotate so that the top-left (smallest x+y) is first
    sums = ordered[:, 0] + ordered[:, 1]
    shift = int(np.argmin(sums))
    return np.roll(ordered, -shift, axis=0)


# ---------------------------------------------------------------------------
# Corner finders
# ---------------------------------------------------------------------------

class HybridCornerFinder(CornerFinder):
    """Hull → RDP simplification → pick 4 points with best interior angles.

    True jigsaw corners are near 90° on the convex hull; tab tips are concave
    and filtered out by requiring hull membership.
    """

    def __init__(self, epsilon_frac: float = 0.02) -> None:
        self.epsilon_frac = epsilon_frac

    def find(self, contour: np.ndarray) -> np.ndarray:
        if len(contour) < 4:
            return contour[:4] if len(contour) > 0 else np.zeros((4, 2), dtype=np.int32)

        hull_idx = _convex_hull(contour)
        hull_pts = contour[hull_idx]

        perimeter = float(np.sum(np.linalg.norm(np.diff(hull_pts, axis=0, append=hull_pts[:1]), axis=1)))
        eps = self.epsilon_frac * perimeter
        simplified = _rdp(hull_pts, eps)

        # Close the polygon for angle computation
        if len(simplified) < 4:
            simplified = hull_pts

        n = len(simplified)
        # Score each vertex by how close its interior angle is to 90°
        scores: list[tuple[float, int]] = []
        for i in range(n):
            a = simplified[(i - 1) % n]
            b = simplified[i]
            c = simplified[(i + 1) % n]
            angle = _interior_angle(a, b, c)
            scores.append((abs(angle - 90.0), i))
        scores.sort()

        # Pick 4 best-scoring vertices
        selected = sorted([s[1] for s in scores[:4]])
        corners = simplified[selected]
        return _order_corners_clockwise(corners)


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
    """Split contour into 4 side segments between consecutive corner points.

    Returns list of 4 arrays, each (K_i, 2): points from corner i to corner i+1.
    """
    n = len(contour)
    # Find contour index closest to each corner
    corner_indices: list[int] = []
    for c in corners:
        dists = np.linalg.norm(contour.astype(np.float64) - c.astype(np.float64), axis=1)
        corner_indices.append(int(np.argmin(dists)))

    # Sort corner indices along contour order
    corner_indices.sort()

    sides: list[np.ndarray] = []
    for i in range(4):
        start = corner_indices[i]
        end = corner_indices[(i + 1) % 4]
        if end > start:
            sides.append(contour[start : end + 1])
        else:
            sides.append(np.vstack([contour[start:], contour[: end + 1]]))
    return sides


def _classify_side(profile: np.ndarray) -> SideClass:
    """Classify a side as tab, blank, or flat based on its signed profile.

    Profile is the signed perpendicular distance from the corner-to-corner line.
    Positive = outward (tab), negative = inward (blank), near-zero = flat.
    """
    if len(profile) == 0:
        return "flat"
    peak = float(np.max(np.abs(profile)))
    mean_dev = float(np.mean(profile))
    if peak < 3.0:
        return "flat"
    return "tab" if mean_dev > 0 else "blank"


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
    return (ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / ab_len


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
    # Subsample side points
    indices = np.linspace(0, len(side_pts) - 1, n_samples).astype(int)
    samples: list[np.ndarray] = []
    img_f = image.astype(np.float64) / 255.0 if image.max() > 1.0 else image.astype(np.float64)
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
        corners = self.corner_finder.find(piece.contour)
        piece.corners = corners

        side_segments = _split_contour_by_corners(piece.contour, corners)
        sides: list[Side] = []
        for i, seg in enumerate(side_segments):
            profile = _signed_profile(seg)
            cls = _classify_side(profile)
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
                ribbon=np.empty((0,)),  # filled in Milestone 2
                contour_pts=seg,
            ))
        piece.sides = sides

        # Classify border/corner pieces
        flat_count = sum(1 for s in sides if s.cls == "flat")
        piece.is_border = flat_count >= 1
        piece.is_corner = flat_count >= 2

        return piece
