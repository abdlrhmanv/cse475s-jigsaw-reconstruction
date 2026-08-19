"""Infer an R×C board from the number of extracted pieces."""

from __future__ import annotations


def infer_board_shape(
    n_pieces: int,
    prefer_rows: int | None = None,
    prefer_cols: int | None = None,
) -> tuple[int, int]:
    """Return (rows, cols) whose product is ``n_pieces``.

    If the YAML preference already multiplies to N, keep it (35 → 7×5).
    Otherwise pick the factor pair whose aspect is closest to 7:5 (~1.4)
    so 9→3×3, 16→4×4, 35→7×5 rather than 1×N.
    """
    n = max(int(n_pieces), 1)
    if prefer_rows and prefer_cols and int(prefer_rows) * int(prefer_cols) == n:
        return int(prefer_rows), int(prefer_cols)

    pairs = [(r, n // r) for r in range(1, n + 1) if n % r == 0]
    target_aspect = 7 / 5

    def score(rc: tuple[int, int]) -> tuple[float, int, int]:
        rows, cols = rc
        aspect = rows / max(cols, 1)
        return (abs(aspect - target_aspect), abs(rows - cols), -rows)

    return min(pairs, key=score)


def count_for_grid(n_extracted: int, n_labelled: int | None) -> int:
    """Prefer the labelled piece count when CCL over-segments by a few blobs.

    A 9-piece photo that yields 10 CCL blobs should still assemble on 3×3,
    not 2×5. If extraction is far from the label count, trust the blobs.
    """
    extracted = max(int(n_extracted), 1)
    if n_labelled is None or int(n_labelled) <= 0:
        return extracted
    labelled = int(n_labelled)
    slack = max(2, labelled // 8)
    if abs(extracted - labelled) <= slack:
        return labelled
    return extracted
