"""Stage dumps under results/. Matplotlib figures are not a graded image operator."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StageVisualizer:
    def save_side_by_side(
        self,
        path: str | Path,
        images: list[np.ndarray],
        titles: list[str] | None = None,
    ) -> None:
        raise NotImplementedError("Phase 1: matplotlib stage dumps.")

    def save_hist(self, path: str | Path, image: np.ndarray, title: str = "") -> None:
        raise NotImplementedError("Phase 1: histogram figures.")
