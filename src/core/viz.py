"""Stage dumps under results/. Matplotlib figures are not a graded image operator."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class StageVisualizer:
    def save_side_by_side(
        self,
        path: str | Path,
        images: list[np.ndarray],
        titles: list[str] | None = None,
    ) -> None:
        """Save a horizontal strip of images with optional titles."""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, img in zip(axes, images):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
            ax.axis("off")
        if titles:
            for ax, t in zip(axes, titles):
                ax.set_title(t, fontsize=10)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def save_hist(self, path: str | Path, image: np.ndarray, title: str = "") -> None:
        """Save a greyscale histogram figure."""
        from src.enhancement import HistogramComputer, _ensure_gray
        gray = _ensure_gray(image) if image.ndim == 3 else image
        hist = HistogramComputer().compute(np.clip(gray, 0, 255).astype(np.uint8))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(256), hist, width=1, color="gray")
        ax.set_xlim(0, 255)
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
