"""Connected components from scratch (Phase 2). Two-pass with union-find."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Labeler


class ConnectedComponentLabeler(Labeler):
    """Two-pass 8-connected CCL with union-find path compression.

    Filters labels by min_area / max_area to drop dust specks and
    merged-piece blobs before contour tracing.
    """

    def __init__(self, min_area: int = 50, max_area: int | None = None) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def label(self, binary: np.ndarray) -> np.ndarray:
        fg = binary > 0
        h, w = fg.shape
        labels = np.zeros((h, w), dtype=np.int32)
        parent: dict[int, int] = {}
        next_label = 1

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 8-connected neighbour offsets that precede (r, c) in raster order
        nbr_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

        # Pass 1: provisional labels + record equivalences
        for r in range(h):
            for c in range(w):
                if not fg[r, c]:
                    continue
                neighbours = []
                for dr, dc in nbr_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] > 0:
                        neighbours.append(labels[nr, nc])

                if not neighbours:
                    labels[r, c] = next_label
                    parent[next_label] = next_label
                    next_label += 1
                else:
                    min_lbl = min(find(n) for n in neighbours)
                    labels[r, c] = min_lbl
                    for n in neighbours:
                        union(n, min_lbl)

        # Pass 2: resolve all labels to canonical roots
        remap: dict[int, int] = {}
        canonical = 0
        for r in range(h):
            for c in range(w):
                lbl = labels[r, c]
                if lbl == 0:
                    continue
                root = find(lbl)
                if root not in remap:
                    canonical += 1
                    remap[root] = canonical
                labels[r, c] = remap[root]

        # Area filtering
        if self.min_area > 0 or self.max_area is not None:
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                if lbl == 0:
                    continue
                area = int(np.sum(labels == lbl))
                if area < self.min_area:
                    labels[labels == lbl] = 0
                elif self.max_area is not None and area > self.max_area:
                    labels[labels == lbl] = 0

            # Re-number sequentially after filtering
            old_labels = np.unique(labels)
            new_map = np.zeros(old_labels.max() + 1 if len(old_labels) > 0 else 1, dtype=np.int32)
            seq = 0
            for ol in old_labels:
                if ol == 0:
                    continue
                seq += 1
                new_map[ol] = seq
            labels = new_map[labels]

        return labels
