"""Connected components from scratch (Phase 2). Two-pass with union-find."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Labeler


class ConnectedComponentLabeler(Labeler):
    """Two-pass 8-connected CCL with union-find path compression.

    Filters labels by min_area / max_area, bbox aspect ratio (thin tools),
    and optionally keeps only the `keep_n` largest blobs.
    """

    def __init__(
        self,
        min_area: int = 50,
        max_area: int | None = None,
        min_area_frac: float = 0.0,
        max_aspect: float | None = None,
        keep_n: int | None = None,
        min_solidity: float = 0.0,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_area_frac = min_area_frac
        self.max_aspect = max_aspect
        self.keep_n = keep_n
        self.min_solidity = min_solidity

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

        labels = self._filter_blobs(labels)
        return labels

    def _filter_blobs(self, labels: np.ndarray) -> np.ndarray:
        h, w = labels.shape
        min_area = self.min_area
        if self.min_area_frac > 0:
            min_area = max(min_area, int(self.min_area_frac * h * w))

        counts = np.bincount(labels.ravel())
        drop: set[int] = set()
        areas: list[tuple[int, int]] = []  # (area, label)

        for lbl in range(1, len(counts)):
            area = int(counts[lbl])
            if area == 0:
                continue
            if area < min_area:
                drop.add(lbl)
                continue
            if self.max_area is not None and area > self.max_area:
                drop.add(lbl)
                continue
            if self.max_aspect is not None:
                ys, xs = np.nonzero(labels == lbl)
                bh = int(ys.max() - ys.min()) + 1
                bw = int(xs.max() - xs.min()) + 1
                aspect = max(bh, bw) / max(min(bh, bw), 1)
                if aspect > self.max_aspect:
                    drop.add(lbl)
                    continue
                if self.min_solidity > 0:
                    solidity = area / max(bh * bw, 1)
                    if solidity < self.min_solidity:
                        drop.add(lbl)
                        continue
            elif self.min_solidity > 0:
                ys, xs = np.nonzero(labels == lbl)
                bh = int(ys.max() - ys.min()) + 1
                bw = int(xs.max() - xs.min()) + 1
                solidity = area / max(bh * bw, 1)
                if solidity < self.min_solidity:
                    drop.add(lbl)
                    continue
            areas.append((area, lbl))

        if self.keep_n is not None and len(areas) > self.keep_n:
            areas.sort(reverse=True)
            for _, lbl in areas[self.keep_n :]:
                drop.add(lbl)

        if drop:
            mask = np.isin(labels, list(drop))
            labels = labels.copy()
            labels[mask] = 0

        old_labels = np.unique(labels)
        new_map = np.zeros(int(old_labels.max()) + 1 if len(old_labels) > 0 else 1, dtype=np.int32)
        seq = 0
        for ol in old_labels:
            if ol == 0:
                continue
            seq += 1
            new_map[ol] = seq
        return new_map[labels]
