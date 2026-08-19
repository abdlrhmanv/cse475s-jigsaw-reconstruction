"""Re-split dataset so val/test contain multi-piece boards, not 1-piece photos.

Rule
----
- Group every Roboflow variant (`name.rf.*`) by source prefix.
- If any file of a source has >= MIN_PIECES unique labelled pieces, the whole
  source is reconstruction-eligible and is split 80/10/10, stratified by that
  source's max piece-count bucket (9–15 / 16–21 / 35).
- Sources that are only singles stay in train.

Usage
-----
    python scripts/resplit.py          # dry-run
    python scripts/resplit.py --apply  # move files + rewrite splits.json
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_PIECES = 9
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def unique_classes(label_path: Path) -> set[str]:
    if not label_path.exists():
        return set()
    classes: set[str] = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            classes.add(line.split()[0])
    return classes


def collect_files(data_dir: Path) -> list[tuple[Path, Path | None, int]]:
    """Return (image_path, label_path|None, n_unique_classes) for every image."""
    rows: list[tuple[Path, Path | None, int]] = []
    for split in ("train", "val", "test"):
        img_dir = data_dir / "input" / split
        lbl_dir = data_dir / "ground_truth" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            n = len(unique_classes(lbl)) if lbl.exists() else 0
            rows.append((img, lbl if lbl.exists() else None, n))
    return rows


def bucket(n: int) -> str:
    if n >= 30:
        return "full35"
    if n >= 16:
        return "mid"
    return "small"


def assign_by_source(
    rows: list[tuple[Path, Path | None, int]],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Map image stem → split. Every Roboflow variant of one photo stays together.

    A source is reconstruction-eligible if any of its files has >= MIN_PIECES
    unique labels. Those sources are split 80/10/10 by max piece-count bucket.
    Sources that are only singles all go to train. This prevents a 1-piece
    augmentation landing in train while a 9-piece sibling sits in val/test.
    """
    from src.ml.splits import source_stem

    groups: dict[str, list[str]] = defaultdict(list)
    max_n: dict[str, int] = {}
    for img, _, n in rows:
        src = source_stem(img.stem)
        groups[src].append(img.stem)
        max_n[src] = max(max_n.get(src, 0), n)

    by_bucket: dict[str, list[str]] = defaultdict(list)
    assignment: dict[str, str] = {}
    for src, pieces in max_n.items():
        if pieces >= MIN_PIECES:
            by_bucket[bucket(pieces)].append(src)
        else:
            for stem in groups[src]:
                assignment[stem] = "train"

    for sources in by_bucket.values():
        sources = list(sources)
        rng.shuffle(sources)
        n = len(sources)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        n_test = n - n_train - n_val
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train -= 1
        if n >= 3 and n_val == 0:
            n_val = 1
            n_train -= 1
        for i, src in enumerate(sources):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            for stem in groups[src]:
                assignment[stem] = split
    return assignment


def move_pair(img: Path, lbl: Path | None, data_dir: Path, split: str) -> None:
    dst_img = data_dir / "input" / split / img.name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if img.resolve() != dst_img.resolve():
        shutil.move(str(img), str(dst_img))
    if lbl is not None:
        dst_lbl = data_dir / "ground_truth" / split / lbl.name
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if lbl.resolve() != dst_lbl.resolve():
            shutil.move(str(lbl), str(dst_lbl))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = collect_files(data_dir)

    rng = np.random.default_rng(SEED)
    stem_split = assign_by_source(rows, rng)

    plan: dict[str, list[tuple[Path, Path | None, int]]] = {"train": [], "val": [], "test": []}
    missing = [img.stem for img, _, _ in rows if img.stem not in stem_split]
    if missing:
        raise SystemExit(f"Unassigned stems: {missing[:5]}")
    for img, lbl, n in rows:
        plan[stem_split[img.stem]].append((img, lbl, n))

    for split in ("train", "val", "test"):
        items = plan[split]
        ns = [n for _, _, n in items]
        n_full = sum(n >= 30 for n in ns)
        n_mid = sum(16 <= n < 30 for n in ns)
        n_small = sum(MIN_PIECES <= n < 16 for n in ns)
        n_single = sum(n < MIN_PIECES for n in ns)
        print(
            f"{split:5s}: {len(items):4d} images | "
            f"35-piece={n_full:2d}  16–21={n_mid:2d}  9–15={n_small:3d}  singles={n_single:4d}  "
            f"median_pieces={sorted(ns)[len(ns)//2] if ns else 0}"
        )

    if not args.apply:
        print("\nDry run. Pass --apply to move files.")
        return

    for split, items in plan.items():
        for img, lbl, _ in items:
            move_pair(img, lbl, data_dir, split)

    from src.ml.splits import check_no_leakage, freeze_splits

    splits = freeze_splits(data_dir)
    check_no_leakage(splits)
    from src.ml.splits import check_no_source_leakage
    check_no_source_leakage(splits)
    print(
        f"\nDone. splits.json: train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
