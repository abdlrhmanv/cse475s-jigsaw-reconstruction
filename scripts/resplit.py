"""Re-split dataset so val/test contain multi-piece boards, not 1-piece photos.

Rule
----
- Images with >= MIN_PIECES unique labelled pieces are the reconstruction pool.
  They are split 80/10/10, stratified by piece-count bucket (9–15 / 16–21 / 35).
- All remaining images (0–8 pieces, mostly isolated singles) stay in train.
  They are useful for detection/enhancement, not for reconstruction eval.

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

import numpy as np

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


def assign_multi(items: list[tuple[Path, Path | None, int]], rng: np.random.Generator) -> dict[str, str]:
    """Map image stem → split, stratified by piece-count bucket."""
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for img, _, n in items:
        by_bucket[bucket(n)].append(img.stem)

    assignment: dict[str, str] = {}
    for names in by_bucket.values():
        names = list(names)
        rng.shuffle(names)
        n = len(names)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        n_test = n - n_train - n_val
        # Keep at least one image in val and test when the bucket is large enough.
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train -= 1
        if n >= 3 and n_val == 0:
            n_val = 1
            n_train -= 1
        for i, stem in enumerate(names):
            if i < n_train:
                assignment[stem] = "train"
            elif i < n_train + n_val:
                assignment[stem] = "val"
            else:
                assignment[stem] = "test"
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
    multi = [r for r in rows if r[2] >= MIN_PIECES]
    singles = [r for r in rows if r[2] < MIN_PIECES]

    rng = np.random.default_rng(SEED)
    multi_assign = assign_multi(multi, rng)

    plan: dict[str, list[tuple[Path, Path | None, int]]] = {"train": [], "val": [], "test": []}
    for img, lbl, n in singles:
        plan["train"].append((img, lbl, n))
    for img, lbl, n in multi:
        plan[multi_assign[img.stem]].append((img, lbl, n))

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
    print(
        f"\nDone. splits.json: train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
