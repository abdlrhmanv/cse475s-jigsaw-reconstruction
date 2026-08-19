"""Freeze and load train/val/test splits (Phase 5).

The splits.json file records which image stems belong to each split.
Once frozen, this file must never change — it guarantees reproducibility
and prevents test leakage across experiments.
"""

from __future__ import annotations

import json
from pathlib import Path


def freeze_splits(
    data_dir: str | Path,
    output_path: str | Path = "data/splits.json",
) -> dict[str, list[str]]:
    """Scan data/input/{train,val,test} and write a deterministic splits.json.

    Returns the splits dict.
    """
    data_dir = Path(data_dir)
    splits: dict[str, list[str]] = {}

    for split in ("train", "val", "test"):
        folder = data_dir / "input" / split
        if not folder.exists():
            splits[split] = []
            continue
        stems = sorted(p.stem for p in folder.iterdir() if p.suffix in {".jpg", ".jpeg", ".png"})
        splits[split] = stems

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_splits(path: str | Path = "data/splits.json") -> dict[str, list[str]]:
    """Load a previously frozen splits.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_no_leakage(splits: dict[str, list[str]]) -> None:
    """Raise ValueError if any stem appears in more than one split."""
    all_splits = list(splits.keys())
    for i, s1 in enumerate(all_splits):
        for s2 in all_splits[i + 1 :]:
            overlap = set(splits[s1]) & set(splits[s2])
            if overlap:
                raise ValueError(
                    f"Leakage: {len(overlap)} stems in both {s1} and {s2}: "
                    f"{sorted(overlap)[:5]}..."
                )


def source_stem(stem: str) -> str:
    """Roboflow `name.rf.hash` → `name` so augmentations of one photo group together."""
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem


def check_no_source_leakage(splits: dict[str, list[str]]) -> None:
    """Raise if the same source image (before `.rf.`) appears in two splits."""
    keys = list(splits.keys())
    grouped = {k: {source_stem(s) for s in splits[k]} for k in keys}
    for i, s1 in enumerate(keys):
        for s2 in keys[i + 1 :]:
            overlap = grouped[s1] & grouped[s2]
            if overlap:
                raise ValueError(
                    f"Source leakage: {len(overlap)} images in both {s1} and {s2}: "
                    f"{sorted(overlap)[:5]}..."
                )
