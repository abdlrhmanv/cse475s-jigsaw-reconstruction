"""Build real side-pairs / graphs from described train/val boards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.contour_extraction import YoloBoxExtractor, count_yolo_classes, gt_label_path
from src.core.io import ImageStore
from src.ml.augmentation import RibbonAugmenter
from src.ml.gnn_graph import SideGraph, build_side_graph
from src.ml.pair_dataset import PairGenerator, SidePair
from src.ml.pose_gt import load_piece_names, weak_side_adjacency
from src.ml.splits import load_splits
from src.piece_description import PieceDescriptorImpl


def unique_label_count(image_path: str | Path) -> int:
    return count_yolo_classes(gt_label_path(image_path))


def split_image_paths(
    split: str,
    data_dir: str | Path = "data",
    min_pieces: int = 9,
    max_pieces: int | None = 21,
) -> list[Path]:
    data_dir = Path(data_dir)
    stems = load_splits(data_dir / "splits.json").get(split, [])
    paths: list[Path] = []
    folder = data_dir / "input" / split
    for stem in stems:
        cand = None
        for suffix in (".jpg", ".jpeg", ".png"):
            maybe = folder / f"{stem}{suffix}"
            if maybe.exists():
                cand = maybe
                break
        if cand is None:
            continue
        n = unique_label_count(cand)
        if n < min_pieces:
            continue
        if max_pieces is not None and n > max_pieces:
            continue
        paths.append(cand)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    return [paths[i] for i in order]


def describe_board(image_path: str | Path) -> list:
    """YOLO-crop + describe. Identities come from class ids; needed for pose GT."""
    path = Path(image_path)
    labels = gt_label_path(path)
    if labels is None:
        return []
    image = ImageStore().load(path)
    pieces = YoloBoxExtractor().extract(image, labels)
    descriptor = PieceDescriptorImpl()
    return [descriptor.describe(p) for p in pieces]


def pairs_from_split(
    split: str,
    data_dir: str | Path = "data",
    max_boards: int | None = 80,
    augment: bool = False,
    min_pieces: int = 9,
) -> list[SidePair]:
    names = load_piece_names()
    gen = PairGenerator(
        neg_ratio=1,
        augmenter=RibbonAugmenter() if augment else None,
    )
    collected: list[SidePair] = []
    n_used = 0
    for image_path in split_image_paths(split, data_dir, min_pieces=min_pieces):
        if max_boards is not None and n_used >= max_boards:
            break
        pieces = describe_board(image_path)
        if len(pieces) < min_pieces:
            continue
        adjacency = weak_side_adjacency(pieces, names=names)
        if not adjacency:
            continue
        collected.extend(gen.generate(pieces, adjacency, augment=augment))
        n_used += 1
        print(f"  {split} board {n_used}: {image_path.name}  pieces={len(pieces)}  pairs={len(collected)}")
    return collected


def graphs_from_split(
    split: str,
    data_dir: str | Path = "data",
    max_boards: int | None = 40,
    min_pieces: int = 9,
) -> list[SideGraph]:
    names = load_piece_names()
    graphs: list[SideGraph] = []
    for image_path in split_image_paths(split, data_dir, min_pieces=min_pieces):
        if max_boards is not None and len(graphs) >= max_boards:
            break
        pieces = describe_board(image_path)
        if len(pieces) < min_pieces:
            continue
        adjacency = weak_side_adjacency(pieces, names=names)
        if not adjacency:
            continue
        graphs.append(build_side_graph(pieces, adjacency=adjacency, top_k=8))
        print(f"  {split} graph {len(graphs)}: {image_path.name}  pieces={len(pieces)}")
    return graphs
