"""Run the Phase-8 experiment matrix (E0-E4) and write summary tables.

Default matchers are classical and Siamese. GNN is a weak extra
(real val_ap ~0.26); pass ``--methods classical,siamese,gnn`` for the ablation.

Reported scores are identity-neighbour accuracy and geometry Q. There is no
assembled-image ground truth, so SSIM is not computed.

Usage:
    python scripts/run_phase8.py
    python scripts/run_phase8.py --max-images-e4 5
    python scripts/run_phase8.py --methods classical,siamese,gnn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contour_extraction import count_yolo_classes
from src.core.factory import PipelineFactory


@dataclass
class BoardItem:
    image: Path
    label: Path
    n_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-8 E0-E4 experiments.")
    parser.add_argument("--test-dir", default="data/input/test")
    parser.add_argument("--labels-dir", default="data/ground_truth/test")
    parser.add_argument("--output-root", default="results/evaluation_results")
    parser.add_argument("--min-multipiece", type=int, default=9)
    parser.add_argument(
        "--max-images-e4",
        type=int,
        default=None,
        help="Optional cap for E4; omit for full held-out reconstruction set.",
    )
    parser.add_argument(
        "--e4-max-pieces",
        type=int,
        default=21,
        help="Skip boards larger than this in E4 (35-piece photos are slow). E1 still uses the hardest board.",
    )
    parser.add_argument(
        "--methods",
        default="classical,siamese",
        help="Comma-separated matchers. Default omits GNN (weak extra). Add gnn for the ablation.",
    )
    return parser.parse_args()


def count_unique_classes(label_path: Path) -> int:
    return count_yolo_classes(label_path)


def collect_boards(test_dir: Path, labels_dir: Path) -> list[BoardItem]:
    boards: list[BoardItem] = []
    for image_path in sorted(test_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        boards.append(
            BoardItem(
                image=image_path,
                label=label_path,
                n_classes=count_unique_classes(label_path),
            )
        )
    return boards


PDF_METRICS = (
    "Q",
    "quality_kind",
    "geometry_Q",
    "pose_gt",
    "position_accuracy",
    "orientation_accuracy",
    "neighbour_accuracy",
    "identity_neighbour_accuracy",
    "complete_reconstruction",
    "n_placed",
    "n_labelled",
    "n_pieces",
    "rows",
    "cols",
    "runtime_s",
    "extraction",
)


def run_one(
    method: str,
    config_path: Path,
    image_path: Path,
    output_dir: Path,
) -> dict:
    config = load_method_config(config_path, method)
    config["output_dir"] = str(output_dir)
    pipeline = PipelineFactory.from_config(config)
    t0 = time.perf_counter()
    result = pipeline.run(str(image_path), config)
    elapsed_s = time.perf_counter() - t0
    raw_dissim = float(result.metrics.get("total_dissim", result.state.total_dissim))
    row = {
        "method": method,
        "input": str(image_path),
        "total_dissim": raw_dissim if math.isfinite(raw_dissim) else None,
        "output_dir": str(output_dir),
        "runtime_s": round(elapsed_s, 3),
    }
    for key in PDF_METRICS:
        if key == "runtime_s":
            continue
        row[key] = result.metrics.get(key)
    row["n_pieces"] = int(result.metrics.get("n_pieces", 0))
    row["n_placed"] = int(result.metrics.get("n_placed", 0))
    row["rows"] = int(result.metrics.get("rows", 0))
    row["cols"] = int(result.metrics.get("cols", 0))
    return row


def load_method_config(config_path: Path, method: str) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["method"] = method
    return config


def write_tables(rows: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "phase8_summary.csv"
    json_path = out_root / "phase8_summary.json"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_comparison(rows, out_root)


def _mean(values: list) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def write_comparison(rows: list[dict], out_root: Path) -> None:
    """Mean PDF metrics on E4 (held-out test), one row per matcher."""
    e4 = [r for r in rows if r.get("experiment") == "E4"]
    methods = sorted({r["method"] for r in e4})
    summary: list[dict] = []
    numeric = [
        "Q",
        "geometry_Q",
        "position_accuracy",
        "neighbour_accuracy",
        "identity_neighbour_accuracy",
        "complete_reconstruction",
        "runtime_s",
        "n_placed",
        "n_labelled",
        "n_pieces",
    ]
    for method in methods:
        subset = [r for r in e4 if r["method"] == method]
        row: dict = {
            "method": method,
            "n_boards": len(subset),
            "orientation_accuracy": None,
        }
        for key in numeric:
            row[key] = _mean([r.get(key) for r in subset])
        summary.append(row)
    path = out_root / "phase8_comparison.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = out_root / "phase8_comparison.csv"
    if summary:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Wrote comparison table ({len(summary)} methods) to {csv_path}")


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir)
    labels_dir = Path(args.labels_dir)
    out_root = Path(args.output_root)
    boards = collect_boards(test_dir, labels_dir)
    if not boards:
        raise SystemExit("No test boards with labels found.")

    multipiece = [b for b in boards if b.n_classes >= args.min_multipiece]
    if not multipiece:
        raise SystemExit(
            f"No boards with >= {args.min_multipiece} unique classes found in {test_dir}."
        )

    # E0: smallest board (sanity), E1: largest board (rotations/hard case).
    e0_board = min(multipiece, key=lambda b: b.n_classes)
    e4_pool = [b for b in multipiece if b.n_classes <= args.e4_max_pieces]
    e1_board = max(e4_pool or multipiece, key=lambda b: b.n_classes)
    e4_boards = e4_pool[: args.max_images_e4] if args.max_images_e4 else e4_pool

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    allowed = {"classical", "siamese", "gnn"}
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise SystemExit(f"Unknown methods {unknown}; choose from {sorted(allowed)}.")
    configs = {
        "classical": Path("configs/classical.yaml"),
        "siamese": Path("configs/siamese.yaml"),
        "gnn": Path("configs/gnn.yaml"),
    }
    for m in methods:
        path = configs[m]
        if not path.exists():
            raise SystemExit(f"Missing config for {m}: {path}")

    rows: list[dict] = []

    # E0 baseline classical (or first requested method)
    e0_method = "classical" if "classical" in methods else methods[0]
    rows.append(
        {
            "experiment": "E0",
            "n_classes": e0_board.n_classes,
            **run_one(
                method=e0_method,
                config_path=configs[e0_method],
                image_path=e0_board.image,
                output_dir=out_root / "runs" / f"E0_{e0_method}",
            ),
        }
    )

    # E1 classical on hardest board
    if "classical" in methods:
        rows.append(
            {
                "experiment": "E1",
                "n_classes": e1_board.n_classes,
                **run_one(
                    method="classical",
                    config_path=configs["classical"],
                    image_path=e1_board.image,
                    output_dir=out_root / "runs" / "E1_classical",
                ),
            }
        )

    # E2 Siamese on same board as E1
    if "siamese" in methods:
        rows.append(
            {
                "experiment": "E2",
                "n_classes": e1_board.n_classes,
                **run_one(
                    method="siamese",
                    config_path=configs["siamese"],
                    image_path=e1_board.image,
                    output_dir=out_root / "runs" / "E2_siamese",
                ),
            }
        )

    # E3 GNN on same board as E1 (optional; weak extra)
    if "gnn" in methods:
        rows.append(
            {
                "experiment": "E3",
                "n_classes": e1_board.n_classes,
                **run_one(
                    method="gnn",
                    config_path=configs["gnn"],
                    image_path=e1_board.image,
                    output_dir=out_root / "runs" / "E3_gnn",
                ),
            }
        )

    # E4: requested methods on held-out test boards
    for board in e4_boards:
        for method in methods:
            tag = f"E4_{method}_{board.image.stem}"
            rows.append(
                {
                    "experiment": "E4",
                    "n_classes": board.n_classes,
                    **run_one(
                        method=method,
                        config_path=configs[method],
                        image_path=board.image,
                        output_dir=out_root / "runs" / tag,
                    ),
                }
            )

    write_tables(rows, out_root)
    print(f"Wrote {len(rows)} rows to {out_root / 'phase8_summary.csv'}")


if __name__ == "__main__":
    main()

