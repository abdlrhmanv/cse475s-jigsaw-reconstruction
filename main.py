"""CLI entry. Operators are built only through PipelineFactory (never constructed here)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.core.factory import PipelineFactory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSE475s jigsaw reconstruction (classical, Siamese, or GNN matching)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reconstruct = sub.add_parser("reconstruct", help="Reconstruct a scrambled puzzle image.")
    reconstruct.add_argument(
        "--method",
        choices=("classical", "siamese", "gnn"),
        default="classical",
        help="Which CompatibilityMatcher to use. Extraction and assembly stay the same.",
    )
    reconstruct.add_argument(
        "--config",
        default="configs/classical.yaml",
        help="YAML config path.",
    )
    reconstruct.add_argument(
        "--input",
        required=True,
        help="Path to a scrambled puzzle image.",
    )

    train = sub.add_parser("train-siamese", help="Train the Siamese ribbon CNN (synthetic pairs by default).")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--ckpt", default="checkpoints/siamese.pt")

    train_g = sub.add_parser("train-gnn", help="Train the side-node GNN on synthetic grids.")
    train_g.add_argument("--epochs", type=int, default=40)
    train_g.add_argument("--ckpt", default="checkpoints/gnn.pt")
    return parser.parse_args(argv)


def load_config(path: str, method: str) -> dict:
    """Load YAML then overwrite `method` so CLI comparison does not require editing the file."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["method"] = method
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train-siamese":
        from src.ml.train_siamese import train_siamese
        train_siamese(epochs=args.epochs, batch_size=args.batch_size, ckpt_path=args.ckpt)
        return
    if args.command == "train-gnn":
        from src.ml.train_gnn import train_gnn
        train_gnn(epochs=args.epochs, ckpt_path=args.ckpt)
        return
    if args.command != "reconstruct":
        raise SystemExit(2)
    config = load_config(args.config, args.method)
    pipeline = PipelineFactory.from_config(config)
    result = pipeline.run(str(Path(args.input)), config)
    print(result)


if __name__ == "__main__":
    main()
