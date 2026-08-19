"""CLI entry. Operators are built only through PipelineFactory (never constructed here)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.core.factory import PipelineFactory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSE480 jigsaw reconstruction. Default matcher is classical; Siamese is the ML matcher; GNN is a weak extra."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reconstruct = sub.add_parser("reconstruct", help="Reconstruct a scrambled puzzle image.")
    reconstruct.add_argument(
        "--method",
        choices=("classical", "siamese", "gnn"),
        default="classical",
        help="classical (default), siamese (ML matcher), or gnn (weak extra / ablation).",
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

    train = sub.add_parser("train-siamese", help="Train the Siamese ribbon CNN.")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--ckpt", default="checkpoints/siamese.pt")
    train.add_argument("--real", action="store_true", help="Train on real boards using weak pose GT from piece IDs.")
    train.add_argument("--max-boards", type=int, default=80)
    train.add_argument("--data-dir", default="data")

    train_g = sub.add_parser("train-gnn", help="Train the side-node GNN (weak extra; prefer Siamese).")
    train_g.add_argument("--epochs", type=int, default=40)
    train_g.add_argument("--ckpt", default="checkpoints/gnn.pt")
    train_g.add_argument("--real", action="store_true", help="Train on real boards using weak pose GT from piece IDs.")
    train_g.add_argument("--max-boards", type=int, default=40)
    train_g.add_argument("--data-dir", default="data")
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
        train_siamese(
            epochs=args.epochs,
            batch_size=args.batch_size,
            ckpt_path=args.ckpt,
            real=args.real,
            max_boards=args.max_boards,
            data_dir=args.data_dir,
        )
        return
    if args.command == "train-gnn":
        from src.ml.train_gnn import train_gnn
        train_gnn(
            epochs=args.epochs,
            ckpt_path=args.ckpt,
            real=args.real,
            max_boards=args.max_boards,
            data_dir=args.data_dir,
        )
        return
    if args.command != "reconstruct":
        raise SystemExit(2)
    config = load_config(args.config, args.method)
    pipeline = PipelineFactory.from_config(config)
    result = pipeline.run(str(Path(args.input)), config)
    print(result)


if __name__ == "__main__":
    main()
