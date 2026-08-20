"""Run an approved YOLO training preset without machine-specific notebook paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRESETS_PATH = REPOSITORY_ROOT / "configs" / "TRAINING_PRESETS.json"

TRAIN_ARGUMENTS = {
    "epochs",
    "patience",
    "batch",
    "imgsz",
    "device",
    "workers",
    "pretrained",
    "optimizer",
    "seed",
    "deterministic",
    "cache",
    "amp",
}


def load_presets() -> dict[str, dict[str, Any]]:
    with PRESETS_PATH.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return document["experiments"]


def parse_args(experiment_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=experiment_names)
    parser.add_argument("--list", action="store_true", help="List available experiment IDs and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without training")
    parser.add_argument("--model", help="Override the pretrained model/checkpoint")
    parser.add_argument("--device", help="Override the Ultralytics device, for example 0 or cpu")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--workers", type=int, help="Override data-loader workers")
    parser.add_argument(
        "--output-root",
        default=str(REPOSITORY_ROOT / "runs" / "reproduction"),
        help="Directory for new run outputs",
    )
    args = parser.parse_args()
    if not args.list and not args.experiment:
        parser.error("--experiment is required unless --list is used")
    return args


def resolve_run(experiment_id: str, preset: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    data_path = (REPOSITORY_ROOT / preset["data"]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model_name = args.model or preset["model"]
    train_args = {key: value for key, value in preset.items() if key in TRAIN_ARGUMENTS}
    for key in ("device", "epochs", "batch", "workers"):
        override = getattr(args, key)
        if override is not None:
            train_args[key] = override

    train_args.update(
        {
            "data": str(data_path),
            "project": str(Path(args.output_root).resolve()),
            "name": experiment_id.lower(),
        }
    )
    return model_name, train_args


def main() -> None:
    presets = load_presets()
    args = parse_args(sorted(presets))
    if args.list:
        for experiment_id in sorted(presets):
            print(experiment_id)
        return

    preset = presets[args.experiment]
    model_name, train_args = resolve_run(args.experiment, preset, args)
    plan = {
        "experiment": args.experiment,
        "parameter_source": preset["parameter_source"],
        "model": model_name,
        "train": train_args,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("DRY RUN PASS: no training started")
        return

    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(**train_args)


if __name__ == "__main__":
    main()
