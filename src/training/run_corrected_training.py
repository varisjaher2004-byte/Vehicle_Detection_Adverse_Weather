"""Run the audited DAWN and Combined YOLOv8l training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS = {
    "DAWN_FOG": ("dawn_fog.yaml", 4, 4),
    "DAWN_RAIN": ("dawn_rain.yaml", 4, 4),
    "DAWN_ENTIRE": ("dawn_entire.yaml", 2, 2),
    "COMBINED": ("combined_runtime.yaml", 2, 4),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_yaml(path: Path, acdc_root: Path, dawn_root: Path) -> None:
    names = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    body = [
        "path: .",
        "train:",
        f"  - {acdc_root.resolve().as_posix()}/images/train",
        f"  - {dawn_root.resolve().as_posix()}/images/train",
        "val:",
        f"  - {acdc_root.resolve().as_posix()}/images/val",
        f"  - {dawn_root.resolve().as_posix()}/images/val",
        "",
        "nc: 8",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(names)],
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dawn-root", type=Path, required=True)
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dawn = args.prepared_dawn_root.resolve()
    acdc = args.acdc_root.resolve()
    model = args.model.resolve()
    output = args.output_root.resolve()
    for required in (dawn / "dawn_fog.yaml", dawn / "dawn_rain.yaml", dawn / "dawn_entire.yaml", model):
        if not required.is_file():
            raise FileNotFoundError(required)
    for split in ("train", "val"):
        if not (acdc / "images" / split).is_dir() or not (acdc / "labels" / split).is_dir():
            raise FileNotFoundError(f"Incomplete ACDC {split} split")

    output.mkdir(parents=True, exist_ok=True)
    combined = output / "combined_runtime.yaml"
    combined_yaml(combined, acdc, dawn)
    plan = {
        name: {
            "data": str((dawn / yaml_name) if name != "COMBINED" else combined),
            "batch": batch,
            "workers": workers,
            "epochs": 100,
            "imgsz": 640,
            "seed": 0,
            "deterministic": True,
        }
        for name, (yaml_name, batch, workers) in EXPERIMENTS.items()
        if name in args.experiments
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("CORRECTED TRAINING DRY RUN PASS")
        return

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the locked training protocol")
    progress_path = output / "corrected_training_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiments": {}
    }
    for name, spec in plan.items():
        run_dir = output / name.lower()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        state = progress["experiments"].setdefault(name, {})
        if state.get("status") == "complete" and best.is_file():
            print(f"SKIP COMPLETE: {name}")
            continue
        state.update({"status": "running", "protocol": spec})
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        if last.is_file() and not best.is_file():
            YOLO(str(last)).train(resume=True)
        else:
            YOLO(str(model)).train(
                data=spec["data"], epochs=100, patience=100, batch=spec["batch"], imgsz=640,
                device="0", workers=spec["workers"], pretrained=True, optimizer="auto", seed=0,
                deterministic=True, cache=False, amp=True, plots=True, save=True,
                project=str(output), name=name.lower(), exist_ok=True, verbose=True,
            )
        if not best.is_file():
            raise FileNotFoundError(best)
        metrics = YOLO(str(best)).val(data=spec["data"], split="val", imgsz=640, batch=spec["batch"], device="0")
        precision, recall = float(metrics.box.mp), float(metrics.box.mr)
        state.update({
            "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_checkpoint_sha256": sha256(best),
            "metrics": {"precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
        })
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"COMPLETE: {name}")


if __name__ == "__main__":
    main()

