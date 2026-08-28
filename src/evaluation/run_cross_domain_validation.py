"""Evaluate locked ACDC, corrected DAWN and Combined checkpoints consistently."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yaml_for(path: Path, train_dirs: list[Path], val_dirs: list[Path]) -> None:
    lines = ["path: .", "train:", *[f"  - {p.resolve().as_posix()}" for p in train_dirs],
             "val:", *[f"  - {p.resolve().as_posix()}" for p in val_dirs], "", "nc: 8", "names:",
             *[f"  {index}: {name}" for index, name in enumerate(NAMES)]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-checkpoint", type=Path, required=True)
    parser.add_argument("--dawn-checkpoint", type=Path, required=True)
    parser.add_argument("--combined-checkpoint", type=Path, required=True)
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--prepared-dawn-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoints = {"ACDC": args.acdc_checkpoint.resolve(), "DAWN": args.dawn_checkpoint.resolve(),
                   "ACDC+DAWN": args.combined_checkpoint.resolve()}
    if any(not path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("One or more checkpoint paths do not exist")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    acdc = args.acdc_root.resolve()
    dawn = args.prepared_dawn_root.resolve()
    configs = output / "runtime_configs"
    configs.mkdir(exist_ok=True)
    yaml_for(configs / "acdc.yaml", [acdc / "images" / "train"], [acdc / "images" / "val"])
    yaml_for(configs / "dawn.yaml", [dawn / "images" / "train"], [dawn / "images" / "val"])
    yaml_for(configs / "union.yaml", [acdc / "images" / "train", dawn / "images" / "train"],
             [acdc / "images" / "val", dawn / "images" / "val"])
    plan = [
        ("ACDC", "ACDC", "within-domain control"),
        ("ACDC", "DAWN", "cross-domain transfer"),
        ("DAWN", "ACDC", "cross-domain transfer"),
        ("DAWN", "DAWN", "within-domain control"),
        ("ACDC+DAWN", "ACDC", "combined constituent-domain"),
        ("ACDC+DAWN", "DAWN", "combined constituent-domain"),
        ("ACDC+DAWN", "ACDC_DAWN_UNION", "combined within-domain control"),
    ]
    print(json.dumps({"checkpoints": {k: sha256(v) for k, v in checkpoints.items()}, "plan": plan}, indent=2))
    if args.dry_run:
        print("CROSS-DOMAIN VALIDATION DRY RUN PASS")
        return

    import torch
    from ultralytics import YOLO

    data = {
        "ACDC": configs / "acdc.yaml",
        "DAWN": configs / "dawn.yaml",
        "ACDC_DAWN_UNION": configs / "union.yaml",
    }
    rows: list[dict[str, object]] = []
    for model_domain, evaluation_domain, relation in plan:
        metrics = YOLO(str(checkpoints[model_domain])).val(
            data=str(data[evaluation_domain]), split="val", imgsz=640, batch=4,
            device="0" if torch.cuda.is_available() else "cpu", workers=4,
            project=str(output / "outputs"), name=f"{model_domain}_on_{evaluation_domain}".replace("+", "_plus_"),
            exist_ok=True, plots=True, verbose=True,
        )
        precision, recall = float(metrics.box.mp), float(metrics.box.mr)
        rows.append({
            "model_training_domain": model_domain, "evaluation_domain": evaluation_domain,
            "relationship": relation, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "checkpoint_sha256": sha256(checkpoints[model_domain]),
        })
    with (output / "final_cross_domain_validation_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "final_cross_domain_validation_matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("FINAL CROSS-DOMAIN VALIDATION MATRIX COMPLETE")


if __name__ == "__main__":
    main()
