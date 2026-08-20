"""Verify portable dataset configs and the approved real-world evidence manifest."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "EVIDENCE_MANIFEST.csv"
PRESETS_PATH = REPOSITORY_ROOT / "configs" / "TRAINING_PRESETS.json"

CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CONFIG_PATHS = {
    "configs/ACDC/acdc_entire.yaml": "ACDC/YOLO_ENTIRE",
    "configs/ACDC/acdc_fog.yaml": "ACDC/YOLO_FOG",
    "configs/ACDC/acdc_night.yaml": "ACDC/YOLO_NIGHT",
    "configs/ACDC/acdc_rain.yaml": "ACDC/YOLO_RAIN",
    "configs/ACDC/acdc_snow.yaml": "ACDC/YOLO_SNOW",
    "configs/DAWN/dawn_entire.yaml": "DAWN/YOLO_ENTIRE",
    "configs/DAWN/dawn_fog.yaml": "DAWN/YOLO_FOG",
    "configs/DAWN/dawn_rain.yaml": "DAWN/YOLO_RAIN",
    "configs/COMBINED/combined.yaml": "YOLO_COMBINED",
}

APPROVED_RESULTS = {
    "results/ACDC/ENTIRE/results.csv",
    "results/ACDC/FOG/results.csv",
    "results/ACDC/NIGHT/results.csv",
    "results/ACDC/RAIN/results.csv",
    "results/ACDC/SNOW/results.csv",
    "results/DAWN/ENTIRE/results.csv",
    "results/DAWN/FOG/results.csv",
    "results/DAWN/RAIN/results.csv",
    "results/COMBINED/results.csv",
}

SOURCE_COLUMNS = {
    "epoch_index": "epoch",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def _normalise_names(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(value[index] if index in value else value[str(index)]) for index in range(len(value))]
    raise AssertionError("names must be a list or numeric-key mapping")


def verify_configs() -> None:
    for relative_path, expected_root in CONFIG_PATHS.items():
        config_path = REPOSITORY_ROOT / relative_path
        assert config_path.is_file(), f"Missing config: {relative_path}"
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert config["path"] == expected_root, f"Unexpected dataset root in {relative_path}"
        assert not Path(config["path"]).is_absolute(), f"Absolute path in {relative_path}"
        assert config["train"] == "images/train", f"Unexpected train split in {relative_path}"
        assert config["val"] == "images/val", f"Unexpected val split in {relative_path}"
        assert int(config["nc"]) == len(CLASS_NAMES), f"Unexpected nc in {relative_path}"
        assert _normalise_names(config["names"]) == CLASS_NAMES, f"Unexpected names in {relative_path}"
        print(f"CONFIG PASS: {relative_path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"No rows in {path.relative_to(REPOSITORY_ROOT)}"
    return rows


def _as_float(row: dict[str, str], column: str) -> float:
    return float(row[column].strip())


def verify_manifest() -> None:
    manifest_rows = _read_csv(MANIFEST_PATH)
    listed_sources = {row["source_csv"] for row in manifest_rows}
    assert listed_sources == APPROVED_RESULTS, "Manifest membership differs from the approved result set"
    assert len(manifest_rows) == len(APPROVED_RESULTS), "Manifest contains duplicate result sources"

    for manifest_row in manifest_rows:
        source_relative = manifest_row["source_csv"]
        source_path = REPOSITORY_ROOT / source_relative
        assert source_path.is_file(), f"Missing result source: {source_relative}"
        assert manifest_row["selection_rule"] == "maximum metrics/mAP50(B)", (
            f"Unexpected selection rule for {source_relative}"
        )

        source_rows = _read_csv(source_path)
        selected = max(source_rows, key=lambda row: _as_float(row, "metrics/mAP50(B)"))

        assert int(manifest_row["epoch_index"]) == int(float(selected["epoch"])), (
            f"Epoch mismatch for {source_relative}"
        )
        for manifest_column, source_column in SOURCE_COLUMNS.items():
            if manifest_column == "epoch_index":
                continue
            actual = float(manifest_row[manifest_column])
            expected = _as_float(selected, source_column)
            assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-10), (
                f"{manifest_column} mismatch for {source_relative}: {actual} != {expected}"
            )

        precision = _as_float(selected, "metrics/precision(B)")
        recall = _as_float(selected, "metrics/recall(B)")
        expected_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        actual_f1 = float(manifest_row["f1"])
        assert math.isclose(actual_f1, expected_f1, rel_tol=0.0, abs_tol=5e-10), (
            f"F1 mismatch for {source_relative}: {actual_f1} != {expected_f1}"
        )
        print(f"EVIDENCE PASS: {manifest_row['experiment_id']}")


def verify_training_presets() -> None:
    with PRESETS_PATH.open(encoding="utf-8") as handle:
        document = json.load(handle)
    presets = document["experiments"]
    expected_ids = {path.split("/")[1] + "_" + path.split("/")[2] for path in APPROVED_RESULTS}
    expected_ids.discard("COMBINED_results.csv")
    expected_ids.add("COMBINED")
    assert set(presets) == expected_ids, "Training preset membership differs from approved experiments"

    for experiment_id, preset in presets.items():
        data_path = REPOSITORY_ROOT / preset["data"]
        source_path = REPOSITORY_ROOT / preset["parameter_source"]
        assert data_path.is_file(), f"Missing preset data config for {experiment_id}"
        assert source_path.is_file(), f"Missing parameter source for {experiment_id}"
        assert preset["model"] == "yolov8l.pt", f"Unexpected model for {experiment_id}"
        assert int(preset["epochs"]) == 100, f"Unexpected epochs for {experiment_id}"
        assert int(preset["imgsz"]) == 640, f"Unexpected image size for {experiment_id}"
        assert int(preset["batch"]) > 0, f"Invalid batch size for {experiment_id}"
        print(f"PRESET PASS: {experiment_id}")


def main() -> None:
    verify_configs()
    verify_manifest()
    verify_training_presets()
    print(
        "REPOSITORY VERIFICATION PASS: "
        f"{len(CONFIG_PATHS)} configs, {len(APPROVED_RESULTS)} experiments, {len(APPROVED_RESULTS)} presets"
    )


if __name__ == "__main__":
    main()
