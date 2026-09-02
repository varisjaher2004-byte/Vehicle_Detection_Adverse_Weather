"""Verify portable dataset configs and the approved real-world evidence manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

try:
    from .verify_corrected_evidence import main as verify_corrected_evidence
except ImportError:  # Direct script execution from the repository root.
    from verify_corrected_evidence import main as verify_corrected_evidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "EVIDENCE_MANIFEST.csv"
PRESETS_PATH = REPOSITORY_ROOT / "configs" / "TRAINING_PRESETS.json"
CARLA_ROUTE_PATH = REPOSITORY_ROOT / "configs" / "CARLA" / "locked_town10_reference_route.json"
INFERENCE_INVENTORY_PATH = REPOSITORY_ROOT / "results" / "REAL_WORLD_INFERENCE_INVENTORY.csv"

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

FORMAL_NOTEBOOK_PATHS = {
    "notebooks/acdc_entire_training.ipynb",
    "notebooks/acdc_fog_training.ipynb",
    "notebooks/acdc_night_training.ipynb",
    "notebooks/acdc_rain_training.ipynb",
    "notebooks/dawn_entire_training.ipynb",
    "notebooks/dawn_fog_training.ipynb",
    "notebooks/dawn_rain_training.ipynb",
    "notebooks/combined_yolov8_training.ipynb",
    "notebooks/yolov8_inference_pipeline.ipynb",
}

FORMAL_CARLA_SCRIPT_PATHS = {
    "src/carla/CLEAR/lock_town10_reference_route.py",
    "src/carla/CLEAR/generate_town10_clear_raw_scene.py",
    "src/carla/CLEAR/run_town10_clear_detection.py",
    "src/carla/CLEAR/generate_town10_clear_stable_output.py",
    "src/carla/RAIN/generate_town10_rain_scene.py",
    "src/carla/RAIN/run_town10_rain_stable_detection.py",
    "src/carla/RAIN/generate_town10_rain_presentation_output.py",
    "src/carla/FOG/generate_town10_fog_scene.py",
    "src/carla/FOG/generate_town10_fog_presentation_output.py",
    "src/carla/NIGHT/generate_town10_synthetic_night_scene.py",
    "src/carla/NIGHT/run_town10_synthetic_night_diagnostics.py",
}

FORMAL_CARLA_RESULT_PATHS = {
    "results/CARLA/FOG/fog_presentation_metrics.csv",
    "results/CARLA/FOG/fog_presentation_report.json",
    "results/CARLA/FOG/fog_review_frame_090.png",
    "results/CARLA/RAIN/rain_presentation_metrics.csv",
    "results/CARLA/RAIN/rain_presentation_report.json",
    "results/CARLA/RAIN/rain_review_frame_200.png",
    "results/CARLA/NIGHT/night_diagnostic_metrics.csv",
    "results/CARLA/NIGHT/night_diagnostic_report.json",
    "results/CARLA/NIGHT/night_detection_review_frame_070.png",
    "results/CARLA/NIGHT/night_annotated_review_frame_070.png",
}

FORMAL_TRAINING_PLOT_HASHES = {
    "results/ACDC/FOG/BoxP_curve.png": "3891fa544bf0f8148c219d8b70509715664a907f23a225a83fca2a6f54b22eac",
    "results/ACDC/FOG/BoxR_curve.png": "f5762cc3f216e2ab80280b40d65cba31c27eb30b3c4bdb68f03a0db4dc01b5e0",
    "results/DAWN/ENTIRE/BoxP_curve.png": "d206d2a1700ca26054d03a1671d5bdc1d4059db60570baabf7b166f5d9aa928b",
    "results/DAWN/ENTIRE/BoxR_curve.png": "8bb40b04329a429c9606cbdf996c32072630a910271594e596ede25763eac0de",
}

LEGACY_FILENAME_MARKERS = (
    "FINAL_PERFECT_STABLE",
    "FINAL_STABLE",
    "perfect_stable",
    "fog_perfect",
)

INFERENCE_CSV_PATHS = {
    "results/ACDC/ENTIRE/inference_results.csv",
    "results/ACDC/FOG/inference_results.csv",
    "results/ACDC/NIGHT/inference_results.csv",
    "results/ACDC/RAIN/inference_results.csv",
    "results/ACDC/SNOW/inference_results.csv",
    "results/DAWN/ENTIRE/inference_results.csv",
    "results/COMBINED/inference_results.csv",
}

INCLUDED_INFERENCE_STATUS = "included_in_repository"
WITHHELD_ACDC_STATUS = "withheld_acdc_redistribution_restriction"


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


def verify_formal_filenames() -> None:
    expected_paths = (
        FORMAL_NOTEBOOK_PATHS
        | FORMAL_CARLA_SCRIPT_PATHS
        | FORMAL_CARLA_RESULT_PATHS
        | set(FORMAL_TRAINING_PLOT_HASHES)
    )
    for relative_path in expected_paths:
        assert (REPOSITORY_ROOT / relative_path).is_file(), (
            f"Missing formally named artefact: {relative_path}"
        )

    for relative_root in ("notebooks", "src/carla", "results/CARLA"):
        for path in (REPOSITORY_ROOT / relative_root).rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            assert not any(marker in path.name for marker in LEGACY_FILENAME_MARKERS), (
                f"Informal legacy filename retained: {path.relative_to(REPOSITORY_ROOT)}"
            )

    for relative_path, expected_hash in FORMAL_TRAINING_PLOT_HASHES.items():
        actual_hash = _sha256(REPOSITORY_ROOT / relative_path)
        assert actual_hash == expected_hash, f"Training-plot hash mismatch: {relative_path}"

    print(
        "NAMING PASS: "
        f"{len(FORMAL_NOTEBOOK_PATHS)} notebooks, "
        f"{len(FORMAL_CARLA_SCRIPT_PATHS)} CARLA scripts and "
        f"{len(FORMAL_CARLA_RESULT_PATHS)} CARLA evidence files; "
        f"TRAINING PLOT PASS: {len(FORMAL_TRAINING_PLOT_HASHES)} files"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_real_world_inference_inventory() -> None:
    rows = _read_csv(INFERENCE_INVENTORY_PATH)
    assert len(rows) == 1466, "Unexpected real-world inference inventory size"

    image_rows = [row for row in rows if row["artifact_type"] == "annotated_prediction"]
    log_rows = [row for row in rows if row["artifact_type"] == "detection_log"]
    included_rows = [row for row in rows if row["publication_status"] == INCLUDED_INFERENCE_STATUS]
    withheld_rows = [row for row in rows if row["publication_status"] == WITHHELD_ACDC_STATUS]
    included_images = [row for row in included_rows if row["artifact_type"] == "annotated_prediction"]

    assert len(image_rows) == 1459, "Unexpected saved prediction count"
    assert len(log_rows) == 7, "Unexpected inference-log count"
    assert len(included_images) == 240, "Unexpected published DAWN prediction count"
    assert len(included_rows) == 247, "Unexpected published inference artefact count"
    assert len(withheld_rows) == 1219, "Unexpected ACDC-withheld prediction count"
    assert {row["repository_path"] for row in log_rows} == INFERENCE_CSV_PATHS, (
        "Inference-log destinations differ from the approved set"
    )

    repository_paths = [row["repository_path"] for row in included_rows]
    assert len(repository_paths) == len(set(repository_paths)), "Duplicate published inference path"
    assert all(row["dataset_content"] == "DAWN" for row in included_images), (
        "Only DAWN image content may be published in the qualitative gallery"
    )
    assert all(row["dataset_content"] == "ACDC" for row in withheld_rows), (
        "Only restricted ACDC predictions should carry the withheld status"
    )
    assert all(not row["repository_path"] for row in withheld_rows), (
        "Withheld ACDC predictions must not have repository destinations"
    )
    assert all(not row["repository_sha256"] and not row["repository_bytes"] for row in withheld_rows), (
        "Withheld ACDC predictions must not have repository integrity fields"
    )

    for row in rows:
        source_path = Path(row["source_relative_path"])
        assert not source_path.is_absolute(), "Inventory must not expose machine-specific source paths"
        assert len(row["source_sha256"]) == 64 and all(
            char in "0123456789abcdef" for char in row["source_sha256"]
        ), (
            f"Invalid SHA-256 in inference inventory: {row['source_relative_path']}"
        )
        assert int(row["source_bytes"]) > 0, f"Invalid byte size: {row['source_relative_path']}"
        assert int(row["same_content_copies"]) >= 1, (
            f"Invalid content-copy count: {row['source_relative_path']}"
        )

    for row in included_rows:
        relative_path = row["repository_path"]
        artefact_path = REPOSITORY_ROOT / relative_path
        assert artefact_path.is_file(), f"Missing published inference artefact: {relative_path}"
        assert len(row["repository_sha256"]) == 64, f"Missing repository SHA-256: {relative_path}"
        assert artefact_path.stat().st_size == int(row["repository_bytes"]), f"Size mismatch: {relative_path}"
        assert _sha256(artefact_path) == row["repository_sha256"], f"SHA-256 mismatch: {relative_path}"

    assert all(
        row["source_sha256"] == row["repository_sha256"]
        and row["source_bytes"] == row["repository_bytes"]
        for row in included_images
    ), "Published prediction images must remain byte-identical to the saved source outputs"

    print(
        "INFERENCE EVIDENCE PASS: "
        "240 DAWN prediction images and 7 detection logs verified; "
        "1,219 ACDC-background predictions inventoried but withheld under the dataset licence"
    )


def verify_carla_route() -> None:
    with CARLA_ROUTE_PATH.open(encoding="utf-8") as handle:
        route = json.load(handle)
    assert route["status"] == "REFERENCE_ROUTE_LOCKED", "Unexpected CARLA route status"
    assert route["map"] == "Carla/Maps/Town10HD_Opt", "Unexpected CARLA map"
    assert route["path_base_env"] == "DMSC_GENERATED_DATASET_ROOT", "Missing portable CARLA path base"
    assert route["dataset_root"] == ".", "Active CARLA dataset root must be relative"
    assert all(not Path(item["folder"]).is_absolute() for item in route["weather_inventory"].values()), (
        "Absolute weather folder in active CARLA route"
    )
    assert all(not Path(item).is_absolute() for item in route["clear_reference_images"]), (
        "Absolute reference image in active CARLA route"
    )

    selected = route["selected"]
    assert selected["spawn_index"] == 36, "Unexpected CARLA spawn index"
    assert selected["road_id"] == 11, "Unexpected CARLA road ID"
    assert selected["lane_id"] == 1, "Unexpected CARLA lane ID"
    location = selected["transform"]["location"]
    rotation = selected["transform"]["rotation"]
    expected_location = {"x": 67.65974426269531, "y": 69.8227767944336, "z": 0.5999999642372131}
    expected_rotation = {"pitch": 0.0, "yaw": 0.07327299565076828, "roll": 0.0}
    for key, expected in expected_location.items():
        assert math.isclose(float(location[key]), expected, rel_tol=0.0, abs_tol=1e-12), (
            f"Unexpected CARLA location {key}"
        )
    for key, expected in expected_rotation.items():
        assert math.isclose(float(rotation[key]), expected, rel_tol=0.0, abs_tol=1e-12), (
            f"Unexpected CARLA rotation {key}"
        )
    print("CARLA ROUTE PASS: Town10HD_Opt spawn 36")


def main() -> None:
    verify_configs()
    verify_manifest()
    verify_training_presets()
    verify_formal_filenames()
    verify_real_world_inference_inventory()
    verify_carla_route()
    verify_corrected_evidence()
    print(
        "REPOSITORY VERIFICATION PASS: "
        f"{len(CONFIG_PATHS)} configs, {len(APPROVED_RESULTS)} experiments, "
        f"{len(APPROVED_RESULTS)} presets, {len(FORMAL_TRAINING_PLOT_HASHES)} training plots, "
        "1 inference inventory, "
        "1 CARLA route, 1 corrected evidence set"
    )


if __name__ == "__main__":
    main()
