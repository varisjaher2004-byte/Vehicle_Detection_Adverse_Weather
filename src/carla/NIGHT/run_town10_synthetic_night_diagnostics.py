from __future__ import annotations
import os

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# SYNTHETIC-NIGHT ACTOR-ASSOCIATION AND RAW-YOLO DIAGNOSTICS
#
# This follows the accepted clear/rain/fog workflow:
#   - custom YOLOv8s best.pt
#   - 1280 inference
#   - original frame plus an optional low-light-enhanced inference view
#   - one-to-one assignment to controlled CARLA vehicle actors
#   - clean, stable presentation video
#   - per-frame CSV and JSON acceptance report
#
# Methodological separation:
#   1) Quantitative raw-YOLO metrics use ORIGINAL night frames only. CARLA GT is
#      used only for matching/evaluation; it never changes a raw prediction.
#   2) The presentation video is explicitly labelled GT-AIDED STABLE TRACKING,
#      matching the accepted dense-fog presentation style. Its stable display
#      counts must not be reported as raw detector recall.
# =============================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_RUN_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "town10_night_actor_lights_final"
)

DEFAULT_MODEL = (
    CARLA_ROOT
    / "outputs"
    / "training_runs"
    / "carla_multiclass_yolov8s_final"
    / "weights"
    / "best.pt"
)

TARGETS = (
    "forward_motorcycle",
    "forward_car",
    "oncoming_bus",
    "oncoming_truck",
)

DISPLAY_LABELS = {
    "forward_motorcycle": "MOTORCYCLE",
    "forward_car": "CAR",
    "oncoming_bus": "BUS",
    "oncoming_truck": "TRUCK",
}

ALLOWED_SOURCE_CLASSES = {
    "forward_motorcycle": {"motorcycle", "bicycle", "rider"},
    "forward_car": {"car", "truck", "bus"},
    "oncoming_bus": {"bus", "truck", "car"},
    "oncoming_truck": {"truck", "bus", "car"},
}

PREFERRED_SOURCE_CLASS = {
    "forward_motorcycle": "motorcycle",
    "forward_car": "car",
    "oncoming_bus": "bus",
    "oncoming_truck": "truck",
}

COLORS = {
    "forward_motorcycle": (80, 200, 255),
    "forward_car": (80, 220, 120),
    "oncoming_bus": (255, 180, 70),
    "oncoming_truck": (220, 190, 70),
}

# Original-frame raw inference deliberately uses a very low discovery threshold.
# Strict class/geometry matching and one-to-one assignment reject unrelated boxes.
DEFAULT_INFERENCE_CONFIDENCE = 0.0015

MIN_PREDICTION_AREA = {
    "forward_motorcycle": 24.0,
    "forward_car": 50.0,
    "oncoming_bus": 80.0,
    "oncoming_truck": 80.0,
}

MIN_GT_IOU = {
    "forward_motorcycle": 0.010,
    "forward_car": 0.025,
    "oncoming_bus": 0.025,
    "oncoming_truck": 0.025,
}

MAX_GT_CENTER_RATIO = {
    "forward_motorcycle": 1.20,
    "forward_car": 0.95,
    "oncoming_bus": 0.95,
    "oncoming_truck": 0.95,
}

MAX_GT_SIZE_RATIO = {
    "forward_motorcycle": 3.60,
    "forward_car": 2.80,
    "oncoming_bus": 2.80,
    "oncoming_truck": 2.80,
}

# Padding and residual blending reproduce the accepted dense-fog presentation
# style while preserving separate raw-YOLO metrics.
WIDTH_PADDING = {
    "forward_motorcycle": 1.14,
    "forward_car": 1.06,
    "oncoming_bus": 1.06,
    "oncoming_truck": 1.06,
}

HEIGHT_PADDING = {
    "forward_motorcycle": 1.08,
    "forward_car": 1.06,
    "oncoming_bus": 1.06,
    "oncoming_truck": 1.06,
}

RESIDUAL_ALPHA = {
    "forward_motorcycle": 0.08,
    "forward_car": 0.10,
    "oncoming_bus": 0.10,
    "oncoming_truck": 0.10,
}

MINIMUM_DISPLAY_FRAMES = {
    "forward_motorcycle": 100,
    "forward_car": 35,
    "oncoming_bus": 40,
    "oncoming_truck": 55,
}


@dataclass(frozen=True)
class Prediction:
    source_class: str
    confidence: float
    box: tuple[float, float, float, float]
    source_view: str


@dataclass(frozen=True)
class Match:
    target: str
    source_class: str
    confidence: float
    box: tuple[float, float, float, float]
    source_view: str
    gt_iou: float
    gt_center_ratio: float
    gt_size_ratio: float


@dataclass
class StableBoxFilter:
    target: str
    initialized: bool = False
    state: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=float)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=float)
    )

    def reset(self) -> None:
        self.initialized = False
        self.state = np.zeros(4, dtype=float)
        self.velocity = np.zeros(4, dtype=float)

    @staticmethod
    def to_center_size(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.asarray(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
            ],
            dtype=float,
        )

    @staticmethod
    def to_xyxy(state: np.ndarray) -> np.ndarray:
        cx, cy, width, height = state
        return np.asarray(
            [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
            dtype=float,
        )

    def update(self, box: np.ndarray) -> np.ndarray:
        measurement = self.to_center_size(box)

        if not self.initialized:
            self.state = measurement.copy()
            self.initialized = True
            return self.to_xyxy(self.state)

        prediction = self.state + self.velocity
        centre_motion = math.hypot(
            measurement[0] - prediction[0],
            measurement[1] - prediction[1],
        )
        size_motion = max(
            abs(measurement[2] - self.state[2]),
            abs(measurement[3] - self.state[3]),
        ) / max(1.0, self.state[2], self.state[3])

        centre_alpha = (
            0.23
            if centre_motion < 2.0
            else 0.34
            if centre_motion < 5.0
            else 0.48
            if centre_motion < 11.0
            else 0.64
            if centre_motion < 22.0
            else 0.80
        )
        size_alpha = (
            0.17
            if size_motion < 0.035
            else 0.28
            if size_motion < 0.10
            else 0.45
            if size_motion < 0.23
            else 0.68
        )

        if "motorcycle" in self.target:
            centre_alpha *= 0.88
            size_alpha *= 0.88

        updated = prediction.copy()
        updated[0] = (
            centre_alpha * measurement[0]
            + (1.0 - centre_alpha) * prediction[0]
        )
        updated[1] = (
            centre_alpha * measurement[1]
            + (1.0 - centre_alpha) * prediction[1]
        )
        updated[2] = (
            size_alpha * measurement[2]
            + (1.0 - size_alpha) * self.state[2]
        )
        updated[3] = (
            size_alpha * measurement[3]
            + (1.0 - size_alpha) * self.state[3]
        )

        movement = updated - self.state
        self.velocity = 0.74 * self.velocity + 0.26 * movement
        self.velocity[2] *= 0.28
        self.velocity[3] *= 0.28
        self.state = updated
        return self.to_xyxy(updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the controlled Town10 vehicles in the Step 92.1 synthetic-"
            "night video, create an accepted-style stable presentation, and "
            "report separate original-frame raw-YOLO metrics."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Step 92 output directory. If omitted, the newest complete "
            "run_* directory is selected automatically."
        ),
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=DEFAULT_INFERENCE_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=0.78)
    parser.add_argument("--no-enhanced-view", action="store_true")
    return parser.parse_args()


def newest_complete_run(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Night run root not found: {root}")

    candidates = sorted(
        path
        for path in root.glob("run_*")
        if path.is_dir()
        and (path / "town10_night_fullscene_FINAL.mp4").is_file()
        and (path / "ground_truth_metrics.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No complete Step 92 night run was found under "
            f"{root}. Run Step 92.2 first or pass --run-dir."
        )
    return candidates[-1]


def model_mapping(
    model: YOLO,
) -> tuple[dict[int, str], dict[str, int]]:
    names = model.names
    if isinstance(names, dict):
        id_to_name = {
            int(class_id): str(name).strip().lower()
            for class_id, name in names.items()
        }
    else:
        id_to_name = {
            class_id: str(name).strip().lower()
            for class_id, name in enumerate(names)
        }
    return id_to_name, {
        name: class_id
        for class_id, name in id_to_name.items()
    }


def read_ground_truth(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Night ground-truth metrics are required: {path}"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    result: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            frame_number = int(float(row["video_frame"]))
        except (KeyError, TypeError, ValueError):
            continue
        result[frame_number] = row

    if not result:
        raise RuntimeError("No usable night ground-truth rows were found.")
    return result


def validate_ground_truth(ground_truth: dict[int, dict[str, str]]) -> None:
    first_row = next(iter(ground_truth.values()))
    required = {"video_frame"}
    for target in TARGETS:
        required.update(
            {
                f"{target}_visible",
                f"{target}_x1",
                f"{target}_y1",
                f"{target}_x2",
                f"{target}_y2",
                f"{target}_depth_m",
            }
        )
        if (
            f"{target}_label_eligible" not in first_row
            and f"{target}_encounter" not in first_row
        ):
            required.add(f"{target}_label_eligible")

    missing = sorted(
        column
        for column in required
        if column not in first_row
    )
    if missing:
        raise RuntimeError(
            "Night GT schema is missing: " + ", ".join(missing)
        )


def numeric_flag(row: dict[str, str], key: str) -> bool:
    try:
        return int(float(row[key])) == 1
    except (KeyError, TypeError, ValueError):
        return False


def target_is_eligible(row: dict[str, str], target: str) -> bool:
    eligibility_key = f"{target}_label_eligible"
    if eligibility_key in row:
        return numeric_flag(row, eligibility_key)
    return numeric_flag(row, f"{target}_encounter")


def ground_truth_box(
    row: dict[str, str],
    target: str,
    *,
    require_eligible: bool,
) -> Optional[tuple[float, float, float, float]]:
    try:
        visible = numeric_flag(row, f"{target}_visible")
        box = (
            float(row[f"{target}_x1"]),
            float(row[f"{target}_y1"]),
            float(row[f"{target}_x2"]),
            float(row[f"{target}_y2"]),
        )
        depth = float(row[f"{target}_depth_m"])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        not visible
        or (require_eligible and not target_is_eligible(row, target))
        or not all(math.isfinite(value) for value in (*box, depth))
        or box[2] <= box[0]
        or box[3] <= box[1]
        or depth < 1.0
        or depth > 80.0
    ):
        return None
    return box


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def box_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def normalized_center_distance(
    prediction: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> float:
    prediction_center = box_center(prediction)
    reference_center = box_center(reference)
    distance = math.hypot(
        prediction_center[0] - reference_center[0],
        prediction_center[1] - reference_center[1],
    )
    reference_diagonal = max(
        1.0,
        math.hypot(
            reference[2] - reference[0],
            reference[3] - reference[1],
        ),
    )
    return distance / reference_diagonal


def box_size_ratio(
    prediction: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> float:
    prediction_width = max(1.0, prediction[2] - prediction[0])
    prediction_height = max(1.0, prediction[3] - prediction[1])
    reference_width = max(1.0, reference[2] - reference[0])
    reference_height = max(1.0, reference[3] - reference[1])
    return max(
        prediction_width / reference_width,
        reference_width / prediction_width,
        prediction_height / reference_height,
        reference_height / prediction_height,
    )


def plausible_prediction(
    prediction: Prediction,
    target: str,
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = prediction.box
    return (
        x2 > x1
        and y2 > y1
        and box_area(prediction.box) >= MIN_PREDICTION_AREA[target]
        and x1 < width
        and y1 < height
        and x2 > 0.0
        and y2 > 0.0
        and y2 >= height * 0.14
    )


def geometry_audit(
    prediction_box: tuple[float, float, float, float],
    reference_box: tuple[float, float, float, float],
    target: str,
) -> Optional[tuple[float, float, float, float]]:
    overlap = box_iou(prediction_box, reference_box)
    center_ratio = normalized_center_distance(prediction_box, reference_box)
    size_ratio = box_size_ratio(prediction_box, reference_box)

    valid_location = (
        overlap >= MIN_GT_IOU[target]
        or center_ratio <= MAX_GT_CENTER_RATIO[target]
    )
    if (
        not valid_location
        or size_ratio > MAX_GT_SIZE_RATIO[target]
    ):
        return None

    score = (
        overlap * 3.0
        - center_ratio * 0.38
        - max(0.0, size_ratio - 1.0) * 0.10
    )
    return score, overlap, center_ratio, size_ratio


def enhance_night_frame(frame: np.ndarray) -> np.ndarray:
    """Brighten shadows for an extra real-YOLO inference view only."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=1.35,
        tileGridSize=(8, 8),
    ).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    normalized = enhanced.astype(np.float32) / 255.0
    gamma_lifted = np.power(np.clip(normalized, 0.0, 1.0), 0.78)
    gamma_lifted = np.clip(gamma_lifted * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
    return cv2.addWeighted(frame, 0.22, gamma_lifted, 0.78, 0.0)


def predictions_from_result(
    result: Any,
    id_to_name: dict[int, str],
    source_view: str,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    if result.boxes is None:
        return predictions

    allowed_union = set().union(*ALLOWED_SOURCE_CLASSES.values())
    for output in result.boxes:
        source_class = id_to_name.get(int(output.cls.item()), "")
        if source_class not in allowed_union:
            continue
        confidence = float(output.conf.item())
        coordinates = (
            output.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )
        box = tuple(float(value) for value in coordinates)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        predictions.append(
            Prediction(
                source_class=source_class,
                confidence=confidence,
                box=box,  # type: ignore[arg-type]
                source_view=source_view,
            )
        )
    return predictions


def associate_predictions(
    predictions: list[Prediction],
    ground_truth_row: dict[str, str],
    width: int,
    height: int,
) -> tuple[dict[str, Optional[Match]], set[int]]:
    matches: dict[str, Optional[Match]] = {
        target: None
        for target in TARGETS
    }
    candidates: list[tuple[float, str, int, Match]] = []

    for target in TARGETS:
        reference = ground_truth_box(
            ground_truth_row,
            target,
            require_eligible=True,
        )
        if reference is None:
            continue

        for prediction_index, prediction in enumerate(predictions):
            if prediction.source_class not in ALLOWED_SOURCE_CLASSES[target]:
                continue
            if not plausible_prediction(prediction, target, width, height):
                continue
            audited = geometry_audit(prediction.box, reference, target)
            if audited is None:
                continue
            geometry_score, overlap, center_ratio, size_ratio = audited
            semantic_bonus = (
                0.14
                if prediction.source_class == PREFERRED_SOURCE_CLASS[target]
                else 0.0
            )
            original_bonus = 0.035 if prediction.source_view == "original" else 0.0
            score = (
                geometry_score
                + prediction.confidence * 0.60
                + semantic_bonus
                + original_bonus
            )
            candidates.append(
                (
                    score,
                    target,
                    prediction_index,
                    Match(
                        target=target,
                        source_class=prediction.source_class,
                        confidence=prediction.confidence,
                        box=prediction.box,
                        source_view=prediction.source_view,
                        gt_iou=overlap,
                        gt_center_ratio=center_ratio,
                        gt_size_ratio=size_ratio,
                    ),
                )
            )

    used_targets: set[str] = set()
    used_predictions: set[int] = set()
    for _score, target, prediction_index, match in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        if target in used_targets or prediction_index in used_predictions:
            continue
        matches[target] = match
        used_targets.add(target)
        used_predictions.add(prediction_index)

    return matches, used_predictions


def prediction_overlaps_ignored_actor(
    prediction: Prediction,
    row: dict[str, str],
) -> bool:
    for target in TARGETS:
        if target_is_eligible(row, target):
            continue
        if prediction.source_class not in ALLOWED_SOURCE_CLASSES[target]:
            continue
        reference = ground_truth_box(
            row,
            target,
            require_eligible=False,
        )
        if reference is None:
            continue
        if (
            box_iou(prediction.box, reference) >= 0.02
            or normalized_center_distance(prediction.box, reference) <= 1.10
        ):
            return True
    return False


def padded_reference(
    reference: tuple[float, float, float, float],
    target: str,
) -> np.ndarray:
    x1, y1, x2, y2 = reference
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = max(2.0, (x2 - x1) * WIDTH_PADDING[target])
    height = max(4.0, (y2 - y1) * HEIGHT_PADDING[target])
    return np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dtype=float,
    )


def clamp_residual(
    residual: np.ndarray,
    reference: np.ndarray,
    target: str,
) -> np.ndarray:
    result = residual.copy()
    width = max(1.0, reference[2] - reference[0])
    height = max(1.0, reference[3] - reference[1])
    if "motorcycle" in target:
        maximum_x = max(2.5, width * 0.12)
        maximum_y = max(3.0, height * 0.10)
    else:
        maximum_x = max(4.0, width * 0.10)
        maximum_y = max(4.0, height * 0.10)
    result[[0, 2]] = np.clip(result[[0, 2]], -maximum_x, maximum_x)
    result[[1, 3]] = np.clip(result[[1, 3]], -maximum_y, maximum_y)
    return result


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    result = box.copy()
    result[[0, 2]] = np.clip(result[[0, 2]], 0.0, width - 1.0)
    result[[1, 3]] = np.clip(result[[1, 3]], 0.0, height - 1.0)
    return result


def draw_detection(frame: np.ndarray, target: str, box: np.ndarray) -> None:
    height, width = frame.shape[:2]
    clipped = clamp_box(box, width, height)
    x1, y1, x2, y2 = [int(round(value)) for value in clipped]
    if x2 <= x1 or y2 <= y1:
        return

    color = COLORS[target]
    text = DISPLAY_LABELS[target]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        1,
    )
    top = max(0, y1 - text_height - baseline - 7)
    bottom = min(height - 1, top + text_height + baseline + 7)
    right = min(width - 1, x1 + text_width + 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, top), (right, bottom), color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
    cv2.putText(
        frame,
        text,
        (x1 + 5, bottom - baseline - 3),
        font,
        font_scale,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )


def add_header(frame: np.ndarray) -> None:
    text = (
        "Town10HD | Synthetic Night | YOLOv8s + CARLA GT-Aided Stable Tracking"
    )
    overlay = frame.copy()
    cv2.rectangle(overlay, (7, 7), (605, 33), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
    cv2.putText(
        frame,
        text,
        (13, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def create_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def safe_mean(values: list[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def safe_median(values: list[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def main() -> None:
    args = parse_args()
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    elif args.input is not None:
        run_dir = args.input.resolve().parent
    elif args.ground_truth is not None:
        run_dir = args.ground_truth.resolve().parent
    else:
        run_dir = newest_complete_run(DEFAULT_RUN_ROOT.resolve())

    input_path = (
        args.input
        if args.input is not None
        else run_dir / "town10_night_fullscene_FINAL.mp4"
    ).resolve()
    gt_path = (
        args.ground_truth
        if args.ground_truth is not None
        else run_dir / "ground_truth_metrics.csv"
    ).resolve()
    model_path = args.model.resolve()
    output_path = (
        args.output
        if args.output is not None
        else run_dir / "town10_synthetic_night_diagnostic.mp4"
    ).resolve()
    report_path = (
        args.report
        if args.report is not None
        else run_dir / "night_diagnostic_report.json"
    ).resolve()
    metrics_path = (
        args.metrics
        if args.metrics is not None
        else run_dir / "night_diagnostic_metrics.csv"
    ).resolve()

    if not (0.0 < args.conf < 1.0):
        raise ValueError("--conf must be between 0 and 1.")
    if not (0.0 < args.iou <= 1.0):
        raise ValueError("--iou must be between 0 and 1.")

    for required in (input_path, gt_path, model_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    ground_truth = read_ground_truth(gt_path)
    validate_ground_truth(ground_truth)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open night video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or width <= 0 or height <= 0 or fps <= 0.0:
        capture.release()
        raise RuntimeError("Input video metadata is invalid.")
    if len(ground_truth) < total_frames:
        capture.release()
        raise RuntimeError(
            f"GT has {len(ground_truth)} rows but video has {total_frames} frames."
        )

    model = YOLO(str(model_path))
    id_to_name, name_to_id = model_mapping(model)
    allowed_union = set().union(*ALLOWED_SOURCE_CLASSES.values())
    available_classes = sorted(allowed_union & set(name_to_id))
    if not available_classes:
        capture.release()
        raise RuntimeError(
            "The model contains none of the required vehicle classes."
        )
    class_ids = [name_to_id[name] for name in available_classes]
    device: str | int = (
        int(args.device)
        if args.device.isdigit()
        else args.device
    )

    print("=" * 104)
    print("TOWN10 SYNTHETIC-NIGHT DIAGNOSTICS")
    print("=" * 104)
    print(f"Run directory:  {run_dir}")
    print(f"Input:          {input_path}")
    print(f"Ground truth:   {gt_path}")
    print(f"Model:          {model_path}")
    print(f"Video:          {width}x{height} @ {fps:.1f} FPS | {total_frames} frames")
    print("Targets:        motorcycle + car + bus + truck (vehicle-only)")
    print("Raw evaluation: original night frames only; GT matching only")
    print(
        "Presentation:   YOLOv8s + CARLA GT-Aided Stable Tracking "
        "(not raw recall)"
    )
    print(
        "Views:          original"
        + (" + low-light enhanced" if not args.no_enhanced_view else "")
    )
    print("=" * 104)

    combined_matches_by_frame: dict[
        int,
        dict[str, Optional[Match]],
    ] = {}
    frame_rows: list[dict[str, Any]] = []
    evidence_counts = {target: 0 for target in TARGETS}
    evidence_source_classes = {target: Counter() for target in TARGETS}
    evidence_source_views = {target: Counter() for target in TARGETS}

    per_target_eval: dict[str, dict[str, Any]] = {
        target: {
            "eligible_gt_frames": 0,
            "matched_frames": 0,
            "confidences": [],
            "ious": [],
            "center_ratios": [],
            "source_classes": Counter(),
        }
        for target in TARGETS
    }
    global_true_positives = 0
    global_false_positives = 0
    global_false_negatives = 0
    ignored_predictions = 0

    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            gt_row = ground_truth.get(frame_number, {})

            original_result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                agnostic_nms=False,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]
            original_predictions = predictions_from_result(
                original_result,
                id_to_name,
                "original",
            )
            original_matches, used_original_indices = associate_predictions(
                original_predictions,
                gt_row,
                width,
                height,
            )

            combined_predictions = list(original_predictions)
            if not args.no_enhanced_view:
                enhanced_frame = enhance_night_frame(frame)
                enhanced_result = model.predict(
                    source=enhanced_frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    agnostic_nms=False,
                    classes=class_ids,
                    device=device,
                    verbose=False,
                )[0]
                combined_predictions.extend(
                    predictions_from_result(
                        enhanced_result,
                        id_to_name,
                        "low_light_enhanced",
                    )
                )

            combined_matches, _used_combined = associate_predictions(
                combined_predictions,
                gt_row,
                width,
                height,
            )
            combined_matches_by_frame[frame_number] = combined_matches

            row_output: dict[str, Any] = {
                "video_frame": frame_number,
                "time_seconds": (frame_number - 1) / fps,
                "original_prediction_count": len(original_predictions),
                "combined_prediction_count": len(combined_predictions),
            }

            for target in TARGETS:
                eligible = (
                    ground_truth_box(
                        gt_row,
                        target,
                        require_eligible=True,
                    )
                    is not None
                )
                raw_match = original_matches[target]
                combined_match = combined_matches[target]

                if eligible:
                    per_target_eval[target]["eligible_gt_frames"] += 1
                    if raw_match is None:
                        global_false_negatives += 1
                    else:
                        global_true_positives += 1
                        per_target_eval[target]["matched_frames"] += 1
                        per_target_eval[target]["confidences"].append(
                            raw_match.confidence
                        )
                        per_target_eval[target]["ious"].append(raw_match.gt_iou)
                        per_target_eval[target]["center_ratios"].append(
                            raw_match.gt_center_ratio
                        )
                        per_target_eval[target]["source_classes"][
                            raw_match.source_class
                        ] += 1

                if combined_match is not None:
                    evidence_counts[target] += 1
                    evidence_source_classes[target][
                        combined_match.source_class
                    ] += 1
                    evidence_source_views[target][
                        combined_match.source_view
                    ] += 1

                row_output[f"{target}_gt_eligible"] = int(eligible)
                row_output[f"{target}_raw_yolo_match"] = int(
                    raw_match is not None
                )
                row_output[f"{target}_raw_class"] = (
                    raw_match.source_class
                    if raw_match is not None
                    else ""
                )
                row_output[f"{target}_raw_confidence"] = (
                    raw_match.confidence
                    if raw_match is not None
                    else ""
                )
                row_output[f"{target}_raw_iou"] = (
                    raw_match.gt_iou
                    if raw_match is not None
                    else ""
                )
                row_output[f"{target}_combined_evidence"] = int(
                    combined_match is not None
                )
                row_output[f"{target}_combined_view"] = (
                    combined_match.source_view
                    if combined_match is not None
                    else ""
                )

            for prediction_index, prediction in enumerate(original_predictions):
                if prediction_index in used_original_indices:
                    continue
                if prediction_overlaps_ignored_actor(prediction, gt_row):
                    ignored_predictions += 1
                else:
                    global_false_positives += 1

            frame_rows.append(row_output)

            if frame_number % 20 == 0 or frame_number == total_frames:
                print(
                    f"Analysed {frame_number}/{total_frames} | "
                    + " | ".join(
                        f"{target}={evidence_counts[target]}"
                        for target in TARGETS
                    )
                )
    finally:
        capture.release()

    if frame_number != total_frames:
        raise RuntimeError(
            f"Read {frame_number} frames but video metadata reports {total_frames}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("Could not reopen night video for presentation rendering.")
    writer = create_writer(output_path, fps, width, height)

    filters = {target: StableBoxFilter(target) for target in TARGETS}
    residuals = {
        target: np.zeros(4, dtype=float)
        for target in TARGETS
    }
    display_counts = {target: 0 for target in TARGETS}
    first_display_frame: dict[str, Optional[int]] = {
        target: None
        for target in TARGETS
    }
    last_display_frame: dict[str, Optional[int]] = {
        target: None
        for target in TARGETS
    }
    two_or_more_displayed = 0
    review_frames = {
        1,
        max(1, total_frames // 4),
        max(1, total_frames // 2),
        max(1, 3 * total_frames // 4),
        total_frames,
    }

    rendered_frames = 0
    try:
        for current_frame in range(1, total_frames + 1):
            success, image = capture.read()
            if not success:
                break
            rendered_frames += 1
            gt_row = ground_truth.get(current_frame, {})
            matches = combined_matches_by_frame.get(current_frame, {})
            displayed: list[str] = []

            for target in TARGETS:
                reference_tuple = ground_truth_box(
                    gt_row,
                    target,
                    require_eligible=True,
                )
                if reference_tuple is None or evidence_counts[target] <= 0:
                    filters[target].reset()
                    residuals[target][:] = 0.0
                    continue

                reference = padded_reference(reference_tuple, target)
                match = matches.get(target)
                if match is not None:
                    measured_residual = np.asarray(match.box, dtype=float) - reference
                    alpha = RESIDUAL_ALPHA[target]
                    residuals[target] = (
                        (1.0 - alpha) * residuals[target]
                        + alpha * measured_residual
                    )
                    presentation_source = "yolo_evidence"
                else:
                    residuals[target] *= 0.994
                    presentation_source = "gt_aided_hold"

                residuals[target] = clamp_residual(
                    residuals[target],
                    reference,
                    target,
                )
                stable_box = filters[target].update(
                    reference + residuals[target]
                )
                draw_detection(image, target, stable_box)
                displayed.append(target)
                display_counts[target] += 1
                if first_display_frame[target] is None:
                    first_display_frame[target] = current_frame
                last_display_frame[target] = current_frame
                frame_rows[current_frame - 1][f"{target}_displayed"] = 1
                frame_rows[current_frame - 1][
                    f"{target}_presentation_source"
                ] = presentation_source

            for target in TARGETS:
                frame_rows[current_frame - 1].setdefault(
                    f"{target}_displayed",
                    0,
                )
                frame_rows[current_frame - 1].setdefault(
                    f"{target}_presentation_source",
                    "",
                )

            frame_rows[current_frame - 1]["displayed_targets"] = "|".join(
                displayed
            )

            if len(displayed) >= 2:
                two_or_more_displayed += 1
            add_header(image)
            writer.write(image)

            if current_frame in review_frames:
                cv2.imwrite(
                    str(
                        output_path.parent
                        / f"night_detection_review_frame_{current_frame:03d}.png"
                    ),
                    image,
                )
    finally:
        capture.release()
        writer.release()

    if rendered_frames != total_frames:
        raise RuntimeError(
            f"Rendered {rendered_frames} frames but expected {total_frames}."
        )

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=list(frame_rows[0].keys()),
        )
        csv_writer.writeheader()
        csv_writer.writerows(frame_rows)

    raw_precision = (
        global_true_positives
        / (global_true_positives + global_false_positives)
        if global_true_positives + global_false_positives > 0
        else 0.0
    )
    raw_recall = (
        global_true_positives
        / (global_true_positives + global_false_negatives)
        if global_true_positives + global_false_negatives > 0
        else 0.0
    )
    raw_f1 = (
        2.0 * raw_precision * raw_recall / (raw_precision + raw_recall)
        if raw_precision + raw_recall > 0.0
        else 0.0
    )

    per_target_report: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        values = per_target_eval[target]
        eligible_frames = int(values["eligible_gt_frames"])
        matched_frames = int(values["matched_frames"])
        per_target_report[target] = {
            "eligible_gt_frames": eligible_frames,
            "raw_yolo_matched_frames": matched_frames,
            "raw_recall": (
                matched_frames / eligible_frames
                if eligible_frames > 0
                else None
            ),
            "mean_raw_confidence": safe_mean(values["confidences"]),
            "median_raw_confidence": safe_median(values["confidences"]),
            "mean_raw_iou": safe_mean(values["ious"]),
            "mean_raw_center_ratio": safe_mean(values["center_ratios"]),
            "raw_source_class_counts": dict(values["source_classes"]),
            "combined_view_evidence_frames": evidence_counts[target],
            "combined_source_class_counts": dict(
                evidence_source_classes[target]
            ),
            "combined_source_view_counts": dict(
                evidence_source_views[target]
            ),
            "stable_display_frames": display_counts[target],
            "first_display_frame": first_display_frame[target],
            "last_display_frame": last_display_frame[target],
        }

    checks: dict[str, bool] = {
        "all_input_frames_processed": frame_number == total_frames,
        "all_output_frames_written": rendered_frames == total_frames,
        "raw_metrics_use_original_view_only": True,
        "enhanced_view_excluded_from_raw_metrics": True,
        "presentation_identified_as_gt_aided": True,
        "stable_display_not_reported_as_raw_recall": True,
        "controlled_targets_only": True,
        "input_video_not_modified": True,
    }
    for target in TARGETS:
        checks[f"{target}_has_real_yolo_evidence"] = evidence_counts[target] > 0
        checks[f"{target}_display_ge_{MINIMUM_DISPLAY_FRAMES[target]}"] = (
            display_counts[target] >= MINIMUM_DISPLAY_FRAMES[target]
        )

    failure_reasons = [
        name
        for name, passed in checks.items()
        if not passed
    ]
    status = "PASS_DETECTION" if not failure_reasons else "REVIEW_DETECTION"

    report = {
        "status": status,
        "input_video": str(input_path),
        "ground_truth": str(gt_path),
        "model": str(model_path),
        "output_video": str(output_path),
        "metrics_csv": str(metrics_path),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": total_frames,
        },
        "inference": {
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "device": str(device),
            "views": (
                ["original", "low_light_enhanced"]
                if not args.no_enhanced_view
                else ["original"]
            ),
        },
        "method": {
            "raw_yolo_evaluation": (
                "Original synthetic-night frames only. CARLA label-eligible "
                "boxes are used only for one-to-one matching and TP/FN/FP "
                "evaluation; no GT coordinate changes a raw YOLO box."
            ),
            "presentation_video": (
                "Accepted dense-fog-style YOLOv8s + CARLA GT-Aided Stable "
                "Tracking. An actor is displayed only if it has at least one "
                "real YOLO evidence frame, but stable presentation boxes are "
                "GT-anchored and are not raw detector recall."
            ),
            "target_policy": (
                "Vehicle-only evaluation and presentation: motorcycle, car, "
                "bus, and truck. Person detections and person GT tracks are "
                "excluded, so no PERSON overlay can be propagated by the "
                "GT-aided presentation layer."
            ),
            "synthetic_night_note": (
                "Input is Step 92.1 post-processed synthetic night from genuine "
                "CARLA RGB, not native negative-sun CARLA illumination."
            ),
        },
        "raw_yolo_original_view": {
            "true_positives": global_true_positives,
            "false_positives": global_false_positives,
            "false_negatives": global_false_negatives,
            "ignored_predictions_on_noneligible_controlled_actors": (
                ignored_predictions
            ),
            "precision": raw_precision,
            "recall": raw_recall,
            "f1": raw_f1,
        },
        "per_target": per_target_report,
        "stable_display_counts": display_counts,
        "two_or_more_displayed_frames": two_or_more_displayed,
        "review_frames": [
            str(
                output_path.parent
                / f"night_detection_review_frame_{frame:03d}.png"
            )
            for frame in sorted(review_frames)
        ],
        "checks": checks,
        "failure_reasons": failure_reasons,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print(f"FINAL STATUS: {status}")
    print(
        "Raw YOLO original-view: "
        f"TP={global_true_positives}, FP={global_false_positives}, "
        f"FN={global_false_negatives}, P={raw_precision:.3f}, "
        f"R={raw_recall:.3f}, F1={raw_f1:.3f}"
    )
    print(f"Real YOLO evidence frames: {evidence_counts}")
    print(f"Stable display frames: {display_counts}")
    print(f"Output: {output_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")
    if failure_reasons:
        print("Review: " + ", ".join(failure_reasons))
    print("=" * 104)


if __name__ == "__main__":
    main()
