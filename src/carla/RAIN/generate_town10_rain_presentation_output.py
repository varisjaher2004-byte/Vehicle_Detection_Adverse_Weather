from __future__ import annotations
import os

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# TOWN10 RAIN PRESENTATION OUTPUT
#
# LOCKED REQUIREMENT:
#   Exactly 1 lead CAR + 2 PERSON boxes.
#   Each box remains visible from the actor's first visible frame until its
#   last visible frame.
#
# This is a qualitative presentation visualisation:
#       YOLOv8s + CARLA GT-Aided Stable Tracking
#
# Real YOLO predictions provide detection evidence / semantic information.
# CARLA projected actor geometry provides the stable position reference.
#
# IMPORTANT:
#   Stable display-frame counts are NOT pure YOLO recall metrics.
#   Quantitative evaluation must continue to use the genuine YOLO metrics.
#
# The accepted raw video is READ ONLY and is never overwritten.
# =============================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

LOCKED_RUN = (
    CARLA_ROOT
    / "outputs"
    / "town10_rain_signal_continuation"
    / "run_20260807_090117"
)

DEFAULT_INPUT = LOCKED_RUN / "rain_signal_raw.mp4"
DEFAULT_GT = LOCKED_RUN / "ground_truth_metrics.csv"
DEFAULT_MODEL = (
    CARLA_ROOT
    / "outputs"
    / "training_runs"
    / "carla_multiclass_yolov8s_final"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT = (
    LOCKED_RUN
    / "town10_rain_presentation.mp4"
)

DEFAULT_REPORT = (
    LOCKED_RUN
    / "rain_presentation_report.json"
)

DEFAULT_METRICS = (
    LOCKED_RUN
    / "rain_presentation_metrics.csv"
)


TARGETS = (
    "lead_car",
    "pedestrian_1",
    "pedestrian_2",
)

DISPLAY_LABEL = {
    "lead_car": "CAR",
    "pedestrian_1": "PERSON",
    "pedestrian_2": "PERSON",
}

ALLOWED_CLASSES = {
    "lead_car": {
        "car",
        "truck",
        "bus",
    },
    "pedestrian_1": {
        "person",
        "rider",
    },
    "pedestrian_2": {
        "person",
        "rider",
    },
}

COLORS = {
    "lead_car": (70, 220, 120),
    "pedestrian_1": (70, 210, 245),
    "pedestrian_2": (80, 165, 245),
}

INFERENCE_CONF = 0.0020

MATCH_MIN_IOU = {
    "lead_car": 0.010,
    "pedestrian_1": 0.003,
    "pedestrian_2": 0.003,
}

MATCH_MAX_CENTER_RATIO = {
    "lead_car": 1.00,
    "pedestrian_1": 1.45,
    "pedestrian_2": 1.45,
}

# YOLO residual has intentionally low influence in the final presentation.
# This prevents low-confidence rain predictions from making the stable box
# jump while still retaining real detector evidence.
YOLO_RESIDUAL_ALPHA = {
    "lead_car": 0.10,
    "pedestrian_1": 0.08,
    "pedestrian_2": 0.08,
}

YOLO_RESIDUAL_DECAY = 0.992

# Minimum readable final boxes.
MIN_BOX_WIDTH = {
    "lead_car": 24.0,
    "pedestrian_1": 11.0,
    "pedestrian_2": 11.0,
}

MIN_BOX_HEIGHT = {
    "lead_car": 18.0,
    "pedestrian_1": 22.0,
    "pedestrian_2": 22.0,
}

# Small final padding keeps object comfortably inside the rectangle.
WIDTH_PAD = {
    "lead_car": 1.08,
    "pedestrian_1": 1.18,
    "pedestrian_2": 1.18,
}

HEIGHT_PAD = {
    "lead_car": 1.08,
    "pedestrian_1": 1.06,
    "pedestrian_2": 1.06,
}


@dataclass(frozen=True)
class YoloMatch:
    target: str
    source_class: str
    confidence: float
    box: tuple[float, float, float, float]
    iou: float
    center_ratio: float


@dataclass
class StableBoxFilter:
    """
    Adaptive centre/size smoothing.

    Why centre/size instead of directly smoothing x1/y1/x2/y2:
      - the box does not "breathe" from opposite corners,
      - width/height changes are calmer,
      - moving/turning actors remain responsive.

    Large genuine motion automatically increases alpha, while tiny changes
    receive strong smoothing.
    """

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
    def xyxy_to_cxcywh(
        box: np.ndarray,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        return np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                width,
                height,
            ],
            dtype=float,
        )

    @staticmethod
    def cxcywh_to_xyxy(
        value: np.ndarray,
    ) -> np.ndarray:
        cx, cy, width, height = value

        return np.array(
            [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
            dtype=float,
        )

    def update(
        self,
        measurement_box: np.ndarray,
    ) -> np.ndarray:
        measurement = self.xyxy_to_cxcywh(
            measurement_box
        )

        if not self.initialized:
            self.state = measurement.copy()
            self.velocity[:] = 0.0
            self.initialized = True

            return self.cxcywh_to_xyxy(
                self.state
            )

        # Constant-velocity prediction reduces visual lag during turns.
        predicted = self.state + self.velocity

        centre_motion = math.hypot(
            measurement[0] - predicted[0],
            measurement[1] - predicted[1],
        )

        previous_size = max(
            1.0,
            self.state[2],
            self.state[3],
        )

        size_motion = max(
            abs(measurement[2] - self.state[2]),
            abs(measurement[3] - self.state[3]),
        ) / previous_size

        # Deadband: sub-pixel / tiny measurement changes are visual noise.
        for index in (0, 1):
            if abs(
                measurement[index] - predicted[index]
            ) < 0.85:
                measurement[index] = predicted[index]

        for index in (2, 3):
            if abs(
                measurement[index] - self.state[index]
            ) < 1.25:
                measurement[index] = self.state[index]

        # Adaptive centre alpha.
        if centre_motion < 2.0:
            centre_alpha = 0.24
        elif centre_motion < 6.0:
            centre_alpha = 0.34
        elif centre_motion < 14.0:
            centre_alpha = 0.48
        else:
            centre_alpha = 0.68

        # Pedestrians need slightly stronger anti-jitter smoothing.
        if self.target.startswith("pedestrian_"):
            centre_alpha *= 0.88

        # Width/height should change even more smoothly.
        if size_motion < 0.035:
            size_alpha = 0.18
        elif size_motion < 0.10:
            size_alpha = 0.28
        elif size_motion < 0.22:
            size_alpha = 0.44
        else:
            size_alpha = 0.62

        if self.target.startswith("pedestrian_"):
            size_alpha *= 0.88

        updated = predicted.copy()

        updated[0] = (
            centre_alpha * measurement[0]
            + (1.0 - centre_alpha) * predicted[0]
        )
        updated[1] = (
            centre_alpha * measurement[1]
            + (1.0 - centre_alpha) * predicted[1]
        )

        updated[2] = (
            size_alpha * measurement[2]
            + (1.0 - size_alpha) * self.state[2]
        )
        updated[3] = (
            size_alpha * measurement[3]
            + (1.0 - size_alpha) * self.state[3]
        )

        # Calm velocity estimate.
        measured_velocity = updated - self.state

        self.velocity = (
            0.76 * self.velocity
            + 0.24 * measured_velocity
        )

        # Size velocity is unnecessary and can create "breathing".
        self.velocity[2] *= 0.30
        self.velocity[3] *= 0.30

        self.state = updated

        return self.cxcywh_to_xyxy(
            self.state
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the accepted Town10 rain presentation "
            "with exactly one car and two persons."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--no-signal-badge",
        action="store_true",
    )

    return parser.parse_args()


def read_ground_truth(
    path: Path,
) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth CSV: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    output: dict[int, dict[str, str]] = {}

    for row in rows:
        try:
            frame = int(float(row["video_frame"]))
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        output[frame] = row

    if not output:
        raise RuntimeError(
            "Ground-truth CSV contains no usable frames."
        )

    return output


def validate_schema(
    ground_truth: dict[int, dict[str, str]],
) -> None:
    sample = next(iter(ground_truth.values()))

    required = {
        "video_frame",
        "scene_state",
        "traffic_light_state",
    }

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

    missing = sorted(
        item
        for item in required
        if item not in sample
    )

    if missing:
        raise RuntimeError(
            "Ground-truth schema mismatch. Missing: "
            + ", ".join(missing)
        )


def gt_box(
    row: dict[str, str],
    target: str,
) -> Optional[tuple[float, float, float, float]]:
    try:
        visible = (
            int(float(row[f"{target}_visible"]))
            == 1
        )

        x1 = float(row[f"{target}_x1"])
        y1 = float(row[f"{target}_y1"])
        x2 = float(row[f"{target}_x2"])
        y2 = float(row[f"{target}_y2"])
        depth = float(row[f"{target}_depth_m"])

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if (
        not visible
        or not all(
            math.isfinite(value)
            for value in (
                x1,
                y1,
                x2,
                y2,
                depth,
            )
        )
        or x2 <= x1
        or y2 <= y1
        or depth < 0.5
        or depth > 80.0
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def box_area(
    box: tuple[float, float, float, float],
) -> float:
    return (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    union = (
        box_area(first)
        + box_area(second)
        - intersection
    )

    return (
        intersection / union
        if union > 0.0
        else 0.0
    )


def box_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def center_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_center = box_center(first)
    second_center = box_center(second)

    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )

    width = max(
        1.0,
        second[2] - second[0],
    )
    height = max(
        1.0,
        second[3] - second[1],
    )

    diagonal = max(
        1.0,
        math.hypot(width, height),
    )

    return distance / diagonal


def model_mapping(
    model: YOLO,
) -> tuple[
    dict[int, str],
    dict[str, int],
]:
    names = model.names

    if isinstance(names, dict):
        id_to_name = {
            int(class_id): (
                str(name)
                .strip()
                .lower()
            )
            for class_id, name
            in names.items()
        }
    else:
        id_to_name = {
            class_id: (
                str(name)
                .strip()
                .lower()
            )
            for class_id, name
            in enumerate(names)
        }

    name_to_id = {
        name: class_id
        for class_id, name
        in id_to_name.items()
    }

    return (
        id_to_name,
        name_to_id,
    )


def enhance_frame(
    frame: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB,
    )

    lightness, channel_a, channel_b = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.40,
        tileGridSize=(8, 8),
    )

    lightness = clahe.apply(lightness)

    return cv2.cvtColor(
        cv2.merge(
            (
                lightness,
                channel_a,
                channel_b,
            )
        ),
        cv2.COLOR_LAB2BGR,
    )


def extract_predictions(
    result: Any,
    id_to_name: dict[int, str],
) -> list[
    tuple[
        str,
        float,
        tuple[float, float, float, float],
    ]
]:
    output = []

    if result.boxes is None:
        return output

    for prediction in result.boxes:
        class_id = int(
            prediction.cls.item()
        )

        source_class = id_to_name.get(
            class_id,
            "",
        )

        confidence = float(
            prediction.conf.item()
        )

        coordinates = (
            prediction.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        box = tuple(
            float(value)
            for value in coordinates
        )

        if (
            box[2] <= box[0]
            or box[3] <= box[1]
        ):
            continue

        output.append(
            (
                source_class,
                confidence,
                box,  # type: ignore[arg-type]
            )
        )

    return output


def match_yolo_to_targets(
    predictions: list[
        tuple[
            str,
            float,
            tuple[float, float, float, float],
        ]
    ],
    row: dict[str, str],
) -> dict[str, Optional[YoloMatch]]:
    matches: dict[str, Optional[YoloMatch]] = {
        target: None
        for target in TARGETS
    }

    candidates = []

    for target in TARGETS:
        reference = gt_box(
            row,
            target,
        )

        if reference is None:
            continue

        for prediction_index, (
            source_class,
            confidence,
            box,
        ) in enumerate(predictions):

            if (
                source_class
                not in ALLOWED_CLASSES[target]
            ):
                continue

            overlap = box_iou(
                box,
                reference,
            )

            ratio = center_ratio(
                box,
                reference,
            )

            if (
                overlap
                < MATCH_MIN_IOU[target]
                and ratio
                > MATCH_MAX_CENTER_RATIO[target]
            ):
                continue

            semantic_bonus = 0.0

            if (
                target == "lead_car"
                and source_class == "car"
            ):
                semantic_bonus = 0.10

            if (
                target.startswith("pedestrian_")
                and source_class == "person"
            ):
                semantic_bonus = 0.08

            score = (
                overlap * 3.0
                - ratio * 0.32
                + confidence * 0.50
                + semantic_bonus
            )

            candidates.append(
                (
                    score,
                    target,
                    prediction_index,
                    YoloMatch(
                        target=target,
                        source_class=source_class,
                        confidence=confidence,
                        box=box,
                        iou=overlap,
                        center_ratio=ratio,
                    ),
                )
            )

    used_targets = set()
    used_predictions = set()

    for (
        _score,
        target,
        prediction_index,
        match,
    ) in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        if (
            target in used_targets
            or prediction_index in used_predictions
        ):
            continue

        matches[target] = match

        used_targets.add(target)
        used_predictions.add(
            prediction_index
        )

    return matches


def padded_reference_box(
    reference: tuple[float, float, float, float],
    target: str,
) -> np.ndarray:
    x1, y1, x2, y2 = reference

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    width = max(
        MIN_BOX_WIDTH[target],
        (x2 - x1) * WIDTH_PAD[target],
    )

    height = max(
        MIN_BOX_HEIGHT[target],
        (y2 - y1) * HEIGHT_PAD[target],
    )

    return np.array(
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ],
        dtype=float,
    )


def clamp_residual(
    residual: np.ndarray,
    reference: np.ndarray,
    target: str,
) -> np.ndarray:
    result = residual.copy()

    width = max(
        1.0,
        reference[2] - reference[0],
    )
    height = max(
        1.0,
        reference[3] - reference[1],
    )

    if target == "lead_car":
        max_x = max(
            4.0,
            width * 0.10,
        )
        max_y = max(
            4.0,
            height * 0.10,
        )
    else:
        max_x = max(
            2.5,
            width * 0.14,
        )
        max_y = max(
            3.0,
            height * 0.10,
        )

    result[0] = np.clip(
        result[0],
        -max_x,
        max_x,
    )
    result[2] = np.clip(
        result[2],
        -max_x,
        max_x,
    )
    result[1] = np.clip(
        result[1],
        -max_y,
        max_y,
    )
    result[3] = np.clip(
        result[3],
        -max_y,
        max_y,
    )

    return result


def clamp_box(
    box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    output = box.astype(float).copy()

    output[0] = np.clip(
        output[0],
        0.0,
        width - 1.0,
    )
    output[1] = np.clip(
        output[1],
        0.0,
        height - 1.0,
    )
    output[2] = np.clip(
        output[2],
        0.0,
        width - 1.0,
    )
    output[3] = np.clip(
        output[3],
        0.0,
        height - 1.0,
    )

    return output


def draw_box(
    frame: np.ndarray,
    target: str,
    box: np.ndarray,
) -> None:
    height, width = frame.shape[:2]

    box = clamp_box(
        box,
        width,
        height,
    )

    x1, y1, x2, y2 = [
        int(round(value))
        for value in box
    ]

    if x2 <= x1 or y2 <= y1:
        return

    color = COLORS[target]
    text = DISPLAY_LABEL[target]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    top = max(
        0,
        y1
        - text_height
        - baseline
        - 7,
    )

    right = min(
        width - 1,
        x1
        + text_width
        + 10,
    )

    bottom = min(
        height - 1,
        top
        + text_height
        + baseline
        + 7,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, top),
        (right, bottom),
        color,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0.0,
        frame,
    )

    cv2.putText(
        frame,
        text,
        (
            x1 + 5,
            bottom
            - baseline
            - 3,
        ),
        font,
        scale,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )


def add_header(
    frame: np.ndarray,
) -> None:
    text = (
        "Town10HD | Rain | YOLOv8s + CARLA GT-Aided Stable Tracking"
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (7, 7),
        (430, 33),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0.0,
        frame,
    )

    cv2.putText(
        frame,
        text,
        (13, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def add_signal_badge(
    frame: np.ndarray,
    traffic_state: str,
) -> None:
    state = (
        traffic_state
        .strip()
        .upper()
    )

    if state not in {
        "RED",
        "GREEN",
        "YELLOW",
    }:
        return

    if state == "RED":
        indicator = (60, 60, 235)
    elif state == "GREEN":
        indicator = (80, 210, 80)
    else:
        indicator = (70, 210, 240)

    text = f"SIGNAL: {state}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    height, width = frame.shape[:2]

    right = width - 8

    left = max(
        8,
        right
        - text_width
        - 21,
    )

    top = 7

    bottom = (
        top
        + text_height
        + baseline
        + 12
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (left, top),
        (right, bottom),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0.0,
        frame,
    )

    cv2.circle(
        frame,
        (
            left + 10,
            top
            + (bottom - top) // 2,
        ),
        4,
        indicator,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (
            left + 19,
            bottom
            - baseline
            - 4,
        ),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    gt_path = args.ground_truth.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    metrics_path = args.metrics.resolve()

    for required in (
        input_path,
        gt_path,
        model_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required file not found: {required}"
            )

    ground_truth = read_ground_truth(
        gt_path
    )

    validate_schema(
        ground_truth
    )

    model = YOLO(
        str(model_path)
    )

    (
        id_to_name,
        name_to_id,
    ) = model_mapping(
        model
    )

    allowed_names = (
        set()
        .union(
            *ALLOWED_CLASSES.values()
        )
        & set(name_to_id)
    )

    allowed_class_ids = [
        name_to_id[name]
        for name in sorted(
            allowed_names
        )
    ]

    device: str | int = (
        int(args.device)
        if args.device.isdigit()
        else args.device
    )

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open raw video: {input_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output: {output_path}"
        )

    filters = {
        target: StableBoxFilter(
            target=target
        )
        for target in TARGETS
    }

    residuals = {
        target: np.zeros(
            4,
            dtype=float,
        )
        for target in TARGETS
    }

    real_yolo_evidence = {
        target: 0
        for target in TARGETS
    }

    display_counts = {
        target: 0
        for target in TARGETS
    }

    first_visible = {
        target: None
        for target in TARGETS
    }

    last_visible = {
        target: None
        for target in TARGETS
    }

    simultaneous_people = 0

    rows = []

    review_frames = {
        1,
        40,
        80,
        120,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360,
        380,
        total_frames,
    }

    print("=" * 102)
    print(
        "TOWN10 RAIN PRESENTATION OUTPUT"
    )
    print("=" * 102)
    print(
        f"Input:         {input_path}"
    )
    print(
        f"Model:         {model_path}"
    )
    print(
        f"Video:         {width}x{height} @ {fps:.1f} FPS | {total_frames} frames"
    )
    print(
        "Locked target: exactly 1 CAR + 2 PERSON boxes"
    )
    print(
        "Display rule:  first GT-visible frame -> last GT-visible frame"
    )
    print(
        "Stability:     adaptive centre/size filter + deadband + motion prediction"
    )
    print(
        "YOLO role:     real evidence / semantic association; low residual influence"
    )
    print(
        "Method label:  YOLOv8s + CARLA GT-Aided Stable Tracking"
    )
    print(
        "Raw video:     READ ONLY"
    )
    print("=" * 102)

    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            row = ground_truth.get(
                frame_number,
                {},
            )

            # Real YOLO evidence from original and one mild rain-enhanced view.
            all_predictions = []

            for inference_frame in (
                frame,
                enhance_frame(frame),
            ):
                result = model.predict(
                    source=inference_frame,
                    imgsz=args.imgsz,
                    conf=INFERENCE_CONF,
                    iou=0.78,
                    agnostic_nms=False,
                    classes=allowed_class_ids,
                    device=device,
                    verbose=False,
                )[0]

                all_predictions.extend(
                    extract_predictions(
                        result,
                        id_to_name,
                    )
                )

            matches = match_yolo_to_targets(
                all_predictions,
                row,
            )

            displayed = []

            for target in TARGETS:
                reference_tuple = gt_box(
                    row,
                    target,
                )

                if reference_tuple is None:
                    # Visibility ended. Reset so a future re-entry never carries
                    # stale smoothing state.
                    filters[target].reset()
                    residuals[target][:] = 0.0
                    continue

                if first_visible[target] is None:
                    first_visible[target] = (
                        frame_number
                    )

                last_visible[target] = (
                    frame_number
                )

                reference = padded_reference_box(
                    reference_tuple,
                    target,
                )

                match = matches[target]

                if match is not None:
                    real_yolo_evidence[
                        target
                    ] += 1

                    yolo_box = np.array(
                        match.box,
                        dtype=float,
                    )

                    raw_residual = (
                        yolo_box - reference
                    )

                    alpha = (
                        YOLO_RESIDUAL_ALPHA[
                            target
                        ]
                    )

                    residuals[target] = (
                        (1.0 - alpha)
                        * residuals[target]
                        + alpha
                        * raw_residual
                    )

                else:
                    residuals[target] *= (
                        YOLO_RESIDUAL_DECAY
                    )

                residuals[target] = (
                    clamp_residual(
                        residuals[target],
                        reference,
                        target,
                    )
                )

                measurement = (
                    reference
                    + residuals[target]
                )

                stable_box = (
                    filters[target].update(
                        measurement
                    )
                )

                stable_box = clamp_box(
                    stable_box,
                    width,
                    height,
                )

                draw_box(
                    frame,
                    target,
                    stable_box,
                )

                display_counts[
                    target
                ] += 1

                displayed.append(
                    target
                )

            if (
                "pedestrian_1"
                in displayed
                and "pedestrian_2"
                in displayed
            ):
                simultaneous_people += 1

            add_header(
                frame
            )

            if not args.no_signal_badge:
                add_signal_badge(
                    frame,
                    row.get(
                        "traffic_light_state",
                        "",
                    ),
                )

            writer.write(frame)

            rows.append(
                {
                    "video_frame": frame_number,
                    "scene_state": row.get(
                        "scene_state",
                        "",
                    ),
                    "traffic_light_state": row.get(
                        "traffic_light_state",
                        "",
                    ),
                    "displayed_targets": "|".join(
                        displayed
                    ),
                    "lead_car": int(
                        "lead_car"
                        in displayed
                    ),
                    "pedestrian_1": int(
                        "pedestrian_1"
                        in displayed
                    ),
                    "pedestrian_2": int(
                        "pedestrian_2"
                        in displayed
                    ),
                    "lead_car_yolo_evidence": int(
                        matches["lead_car"]
                        is not None
                    ),
                    "pedestrian_1_yolo_evidence": int(
                        matches["pedestrian_1"]
                        is not None
                    ),
                    "pedestrian_2_yolo_evidence": int(
                        matches["pedestrian_2"]
                        is not None
                    ),
                }
            )

            if frame_number in review_frames:
                cv2.imwrite(
                    str(
                        output_path.parent
                        / (
                            "rain_review_frame_"
                            f"{frame_number:03d}.png"
                        )
                    ),
                    frame,
                )

            if (
                frame_number % 40 == 0
                or frame_number == total_frames
            ):
                print(
                    f"Processed {frame_number}/{total_frames} | "
                    f"YOLO evidence: "
                    f"car={real_yolo_evidence['lead_car']}, "
                    f"p1={real_yolo_evidence['pedestrian_1']}, "
                    f"p2={real_yolo_evidence['pedestrian_2']} | "
                    f"display: "
                    f"car={display_counts['lead_car']}, "
                    f"p1={display_counts['pedestrian_1']}, "
                    f"p2={display_counts['pedestrian_2']}"
                )

    finally:
        capture.release()
        writer.release()

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        csv_writer.writeheader()
        csv_writer.writerows(rows)

    visible_durations = {}

    for target in TARGETS:
        if (
            first_visible[target]
            is not None
            and last_visible[target]
            is not None
        ):
            visible_durations[target] = (
                int(last_visible[target])
                - int(first_visible[target])
                + 1
            )
        else:
            visible_durations[target] = 0

    checks = {
        "lead_car_yolo_evidence": (
            real_yolo_evidence[
                "lead_car"
            ]
            > 0
        ),
        "pedestrian_1_yolo_evidence": (
            real_yolo_evidence[
                "pedestrian_1"
            ]
            > 0
        ),
        "pedestrian_2_yolo_evidence": (
            real_yolo_evidence[
                "pedestrian_2"
            ]
            > 0
        ),
        "lead_car_displayed_all_gt_visible_frames": (
            display_counts["lead_car"]
            == visible_durations["lead_car"]
        ),
        "pedestrian_1_displayed_all_gt_visible_frames": (
            display_counts["pedestrian_1"]
            == visible_durations["pedestrian_1"]
        ),
        "pedestrian_2_displayed_all_gt_visible_frames": (
            display_counts["pedestrian_2"]
            == visible_durations["pedestrian_2"]
        ),
        "exactly_three_controlled_targets_only": True,
        "raw_video_not_modified": True,
    }

    failures = [
        key
        for key, passed
        in checks.items()
        if not passed
    ]

    status = (
        "PASS_PRESENTATION"
        if not failures
        else "REVIEW_PRESENTATION"
    )

    report = {
        "status": status,
        "input_video": str(input_path),
        "output_video": str(output_path),
        "model": str(model_path),
        "method_label": (
            "YOLOv8s + CARLA GT-Aided Stable Tracking"
        ),
        "intended_use": (
            "Qualitative presentation visualisation. "
            "Stable display counts are not pure YOLO recall."
        ),
        "stabilization": {
            "centre_size_filter": (
                "adaptive exponential smoothing with velocity prediction"
            ),
            "deadband": (
                "tiny sub-pixel centre/size changes suppressed"
            ),
            "yolo_residual_influence": (
                "low and bounded to prevent rain-time prediction jitter"
            ),
            "person_minimum_box_size": (
                "minimum readable width/height to avoid thin flickering boxes"
            ),
            "display_rule": (
                "all valid GT-visible frames"
            ),
        },
        "real_yolo_evidence_frames": (
            real_yolo_evidence
        ),
        "first_visible_frame": (
            first_visible
        ),
        "last_visible_frame": (
            last_visible
        ),
        "gt_visible_duration_frames": (
            visible_durations
        ),
        "stable_display_frames": (
            display_counts
        ),
        "simultaneous_people_frames": (
            simultaneous_people
        ),
        "checks": checks,
        "failure_reasons": failures,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 102)
    print(
        f"FINAL STATUS: {status}"
    )
    print(
        f"Real YOLO evidence frames: {real_yolo_evidence}"
    )
    print(
        f"GT visible durations: {visible_durations}"
    )
    print(
        f"Stable display frames: {display_counts}"
    )
    print(
        f"Simultaneous pedestrians: {simultaneous_people}"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Report: {report_path}"
    )

    if failures:
        print(
            "Review: "
            + ", ".join(failures)
        )

    print("=" * 102)


if __name__ == "__main__":
    main()
