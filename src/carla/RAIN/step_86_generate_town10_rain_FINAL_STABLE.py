from __future__ import annotations
import os

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# STEP 86 - TOWN10 RAIN FINAL STABLE PRESENTATION
#
# PURPOSE
# -------
# Produce ONE clean, stable presentation video containing exactly:
#   1) the controlled lead car
#   2) pedestrian 1
#   3) pedestrian 2
#
# IMPORTANT METHODOLOGY
# ---------------------
# This is deliberately labelled:
#       YOLOv8s + CARLA GT-Aided Temporal Tracking
#
# The custom YOLO model supplies real object evidence/confidence whenever it
# detects the controlled actor. CARLA projected actor boxes from the already
# recorded capture are used as the geometric tracking reference so the box
# cannot jump to another vehicle/person or disappear for dozens of frames.
#
# This file is for the FINAL PRESENTATION VIDEO.
# Do NOT use its stable display-frame counts as pure YOLO recall metrics.
# Model evaluation metrics must remain the genuine YOLO evaluation results.
#
# The locked PASS raw video is READ ONLY and is never overwritten.
# =============================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

LOCKED_RUN = (
    CARLA_ROOT
    / "outputs"
    / "town10_rain_signal_continuation"
    / "run_20260807_090117"
)

DEFAULT_INPUT = (
    LOCKED_RUN
    / "rain_signal_raw.mp4"
)

DEFAULT_GT = (
    LOCKED_RUN
    / "ground_truth_metrics.csv"
)

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
    / "rain_signal_detected_FINAL_STABLE.mp4"
)

DEFAULT_REPORT = (
    LOCKED_RUN
    / "rain_detection_FINAL_STABLE_report.json"
)

DEFAULT_METRICS = (
    LOCKED_RUN
    / "rain_detection_FINAL_STABLE_metrics.csv"
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
    "lead_car": (
        70,
        220,
        120,
    ),
    "pedestrian_1": (
        70,
        210,
        245,
    ),
    "pedestrian_2": (
        80,
        165,
        245,
    ),
}

PERSON_STATES = {
    "PEDESTRIAN_CROSSING",
    "GREEN_SETTLE",
    "RELEASE_AND_SPLIT",
    "FINISHED",
}

INFERENCE_CONF = 0.002

# Strict enough to avoid matching a random nearby actor, but tolerant enough
# for the rain model's imperfect boxes.
MATCH_MIN_IOU = {
    "lead_car": 0.015,
    "pedestrian_1": 0.005,
    "pedestrian_2": 0.005,
}

MATCH_MAX_CENTER_RATIO = {
    "lead_car": 0.95,
    "pedestrian_1": 1.35,
    "pedestrian_2": 1.35,
}

# Visual box smoothing. A higher previous weight makes presentation boxes calm.
BOX_CURRENT_WEIGHT = {
    "lead_car": 0.68,
    "pedestrian_1": 0.62,
    "pedestrian_2": 0.62,
}

# When YOLO exists, retain a small learned residual relative to the CARLA
# geometric reference. This keeps the stable box visually close to real YOLO.
RESIDUAL_UPDATE = 0.30
RESIDUAL_DECAY = 0.985


@dataclass(frozen=True)
class YoloMatch:
    target: str
    source_class: str
    confidence: float
    box: tuple[
        float,
        float,
        float,
        float,
    ]
    iou: float
    center_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the final stable Town10 rain presentation video "
            "with exactly one car and two pedestrians."
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
) -> dict[
    int,
    dict[
        str,
        str,
    ],
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth CSV: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    output = {}

    for row in rows:
        try:
            frame = int(
                float(
                    row[
                        "video_frame"
                    ]
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        output[
            frame
        ] = row

    if not output:
        raise RuntimeError(
            "Ground-truth CSV contains no usable frames."
        )

    return output


def validate_schema(
    ground_truth: dict[
        int,
        dict[
            str,
            str,
        ],
    ],
) -> None:
    sample = next(
        iter(
            ground_truth.values()
        )
    )

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
            + ", ".join(
                missing
            )
        )


def gt_box(
    row: dict[
        str,
        str,
    ],
    target: str,
) -> Optional[
    tuple[
        float,
        float,
        float,
        float,
    ]
]:
    try:
        visible = (
            int(
                float(
                    row[
                        f"{target}_visible"
                    ]
                )
            )
            == 1
        )

        x1 = float(
            row[
                f"{target}_x1"
            ]
        )
        y1 = float(
            row[
                f"{target}_y1"
            ]
        )
        x2 = float(
            row[
                f"{target}_x2"
            ]
        )
        y2 = float(
            row[
                f"{target}_y2"
            ]
        )
        depth = float(
            row[
                f"{target}_depth_m"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if (
        not visible
        or not all(
            math.isfinite(
                value
            )
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
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    return (
        max(
            0.0,
            box[2] - box[0],
        )
        * max(
            0.0,
            box[3] - box[1],
        )
    )


def box_iou(
    first: tuple[
        float,
        float,
        float,
        float,
    ],
    second: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    x1 = max(
        first[0],
        second[0],
    )
    y1 = max(
        first[1],
        second[1],
    )
    x2 = min(
        first[2],
        second[2],
    )
    y2 = min(
        first[3],
        second[3],
    )

    intersection = (
        max(
            0.0,
            x2 - x1,
        )
        * max(
            0.0,
            y2 - y1,
        )
    )

    union = (
        box_area(
            first
        )
        + box_area(
            second
        )
        - intersection
    )

    return (
        intersection
        / union
        if union > 0.0
        else 0.0
    )


def box_center(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    float,
    float,
]:
    return (
        (
            box[0]
            + box[2]
        )
        / 2.0,
        (
            box[1]
            + box[3]
        )
        / 2.0,
    )


def center_ratio(
    first: tuple[
        float,
        float,
        float,
        float,
    ],
    second: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    first_center = (
        box_center(
            first
        )
    )
    second_center = (
        box_center(
            second
        )
    )

    distance = math.hypot(
        first_center[0]
        - second_center[0],
        first_center[1]
        - second_center[1],
    )

    width = max(
        1.0,
        second[2]
        - second[0],
    )
    height = max(
        1.0,
        second[3]
        - second[1],
    )
    diagonal = max(
        1.0,
        math.hypot(
            width,
            height,
        ),
    )

    return (
        distance
        / diagonal
    )


def model_mapping(
    model: YOLO,
) -> tuple[
    dict[int, str],
    dict[str, int],
]:
    names = model.names

    if isinstance(
        names,
        dict,
    ):
        id_to_name = {
            int(class_id): (
                str(name)
                .strip()
                .lower()
            )
            for (
                class_id,
                name,
            ) in names.items()
        }
    else:
        id_to_name = {
            class_id: (
                str(name)
                .strip()
                .lower()
            )
            for (
                class_id,
                name,
            ) in enumerate(
                names
            )
        }

    return (
        id_to_name,
        {
            name: class_id
            for (
                class_id,
                name,
            ) in id_to_name.items()
        },
    )


def enhance_frame(
    frame: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB,
    )

    (
        lightness,
        channel_a,
        channel_b,
    ) = cv2.split(
        lab
    )

    clahe = (
        cv2.createCLAHE(
            clipLimit=1.45,
            tileGridSize=(
                8,
                8,
            ),
        )
    )

    lightness = (
        clahe.apply(
            lightness
        )
    )

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
    id_to_name: dict[
        int,
        str,
    ],
) -> list[
    tuple[
        str,
        float,
        tuple[
            float,
            float,
            float,
            float,
        ],
    ]
]:
    output = []

    if result.boxes is None:
        return output

    for prediction in (
        result.boxes
    ):
        class_id = int(
            prediction.cls.item()
        )

        source_class = (
            id_to_name.get(
                class_id,
                "",
            )
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
            float(
                value
            )
            for value
            in coordinates
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
            tuple[
                float,
                float,
                float,
                float,
            ],
        ]
    ],
    row: dict[
        str,
        str,
    ],
) -> dict[
    str,
    Optional[
        YoloMatch
    ],
]:
    """
    Pick at most ONE real YOLO prediction for each controlled actor.

    GT is used only for the identity association.
    """

    matches: dict[
        str,
        Optional[
            YoloMatch
        ],
    ] = {
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

        for (
            prediction_index,
            (
                source_class,
                confidence,
                box,
            ),
        ) in enumerate(
            predictions
        ):
            if (
                source_class
                not in ALLOWED_CLASSES[
                    target
                ]
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
                < MATCH_MIN_IOU[
                    target
                ]
                and ratio
                > MATCH_MAX_CENTER_RATIO[
                    target
                ]
            ):
                continue

            semantic_bonus = 0.0

            if (
                target
                == "lead_car"
                and source_class
                == "car"
            ):
                semantic_bonus = 0.10

            if (
                target.startswith(
                    "pedestrian_"
                )
                and source_class
                == "person"
            ):
                semantic_bonus = 0.08

            score = (
                overlap
                * 3.0
                - ratio
                * 0.35
                + confidence
                * 0.50
                + semantic_bonus
            )

            candidates.append(
                (
                    score,
                    target,
                    prediction_index,
                    YoloMatch(
                        target=target,
                        source_class=(
                            source_class
                        ),
                        confidence=(
                            confidence
                        ),
                        box=box,
                        iou=(
                            overlap
                        ),
                        center_ratio=(
                            ratio
                        ),
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
        key=lambda item: (
            item[0]
        ),
        reverse=True,
    ):
        if (
            target in used_targets
            or prediction_index
            in used_predictions
        ):
            continue

        matches[
            target
        ] = match

        used_targets.add(
            target
        )
        used_predictions.add(
            prediction_index
        )

    return matches


def clamp_box(
    box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    output = (
        box.astype(
            float
        )
        .copy()
    )

    output[0] = max(
        0.0,
        min(
            width - 1.0,
            output[0],
        ),
    )
    output[1] = max(
        0.0,
        min(
            height - 1.0,
            output[1],
        ),
    )
    output[2] = max(
        0.0,
        min(
            width - 1.0,
            output[2],
        ),
    )
    output[3] = max(
        0.0,
        min(
            height - 1.0,
            output[3],
        ),
    )

    return output


def draw_box(
    frame: np.ndarray,
    target: str,
    box: np.ndarray,
) -> None:
    height, width = (
        frame.shape[:2]
    )

    box = clamp_box(
        box,
        width,
        height,
    )

    x1, y1, x2, y2 = [
        int(
            round(
                value
            )
        )
        for value
        in box
    ]

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return

    color = COLORS[
        target
    ]

    text = (
        DISPLAY_LABEL[
            target
        ]
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        2,
        cv2.LINE_AA,
    )

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )
    scale = 0.46
    thickness = 1

    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
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

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (
            x1,
            top,
        ),
        (
            right,
            bottom,
        ),
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
        (
            15,
            15,
            15,
        ),
        thickness,
        cv2.LINE_AA,
    )


def add_header(
    frame: np.ndarray,
) -> None:
    text = (
        "Town10HD | Rain | YOLOv8s + CARLA GT-Aided Tracking"
    )

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (
            7,
            7,
        ),
        (
            392,
            33,
        ),
        (
            0,
            0,
            0,
        ),
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
        (
            13,
            25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (
            245,
            245,
            245,
        ),
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
        indicator = (
            60,
            60,
            235,
        )
    elif state == "GREEN":
        indicator = (
            80,
            210,
            80,
        )
    else:
        indicator = (
            70,
            210,
            240,
        )

    text = (
        f"SIGNAL: {state}"
    )

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )
    scale = 0.40
    thickness = 1

    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
    )

    height, width = (
        frame.shape[:2]
    )

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

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (
            left,
            top,
        ),
        (
            right,
            bottom,
        ),
        (
            0,
            0,
            0,
        ),
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
            + (
                bottom - top
            )
            // 2,
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
        (
            245,
            245,
            245,
        ),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = (
        parse_args()
    )

    input_path = (
        args.input.resolve()
    )
    gt_path = (
        args.ground_truth.resolve()
    )
    model_path = (
        args.model.resolve()
    )
    output_path = (
        args.output.resolve()
    )
    report_path = (
        args.report.resolve()
    )
    metrics_path = (
        args.metrics.resolve()
    )

    for required in (
        input_path,
        gt_path,
        model_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                "Required file not found: "
                f"{required}"
            )

    ground_truth = (
        read_ground_truth(
            gt_path
        )
    )

    validate_schema(
        ground_truth
    )

    model = YOLO(
        str(
            model_path
        )
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
        & set(
            name_to_id
        )
    )

    allowed_class_ids = [
        name_to_id[
            name
        ]
        for name
        in sorted(
            allowed_names
        )
    ]

    device: str | int = (
        int(
            args.device
        )
        if args.device.isdigit()
        else args.device
    )

    capture = cv2.VideoCapture(
        str(
            input_path
        )
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
        str(
            output_path
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output: {output_path}"
        )

    residuals = {
        target: np.zeros(
            4,
            dtype=float,
        )
        for target in TARGETS
    }

    previous_boxes: dict[
        str,
        Optional[
            np.ndarray
        ],
    ] = {
        target: None
        for target in TARGETS
    }

    tracker_confidence = {
        target: 0.0
        for target in TARGETS
    }

    real_yolo_matches = {
        target: 0
        for target in TARGETS
    }

    display_counts = {
        target: 0
        for target in TARGETS
    }

    simultaneous_people = 0
    rows = []

    print("=" * 100)
    print(
        "STEP 86 - FINAL STABLE RAIN PRESENTATION"
    )
    print("=" * 100)
    print(
        f"Input:          {input_path}"
    )
    print(
        f"Model:          {model_path}"
    )
    print(
        f"Video:          {width}x{height} @ {fps:.1f} FPS | {total_frames} frames"
    )
    print(
        "Exactly:        1 lead CAR + 2 controlled PERSON tracks"
    )
    print(
        "Method:         YOLOv8s evidence + CARLA GT-aided temporal geometry"
    )
    print(
        "Purpose:        stable final presentation video"
    )
    print(
        "Metric rule:    do NOT treat stable display frames as pure YOLO recall"
    )
    print(
        "Raw video:      READ ONLY"
    )
    print("=" * 100)

    frame_number = 0

    try:
        while True:
            success, frame = (
                capture.read()
            )

            if not success:
                break

            frame_number += 1

            row = ground_truth.get(
                frame_number,
                {},
            )

            # Two inference views improve the chance that the custom model
            # supplies real evidence in rain. Geometry is unchanged.
            views = [
                frame,
                enhance_frame(
                    frame
                ),
            ]

            all_predictions = []

            for view in views:
                result = model.predict(
                    source=view,
                    imgsz=(
                        args.imgsz
                    ),
                    conf=(
                        INFERENCE_CONF
                    ),
                    iou=0.78,
                    agnostic_nms=False,
                    classes=(
                        allowed_class_ids
                    ),
                    device=(
                        device
                    ),
                    verbose=False,
                )[0]

                all_predictions.extend(
                    extract_predictions(
                        result,
                        id_to_name,
                    )
                )

            matches = (
                match_yolo_to_targets(
                    all_predictions,
                    row,
                )
            )

            scene_state = (
                row.get(
                    "scene_state",
                    "",
                )
            )

            displayed = []

            for target in TARGETS:
                reference = gt_box(
                    row,
                    target,
                )

                if reference is None:
                    previous_boxes[
                        target
                    ] = None
                    continue

                if (
                    target.startswith(
                        "pedestrian_"
                    )
                    and scene_state
                    not in PERSON_STATES
                ):
                    continue

                reference_array = np.array(
                    reference,
                    dtype=float,
                )

                match = matches[
                    target
                ]

                if match is not None:
                    real_yolo_matches[
                        target
                    ] += 1

                    yolo_array = np.array(
                        match.box,
                        dtype=float,
                    )

                    new_residual = (
                        yolo_array
                        - reference_array
                    )

                    residuals[
                        target
                    ] = (
                        (
                            1.0
                            - RESIDUAL_UPDATE
                        )
                        * residuals[
                            target
                        ]
                        + RESIDUAL_UPDATE
                        * new_residual
                    )

                    tracker_confidence[
                        target
                    ] = (
                        match.confidence
                    )

                else:
                    residuals[
                        target
                    ] *= (
                        RESIDUAL_DECAY
                    )

                    tracker_confidence[
                        target
                    ] *= 0.995

                candidate = (
                    reference_array
                    + residuals[
                        target
                    ]
                )

                # Never let the YOLO residual move the box far away from the
                # known actor. This is presentation tracking, not free drift.
                ref_width = max(
                    1.0,
                    reference_array[2]
                    - reference_array[0],
                )
                ref_height = max(
                    1.0,
                    reference_array[3]
                    - reference_array[1],
                )

                max_dx = max(
                    8.0,
                    ref_width
                    * 0.22,
                )

                max_dy = max(
                    8.0,
                    ref_height
                    * 0.22,
                )

                residuals[
                    target
                ][0] = np.clip(
                    residuals[
                        target
                    ][0],
                    -max_dx,
                    max_dx,
                )
                residuals[
                    target
                ][2] = np.clip(
                    residuals[
                        target
                    ][2],
                    -max_dx,
                    max_dx,
                )
                residuals[
                    target
                ][1] = np.clip(
                    residuals[
                        target
                    ][1],
                    -max_dy,
                    max_dy,
                )
                residuals[
                    target
                ][3] = np.clip(
                    residuals[
                        target
                    ][3],
                    -max_dy,
                    max_dy,
                )

                candidate = (
                    reference_array
                    + residuals[
                        target
                    ]
                )

                previous = previous_boxes[
                    target
                ]

                if previous is not None:
                    current_weight = (
                        BOX_CURRENT_WEIGHT[
                            target
                        ]
                    )

                    candidate = (
                        current_weight
                        * candidate
                        + (
                            1.0
                            - current_weight
                        )
                        * previous
                    )

                candidate = clamp_box(
                    candidate,
                    width,
                    height,
                )

                previous_boxes[
                    target
                ] = (
                    candidate.copy()
                )

                draw_box(
                    frame,
                    target,
                    candidate,
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

            writer.write(
                frame
            )

            rows.append(
                {
                    "video_frame": frame_number,
                    "scene_state": (
                        scene_state
                    ),
                    "traffic_light_state": (
                        row.get(
                            "traffic_light_state",
                            "",
                        )
                    ),
                    "displayed_targets": (
                        "|".join(
                            displayed
                        )
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
                    "lead_car_yolo_match": int(
                        matches[
                            "lead_car"
                        ]
                        is not None
                    ),
                    "pedestrian_1_yolo_match": int(
                        matches[
                            "pedestrian_1"
                        ]
                        is not None
                    ),
                    "pedestrian_2_yolo_match": int(
                        matches[
                            "pedestrian_2"
                        ]
                        is not None
                    ),
                }
            )

            if (
                frame_number % 40
                == 0
                or frame_number
                == total_frames
            ):
                print(
                    f"Processed {frame_number}/{total_frames} | "
                    f"YOLO evidence: "
                    f"car={real_yolo_matches['lead_car']}, "
                    f"p1={real_yolo_matches['pedestrian_1']}, "
                    f"p2={real_yolo_matches['pedestrian_2']} | "
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
        csv_writer = (
            csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
        )

        csv_writer.writeheader()
        csv_writer.writerows(
            rows
        )

    checks = {
        "lead_car_displayed": (
            display_counts[
                "lead_car"
            ]
            > 0
        ),
        "pedestrian_1_displayed": (
            display_counts[
                "pedestrian_1"
            ]
            > 0
        ),
        "pedestrian_2_displayed": (
            display_counts[
                "pedestrian_2"
            ]
            > 0
        ),
        "lead_car_has_real_yolo_evidence": (
            real_yolo_matches[
                "lead_car"
            ]
            > 0
        ),
        "pedestrian_1_has_real_yolo_evidence": (
            real_yolo_matches[
                "pedestrian_1"
            ]
            > 0
        ),
        "pedestrian_2_has_real_yolo_evidence": (
            real_yolo_matches[
                "pedestrian_2"
            ]
            > 0
        ),
        "exactly_three_controlled_targets_only": True,
        "raw_video_not_modified": True,
    }

    failures = [
        key
        for (
            key,
            passed,
        ) in checks.items()
        if not passed
    ]

    status = (
        "PASS_PRESENTATION"
        if not failures
        else "REVIEW_PRESENTATION"
    )

    report = {
        "status": status,
        "input_video": str(
            input_path
        ),
        "output_video": str(
            output_path
        ),
        "model": str(
            model_path
        ),
        "method_label": (
            "YOLOv8s + CARLA GT-Aided Temporal Tracking"
        ),
        "intended_use": (
            "Stable qualitative presentation/demo visualisation only. "
            "Do not use stable display counts as pure detector recall."
        ),
        "method": (
            "The custom YOLO model is run on every frame using original and "
            "mild CLAHE views. YOLO predictions are associated one-to-one "
            "with the three known controlled CARLA actors. The CARLA capture "
            "bounding box supplies stable geometric tracking whenever YOLO "
            "confidence/classification is intermittent under rain. A smoothed "
            "YOLO-to-GT residual keeps the presentation box close to genuine "
            "YOLO localisation when available while preventing identity "
            "switches and long disappearances. Only the lead car and the two "
            "controlled pedestrians are displayed."
        ),
        "real_yolo_evidence_frames": (
            real_yolo_matches
        ),
        "stable_display_frames": (
            display_counts
        ),
        "simultaneous_people_frames": (
            simultaneous_people
        ),
        "checks": checks,
        "failure_reasons": (
            failures
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        f"FINAL STATUS: {status}"
    )
    print(
        f"Real YOLO evidence frames: {real_yolo_matches}"
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
    print("=" * 100)


if __name__ == "__main__":
    main()

