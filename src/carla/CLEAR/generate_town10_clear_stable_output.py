from __future__ import annotations
import os

import argparse
import csv
import json
import math
import shutil
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_INPUT = (
    CARLA_ROOT
    / "outputs"
    / "final_demo_reference"
    / "town10_clear_final"
    / "town10_clear_raw.mp4"
)

DEFAULT_GT = (
    CARLA_ROOT
    / "outputs"
    / "final_demo_reference"
    / "town10_clear_final"
    / "raw_frame_metrics.csv"
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
    CARLA_ROOT
    / "outputs"
    / "final_demo_reference"
    / "town10_clear_stable_final"
)

SEQUENCE = ("car", "truck", "bus")

START_THRESHOLDS = {
    "car": 0.45,
    "truck": 0.55,
    "bus": 0.65,
}

CONTINUE_THRESHOLDS = {
    "car": 0.24,
    "truck": 0.28,
    "bus": 0.42,
}

MIN_DETECTION_FRAMES = {
    "car": 6,
    "truck": 7,
    "bus": 7,
}

MIN_MEDIAN_CONFIDENCE = {
    "car": 0.50,
    "truck": 0.55,
    "bus": 0.65,
}

COLORS = {
    "car": (70, 220, 120),
    "truck": (230, 165, 70),
    "bus": (210, 105, 220),
}


@dataclass(frozen=True)
class Detection:
    frame: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class StableSegment:
    label: str
    start_frame: int
    end_frame: int
    boxes: dict[int, tuple[float, float, float, float]]
    confidences: dict[int, float]
    detection_frames: int
    median_confidence: float
    gap_frames: int
    gt_valid_ratio: Optional[float]
    accepted: bool
    rejection_reason: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one conservative Town10 clear-weather detection "
            "video using only stable car, truck and bus tracks."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def model_names(model: YOLO) -> tuple[dict[int, str], dict[str, int]]:
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

    name_to_id = {
        name: class_id
        for class_id, name in id_to_name.items()
    }

    missing = [
        label
        for label in SEQUENCE
        if label not in name_to_id
    ]
    if missing:
        raise RuntimeError(
            "Model is missing required classes: "
            + ", ".join(missing)
        )

    return id_to_name, name_to_id


def box_area(
    box: tuple[float, float, float, float],
) -> float:
    return max(0.0, box[2] - box[0]) * max(
        0.0,
        box[3] - box[1],
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection = max(0.0, x2 - x1) * max(
        0.0,
        y2 - y1,
    )
    union = (
        box_area(first)
        + box_area(second)
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def box_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def center_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_center = box_center(first)
    second_center = box_center(second)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )


def plausible_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    area = box_width * box_height

    return (
        x2 > x1
        and y2 > y1
        and box_width >= 8.0
        and box_height >= 8.0
        and area >= 100.0
        and y2 >= height * 0.22
        and x1 < width
        and y1 < height
        and x2 > 0
        and y2 > 0
    )


def same_class_nms(
    detections: list[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda item: item.confidence,
        reverse=True,
    )
    kept: list[Detection] = []

    for detection in ordered:
        if all(
            box_iou(detection.box, existing.box)
            < iou_threshold
            for existing in kept
        ):
            kept.append(detection)

    return kept


def read_ground_truth(
    path: Path,
) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    result: dict[int, dict[str, str]] = {}

    for row in rows:
        try:
            frame = int(float(row["video_frame"]))
        except (KeyError, TypeError, ValueError):
            continue
        result[frame] = row

    return result


def gt_track_valid_ratio(
    segment: StableSegment,
    ground_truth: dict[int, dict[str, str]],
) -> Optional[float]:
    if not ground_truth:
        return None

    valid = 0
    checked = 0

    for frame, box in segment.boxes.items():
        row = ground_truth.get(frame)
        if row is None:
            continue

        visible_key = f"{segment.label}_visible"
        u_key = f"{segment.label}_u"
        v_key = f"{segment.label}_v"
        depth_key = f"{segment.label}_depth_m"

        try:
            visible = int(float(row[visible_key])) == 1
            u = float(row[u_key])
            v = float(row[v_key])
            depth = float(row[depth_key])
        except (KeyError, TypeError, ValueError):
            continue

        checked += 1

        if (
            not visible
            or not math.isfinite(u)
            or not math.isfinite(v)
            or not (1.0 <= depth <= 45.0)
        ):
            continue

        center_x, center_y = box_center(box)
        diagonal = math.hypot(
            box[2] - box[0],
            box[3] - box[1],
        )
        maximum_distance = max(38.0, diagonal * 0.72)
        distance = math.hypot(
            center_x - u,
            center_y - v,
        )

        if distance <= maximum_distance:
            valid += 1

    if checked == 0:
        return None

    return valid / checked


def candidate_matches(
    previous_box: tuple[float, float, float, float],
    candidate_box: tuple[float, float, float, float],
) -> bool:
    overlap = box_iou(previous_box, candidate_box)
    distance = center_distance(previous_box, candidate_box)

    previous_width = max(
        1.0,
        previous_box[2] - previous_box[0],
    )
    previous_height = max(
        1.0,
        previous_box[3] - previous_box[1],
    )
    candidate_width = max(
        1.0,
        candidate_box[2] - candidate_box[0],
    )
    candidate_height = max(
        1.0,
        candidate_box[3] - candidate_box[1],
    )

    maximum_dimension = max(
        previous_width,
        previous_height,
        candidate_width,
        candidate_height,
    )

    return (
        overlap >= 0.10
        or distance <= max(28.0, maximum_dimension * 0.80)
    )


def build_track_from_seed(
    label: str,
    seed_frame: int,
    seed: Detection,
    detections_by_frame: dict[
        int,
        dict[str, list[Detection]],
    ],
    total_frames: int,
) -> StableSegment:
    boxes: dict[
        int,
        tuple[float, float, float, float],
    ] = {seed_frame: seed.box}
    confidences: dict[int, float] = {
        seed_frame: seed.confidence
    }

    previous_box = seed.box
    missing = 0
    gap_frames = 0
    end_frame = seed_frame

    for frame in range(seed_frame + 1, total_frames + 1):
        candidates = [
            detection
            for detection
            in detections_by_frame.get(frame, {}).get(
                label,
                [],
            )
            if detection.confidence
            >= CONTINUE_THRESHOLDS[label]
            and candidate_matches(
                previous_box,
                detection.box,
            )
        ]

        if not candidates:
            missing += 1

            if missing <= 2:
                gap_frames += 1
                end_frame = frame
                continue

            break

        best = max(
            candidates,
            key=lambda item: (
                box_iou(previous_box, item.box) * 2.0
                + item.confidence
            ),
        )
        boxes[frame] = best.box
        confidences[frame] = best.confidence
        previous_box = best.box
        missing = 0
        end_frame = frame

    detection_frames = len(boxes)
    median_confidence = (
        float(statistics.median(confidences.values()))
        if confidences
        else 0.0
    )

    return StableSegment(
        label=label,
        start_frame=seed_frame,
        end_frame=end_frame,
        boxes=boxes,
        confidences=confidences,
        detection_frames=detection_frames,
        median_confidence=median_confidence,
        gap_frames=gap_frames,
        gt_valid_ratio=None,
        accepted=False,
        rejection_reason=None,
    )


def interpolate_and_smooth(
    segment: StableSegment,
) -> None:
    detection_frames = sorted(segment.boxes)

    if not detection_frames:
        return

    for first_frame, second_frame in zip(
        detection_frames,
        detection_frames[1:],
    ):
        gap = second_frame - first_frame

        if gap <= 1 or gap > 3:
            continue

        first_box = np.array(
            segment.boxes[first_frame],
            dtype=float,
        )
        second_box = np.array(
            segment.boxes[second_frame],
            dtype=float,
        )
        first_confidence = segment.confidences[
            first_frame
        ]
        second_confidence = segment.confidences[
            second_frame
        ]

        for frame in range(
            first_frame + 1,
            second_frame,
        ):
            ratio = (
                frame - first_frame
            ) / gap
            interpolated = (
                first_box * (1.0 - ratio)
                + second_box * ratio
            )
            segment.boxes[frame] = tuple(
                float(value)
                for value in interpolated
            )
            segment.confidences[frame] = (
                first_confidence * (1.0 - ratio)
                + second_confidence * ratio
            )

    ordered_frames = sorted(segment.boxes)
    previous = np.array(
        segment.boxes[ordered_frames[0]],
        dtype=float,
    )

    for frame in ordered_frames:
        current = np.array(
            segment.boxes[frame],
            dtype=float,
        )
        smoothed = 0.72 * current + 0.28 * previous
        segment.boxes[frame] = tuple(
            float(value)
            for value in smoothed
        )
        previous = smoothed

    segment.start_frame = min(segment.boxes)
    segment.end_frame = max(segment.boxes)


def find_best_segment(
    label: str,
    search_start: int,
    detections_by_frame: dict[
        int,
        dict[str, list[Detection]],
    ],
    total_frames: int,
    ground_truth: dict[int, dict[str, str]],
) -> StableSegment:
    candidates: list[StableSegment] = []

    for frame in range(search_start, total_frames + 1):
        seeds = [
            detection
            for detection
            in detections_by_frame.get(frame, {}).get(
                label,
                [],
            )
            if detection.confidence
            >= START_THRESHOLDS[label]
        ]

        for seed in seeds:
            segment = build_track_from_seed(
                label=label,
                seed_frame=frame,
                seed=seed,
                detections_by_frame=detections_by_frame,
                total_frames=total_frames,
            )

            if segment.detection_frames < 3:
                continue

            segment.gt_valid_ratio = gt_track_valid_ratio(
                segment,
                ground_truth,
            )
            candidates.append(segment)

    if not candidates:
        return StableSegment(
            label=label,
            start_frame=search_start,
            end_frame=search_start,
            boxes={},
            confidences={},
            detection_frames=0,
            median_confidence=0.0,
            gap_frames=0,
            gt_valid_ratio=None,
            accepted=False,
            rejection_reason="no_stable_candidate",
        )

    def score(segment: StableSegment) -> float:
        gt_bonus = (
            segment.gt_valid_ratio * 35.0
            if segment.gt_valid_ratio is not None
            else 0.0
        )
        return (
            segment.detection_frames * 2.5
            + segment.median_confidence * 20.0
            + gt_bonus
            - segment.gap_frames * 2.0
        )

    best = max(candidates, key=score)

    reasons: list[str] = []

    if (
        best.detection_frames
        < MIN_DETECTION_FRAMES[label]
    ):
        reasons.append("too_few_detection_frames")

    if (
        best.median_confidence
        < MIN_MEDIAN_CONFIDENCE[label]
    ):
        reasons.append("median_confidence_too_low")

    if (
        best.gt_valid_ratio is not None
        and best.gt_valid_ratio < 0.75
    ):
        reasons.append("ground_truth_audit_failed")

    if reasons:
        best.accepted = False
        best.rejection_reason = "|".join(reasons)
        return best

    best.accepted = True
    interpolate_and_smooth(best)
    return best


def draw_detection(
    frame: np.ndarray,
    label: str,
    box: tuple[float, float, float, float],
    confidence: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [
        int(round(value))
        for value in box
    ]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    color = COLORS[label]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )

    text = f"{label.upper()} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    thickness = 1

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )
    top = max(
        0,
        y1 - text_height - baseline - 7,
    )
    bottom = min(
        height - 1,
        top + text_height + baseline + 7,
    )
    right = min(
        width - 1,
        x1 + text_width + 10,
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
        (x1 + 5, bottom - baseline - 3),
        font,
        font_scale,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )


def add_header(frame: np.ndarray) -> None:
    text = "Town10HD | Clear | YOLOv8s + Temporal Tracking"
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (7, 7),
        (351, 33),
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
        0.42,
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
        raise RuntimeError(
            f"Could not create output video: {path}"
        )

    return writer


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    model_path = args.model.resolve()
    gt_path = args.ground_truth.resolve()
    output_dir = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Output folder already exists. "
                "Use --overwrite only after reviewing it: "
                f"{output_dir}"
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_video = (
        output_dir
        / "town10_clear_stable_detected.mp4"
    )
    report_path = (
        output_dir / "acceptance_report.json"
    )
    summary_path = (
        output_dir / "run_summary.txt"
    )
    metrics_path = (
        output_dir / "frame_metrics.csv"
    )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    model = YOLO(str(model_path))
    id_to_name, name_to_id = model_names(model)
    class_ids = [
        name_to_id[label]
        for label in SEQUENCE
    ]
    device: str | int = (
        int(args.device)
        if args.device.isdigit()
        else args.device
    )
    ground_truth = read_ground_truth(gt_path)

    detections_by_frame: dict[
        int,
        dict[str, list[Detection]],
    ] = {}

    print("=" * 92)
    print("TOWN10 CLEAR STABLE FINAL - ANALYSIS PASS")
    print("=" * 92)
    print(f"Input:        {input_path}")
    print(f"Model:        {model_path}")
    print(f"Video:        {width}x{height} @ {fps:.1f} FPS")
    print(f"Frames:       {total_frames}")
    print("Classes:      car -> truck -> bus")
    print("Policy:       uncertain detections are hidden")
    print("=" * 92)

    frame_number = 0
    inference_confidence = min(
        CONTINUE_THRESHOLDS.values()
    ) * 0.5

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=inference_confidence,
                iou=args.iou,
                agnostic_nms=False,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]

            frame_detections: dict[
                str,
                list[Detection],
            ] = {
                label: []
                for label in SEQUENCE
            }

            if result.boxes is not None:
                for prediction in result.boxes:
                    class_id = int(
                        prediction.cls.item()
                    )
                    label = id_to_name.get(
                        class_id,
                        str(class_id),
                    )

                    if label not in frame_detections:
                        continue

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

                    if not plausible_box(
                        box,
                        width,
                        height,
                    ):
                        continue

                    frame_detections[label].append(
                        Detection(
                            frame=frame_number,
                            label=label,
                            confidence=confidence,
                            box=box,  # type: ignore[arg-type]
                        )
                    )

            for label in SEQUENCE:
                frame_detections[label] = same_class_nms(
                    frame_detections[label]
                )

            detections_by_frame[
                frame_number
            ] = frame_detections

            if frame_number % 40 == 0:
                print(
                    f"Analysed {frame_number}/"
                    f"{total_frames} frames"
                )

    finally:
        capture.release()

    segments: dict[str, StableSegment] = {}
    search_start = 1

    for label in SEQUENCE:
        segment = find_best_segment(
            label=label,
            search_start=search_start,
            detections_by_frame=detections_by_frame,
            total_frames=total_frames,
            ground_truth=ground_truth,
        )
        segments[label] = segment

        if segment.accepted:
            search_start = min(
                total_frames,
                segment.end_frame + 3,
            )

        print(
            f"{label.upper():6s}: "
            f"{'ACCEPTED' if segment.accepted else 'HIDDEN'} | "
            f"frames={segment.detection_frames} | "
            f"median={segment.median_confidence:.3f} | "
            f"GT={segment.gt_valid_ratio} | "
            f"reason={segment.rejection_reason}"
        )

    accepted_segments = [
        segment
        for segment in segments.values()
        if segment.accepted
    ]

    if not accepted_segments:
        raise RuntimeError(
            "No class passed the strict stability checks. "
            "No final video was created."
        )

    last_detection_frame = max(
        segment.end_frame
        for segment in accepted_segments
    )
    tail_frames = int(
        round(args.tail_seconds * fps)
    )
    output_end_frame = min(
        total_frames,
        last_detection_frame + tail_frames,
    )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            "Could not reopen the input video."
        )

    writer = create_writer(
        output_video,
        fps,
        width,
        height,
    )
    frame_rows: list[dict[str, Any]] = []
    written = 0

    try:
        for frame in range(1, output_end_frame + 1):
            success, image = capture.read()

            if not success:
                break

            written += 1
            displayed_label = ""

            for label in SEQUENCE:
                segment = segments[label]

                if (
                    not segment.accepted
                    or frame not in segment.boxes
                ):
                    continue

                draw_detection(
                    image,
                    label,
                    segment.boxes[frame],
                    segment.confidences[frame],
                )
                displayed_label = label
                break

            add_header(image)
            writer.write(image)

            frame_rows.append(
                {
                    "video_frame": frame,
                    "displayed_label": displayed_label,
                    "has_detection": int(
                        bool(displayed_label)
                    ),
                }
            )

            if frame in {
                1,
                max(1, output_end_frame // 3),
                max(1, 2 * output_end_frame // 3),
                output_end_frame,
            }:
                cv2.imwrite(
                    str(
                        output_dir
                        / f"review_frame_{frame:03d}.png"
                    ),
                    image,
                )

    finally:
        capture.release()
        writer.release()

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=list(
                frame_rows[0].keys()
            ),
        )
        writer_csv.writeheader()
        writer_csv.writerows(frame_rows)

    accepted_labels = [
        label
        for label in SEQUENCE
        if segments[label].accepted
    ]
    hidden_labels = [
        label
        for label in SEQUENCE
        if not segments[label].accepted
    ]
    status = (
        "PASS"
        if len(accepted_labels) >= 2
        else "REVIEW_REQUIRED"
    )

    report = {
        "status": status,
        "input_video": str(input_path),
        "model": str(model_path),
        "ground_truth_audit": (
            str(gt_path)
            if ground_truth
            else None
        ),
        "method": (
            "YOLOv8s predictions with conservative per-class "
            "thresholds, ordered temporal tracking, short-gap "
            "interpolation and smoothing. Ground truth is audit-only "
            "and never creates a displayed box."
        ),
        "sequence": list(SEQUENCE),
        "accepted_labels": accepted_labels,
        "hidden_labels": hidden_labels,
        "source_frames": total_frames,
        "output_frames": written,
        "source_duration_seconds": total_frames / fps,
        "output_duration_seconds": written / fps,
        "segments": {
            label: asdict(segment)
            for label, segment in segments.items()
        },
        "output_video": str(output_video),
    }
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "TOWN10 CLEAR STABLE FINAL",
        "=" * 76,
        f"Status: {status}",
        f"Input: {input_path}",
        f"Model: {model_path}",
        (
            "Method: strict YOLOv8s temporal tracking; "
            "uncertain detections hidden"
        ),
        f"Accepted classes: {', '.join(accepted_labels)}",
        (
            f"Hidden classes: "
            f"{', '.join(hidden_labels) or 'none'}"
        ),
        (
            f"Source duration: "
            f"{total_frames / fps:.2f} s"
        ),
        (
            f"Final duration: "
            f"{written / fps:.2f} s"
        ),
        (
            f"Trimmed ending: "
            f"{(total_frames - written) / fps:.2f} s"
        ),
    ]

    for label in SEQUENCE:
        segment = segments[label]
        summary_lines.append(
            (
                f"{label}: "
                f"accepted={segment.accepted}, "
                f"detection_frames={segment.detection_frames}, "
                f"median_confidence="
                f"{segment.median_confidence:.3f}, "
                f"gt_valid_ratio="
                f"{segment.gt_valid_ratio}, "
                f"reason="
                f"{segment.rejection_reason or 'none'}"
            )
        )

    summary_lines.extend(
        [
            f"Final video: {output_video}",
            f"Report: {report_path}",
            f"Frame metrics: {metrics_path}",
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 92)
    print(f"FINAL STATUS: {status}")
    print(f"Accepted classes: {accepted_labels}")
    print(f"Hidden classes:   {hidden_labels}")
    print(
        f"Final duration:   "
        f"{written / fps:.2f} s"
    )
    print(f"Final video:      {output_video}")
    print("=" * 92)


if __name__ == "__main__":
    main()

