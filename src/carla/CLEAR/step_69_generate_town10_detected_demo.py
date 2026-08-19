from __future__ import annotations
import os

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_RAW_ROOT = (
    CARLA_ROOT / "outputs" / "town10_professional_raw_demo"
)

DEFAULT_MODEL = (
    CARLA_ROOT
    / "outputs"
    / "training_runs"
    / "carla_multiclass_yolov8s_final"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT_ROOT = (
    CARLA_ROOT / "outputs" / "town10_final_detected_demo"
)

TARGET_CLASSES = {"car", "truck", "bus"}

BOX_COLORS = {
    "car": (54, 210, 255),
    "truck": (80, 220, 110),
    "bus": (255, 170, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline YOLO inference on the accepted Town10 raw "
            "demo, trim the empty ending and export a polished video."
        )
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help=(
            "Accepted raw MP4. When omitted, the latest PASS Town10 "
            "raw-demo run is selected automatically."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--trim-tail-seconds",
        type=float,
        default=1.5,
        help=(
            "Seconds retained after the final ground-truth encounter."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def latest_accepted_raw_video() -> Path:
    candidates = sorted(
        DEFAULT_RAW_ROOT.glob(
            "run_*/town10_professional_raw.mp4"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for video_path in candidates:
        metadata_path = (
            video_path.parent / "run_metadata.json"
        )

        if not metadata_path.is_file():
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            continue

        if metadata.get("status") == "PASS":
            return video_path

    raise FileNotFoundError(
        "No accepted Town10 raw video was found under: "
        f"{DEFAULT_RAW_ROOT}"
    )


def resolve_trim_end_frame(
    input_video: Path,
    fps: float,
    frame_count: int,
    trim_tail_seconds: float,
) -> tuple[int, Optional[Path]]:
    metrics_path = input_video.parent / "frame_metrics.csv"

    if not metrics_path.is_file():
        return frame_count, None

    metrics = pd.read_csv(metrics_path)

    encounter_columns = [
        column
        for column in (
            "car_encounter",
            "truck_encounter",
            "bus_encounter",
        )
        if column in metrics.columns
    ]

    if not encounter_columns:
        return frame_count, metrics_path

    encounter_mask = (
        metrics[encounter_columns]
        .fillna(0)
        .astype(float)
        .max(axis=1)
        > 0
    )

    if not encounter_mask.any():
        return frame_count, metrics_path

    last_row_position = int(
        np.flatnonzero(
            encounter_mask.to_numpy()
        )[-1]
    )
    last_video_frame = int(
        metrics.iloc[last_row_position][
            "video_frame"
        ]
    )

    tail_frames = max(
        0,
        int(round(trim_tail_seconds * fps)),
    )
    trim_end_frame = min(
        frame_count,
        last_video_frame + tail_frames,
    )

    return max(1, trim_end_frame), metrics_path


def clean_class_name(value: Any) -> str:
    return str(value).strip().lower()


def class_name_from_model(
    model: YOLO,
    class_id: int,
) -> str:
    names = model.names

    if isinstance(names, dict):
        return clean_class_name(
            names.get(class_id, class_id)
        )

    return clean_class_name(names[class_id])


def draw_detection(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    color = BOX_COLORS.get(
        label,
        (255, 255, 255),
    )

    thickness = 2
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )

    text = f"{label.upper()} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    text_thickness = 1

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness,
        )
    )

    label_top = max(
        0,
        y1 - text_height - baseline - 8,
    )
    label_bottom = min(
        height - 1,
        label_top + text_height + baseline + 8,
    )
    label_right = min(
        width - 1,
        x1 + text_width + 12,
    )

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, label_top),
        (label_right, label_bottom),
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
        (x1 + 6, label_bottom - baseline - 4),
        font,
        font_scale,
        (15, 15, 15),
        text_thickness,
        cv2.LINE_AA,
    )


def add_minimal_header(
    frame: np.ndarray,
) -> None:
    text = "Town10HD | Clear | YOLOv8s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (
            8 + text_width + 14,
            8 + text_height + baseline + 10,
        ),
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
        (15, 8 + text_height + 4),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def create_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    return writer


def main() -> None:
    args = parse_args()

    input_video = (
        args.input_video.resolve()
        if args.input_video
        else latest_accepted_raw_video()
    )
    model_path = args.model.resolve()

    if not input_video.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    capture = cv2.VideoCapture(str(input_video))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_video}"
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

    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(
            "Input video metadata is invalid."
        )

    trim_end_frame, source_metrics = (
        resolve_trim_end_frame(
            input_video,
            fps,
            total_frames,
            args.trim_tail_seconds,
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        args.output_root.resolve()
        / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    detected_video_path = (
        run_dir / "town10_detected_clean.mp4"
    )
    metrics_path = (
        run_dir / "detection_frame_metrics.csv"
    )
    summary_path = run_dir / "run_summary.txt"
    metadata_path = run_dir / "run_metadata.json"

    model = YOLO(str(model_path))

    device_value: str | int = args.device

    if args.device.isdigit():
        device_value = int(args.device)

    use_half = (
        torch.cuda.is_available()
        and str(args.device).lower()
        not in {"cpu", "-1"}
    )

    writer = create_writer(
        detected_video_path,
        fps,
        width,
        height,
    )

    frame_rows: list[dict[str, Any]] = []
    class_detection_frames: Counter[str] = Counter()
    class_detection_instances: Counter[str] = Counter()
    class_confidences: dict[
        str, list[float]
    ] = defaultdict(list)
    inference_times_ms: list[float] = []

    print("=" * 88)
    print("STEP 69 - TOWN10 OFFLINE YOLO DEMO")
    print("=" * 88)
    print(f"Input:        {input_video}")
    print(f"Model:        {model_path}")
    print(f"Device:       {args.device}")
    print(
        f"Video:        {width}x{height} "
        f"@ {fps:.1f} FPS"
    )
    print(
        f"Frames:       {trim_end_frame}/"
        f"{total_frames}"
    )
    print(
        f"Final length: {trim_end_frame / fps:.2f} s"
    )
    print("Classes:      car, truck, bus")
    print("=" * 88)

    processed_frames = 0
    start_time = time.perf_counter()

    try:
        while processed_frames < trim_end_frame:
            success, frame = capture.read()

            if not success:
                break

            processed_frames += 1

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device_value,
                half=use_half,
                verbose=False,
            )[0]

            inference_ms = float(
                result.speed.get(
                    "inference",
                    float("nan"),
                )
            )
            inference_times_ms.append(
                inference_ms
            )

            frame_class_counts: Counter[str] = (
                Counter()
            )
            frame_confidences: list[float] = []

            boxes = result.boxes

            if boxes is not None:
                for detection in boxes:
                    class_id = int(
                        detection.cls.item()
                    )
                    label = class_name_from_model(
                        model,
                        class_id,
                    )

                    if label not in TARGET_CLASSES:
                        continue

                    confidence = float(
                        detection.conf.item()
                    )
                    coordinates = (
                        detection.xyxy[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    x1, y1, x2, y2 = [
                        int(round(value))
                        for value in coordinates
                    ]

                    draw_detection(
                        frame,
                        (x1, y1, x2, y2),
                        label,
                        confidence,
                    )

                    frame_class_counts[label] += 1
                    class_detection_instances[
                        label
                    ] += 1
                    class_confidences[label].append(
                        confidence
                    )
                    frame_confidences.append(
                        confidence
                    )

            for label in TARGET_CLASSES:
                if frame_class_counts[label] > 0:
                    class_detection_frames[
                        label
                    ] += 1

            add_minimal_header(frame)
            writer.write(frame)

            frame_rows.append(
                {
                    "video_frame": (
                        processed_frames
                    ),
                    "time_seconds": (
                        (processed_frames - 1) / fps
                    ),
                    "car_detections": (
                        frame_class_counts["car"]
                    ),
                    "truck_detections": (
                        frame_class_counts["truck"]
                    ),
                    "bus_detections": (
                        frame_class_counts["bus"]
                    ),
                    "total_target_detections": (
                        sum(
                            frame_class_counts.values()
                        )
                    ),
                    "mean_target_confidence": (
                        float(
                            np.mean(
                                frame_confidences
                            )
                        )
                        if frame_confidences
                        else float("nan")
                    ),
                    "inference_ms": inference_ms,
                }
            )

            if processed_frames in {
                1,
                max(1, trim_end_frame // 3),
                max(
                    1,
                    2 * trim_end_frame // 3,
                ),
                trim_end_frame,
            }:
                cv2.imwrite(
                    str(
                        run_dir
                        / (
                            f"detected_frame_"
                            f"{processed_frames:03d}.png"
                        )
                    ),
                    frame,
                )

            if (
                processed_frames
                % max(1, int(fps * 2))
                == 0
            ):
                print(
                    f"Processed {processed_frames}/"
                    f"{trim_end_frame} frames"
                )

    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError(
            "No frames were processed."
        )

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

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    mean_inference_ms = (
        float(
            np.nanmean(inference_times_ms)
        )
        if inference_times_ms
        else float("nan")
    )

    mean_confidences = {
        label: (
            float(np.mean(values))
            if values
            else None
        )
        for label, values
        in class_confidences.items()
    }

    metadata = {
        "status": "PASS",
        "input_video": str(input_video),
        "model": str(model_path),
        "capture_source": (
            "Accepted Town10 genuine "
            "sensor.camera.rgb raw demo"
        ),
        "device": str(args.device),
        "resolution": [width, height],
        "fps": fps,
        "source_frame_count": (
            total_frames
        ),
        "output_frame_count": (
            processed_frames
        ),
        "source_duration_seconds": (
            total_frames / fps
        ),
        "output_duration_seconds": (
            processed_frames / fps
        ),
        "trimmed_tail_seconds": (
            (total_frames - processed_frames)
            / fps
        ),
        "source_metrics": (
            str(source_metrics)
            if source_metrics
            else None
        ),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "image_size": args.imgsz,
        "half_precision": use_half,
        "class_detection_frames": dict(
            class_detection_frames
        ),
        "class_detection_instances": dict(
            class_detection_instances
        ),
        "class_mean_confidence": (
            mean_confidences
        ),
        "mean_inference_ms": (
            mean_inference_ms
        ),
        "processing_seconds": (
            elapsed_seconds
        ),
        "output_video": (
            str(detected_video_path)
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "STEP 69 - TOWN10 OFFLINE YOLO DEMO",
        "=" * 76,
        "Status: PASS",
        f"Input video: {input_video}",
        f"Model: {model_path}",
        (
            f"Resolution: {width}x{height}"
        ),
        f"FPS: {fps:.1f}",
        (
            f"Source duration: "
            f"{total_frames / fps:.2f} s"
        ),
        (
            f"Final duration: "
            f"{processed_frames / fps:.2f} s"
        ),
        (
            f"Trimmed ending: "
            f"{(total_frames - processed_frames) / fps:.2f} s"
        ),
        (
            "Detection frames: "
            f"car={class_detection_frames['car']}, "
            f"truck={class_detection_frames['truck']}, "
            f"bus={class_detection_frames['bus']}"
        ),
        (
            "Detection instances: "
            f"car={class_detection_instances['car']}, "
            f"truck={class_detection_instances['truck']}, "
            f"bus={class_detection_instances['bus']}"
        ),
        (
            f"Mean inference: "
            f"{mean_inference_ms:.3f} ms"
        ),
        f"Detected video: {detected_video_path}",
        f"Metrics: {metrics_path}",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("FINAL STATUS: PASS")
    print(
        f"Detected video: {detected_video_path}"
    )
    print(
        f"Final duration: "
        f"{processed_frames / fps:.2f} s"
    )
    print(
        "Detection frames: "
        f"car={class_detection_frames['car']}, "
        f"truck={class_detection_frames['truck']}, "
        f"bus={class_detection_frames['bus']}"
    )
    print(
        f"Mean inference: "
        f"{mean_inference_ms:.3f} ms"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()

