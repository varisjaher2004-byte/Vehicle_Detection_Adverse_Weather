from __future__ import annotations
import os

import argparse
import json
import math
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import carla
import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(os.environ.get("DMSC_GENERATED_DATASET_ROOT", ".")).resolve()

WEATHER_FOLDERS = (
    "clear",
    "light_rain",
    "heavy_rain",
    "light_fog",
    "dense_fog",
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve() / "outputs" / "town10_reference_route"


@dataclass
class MatchResult:
    rank: int
    spawn_index: int
    road_id: int
    lane_id: int
    score: float
    feature_score: float
    edge_score: float
    hash_score: float
    histogram_score: float
    transform: carla.Transform
    frame: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and lock the Town10HD_Opt spawn point that most "
            "closely matches representative clear frames from the Generated dataset."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Existing Generated dataset root containing clear, "
            "light_rain, heavy_rain, light_fog and dense_fog folders."
        ),
    )
    parser.add_argument(
        "--max-clear-references",
        type=int,
        default=12,
        help=(
            "Maximum evenly-spaced clear frames used for route matching."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--sensor-fps", type=float, default=5.0)
    parser.add_argument(
        "--scan-count",
        type=int,
        default=50,
        help=(
            "Number of strongest central/urban spawn points to scan. "
            "Use 0 to scan every spawn point."
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()



def natural_sort_key(path: Path) -> tuple[object, ...]:
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(
        int(part) if part.isdigit() else part
        for part in parts
    )


def discover_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []

    images = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=natural_sort_key)
    return images


def evenly_spaced_paths(
    paths: list[Path],
    maximum: int,
) -> list[Path]:
    if maximum < 1:
        raise ValueError(
            "--max-clear-references must be at least 1."
        )

    if len(paths) <= maximum:
        return paths

    indices = np.linspace(
        0,
        len(paths) - 1,
        num=maximum,
        dtype=int,
    )

    selected: list[Path] = []
    seen: set[int] = set()

    for index in indices:
        integer_index = int(index)

        if integer_index in seen:
            continue

        seen.add(integer_index)
        selected.append(paths[integer_index])

    return selected


def image_to_bgr(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def nearby_sidewalk_count(
    waypoint: carla.Waypoint,
) -> int:
    count = 0

    for direction in ("left", "right"):
        current: Optional[carla.Waypoint] = waypoint

        for _ in range(6):
            if current is None:
                break

            current = (
                current.get_left_lane()
                if direction == "left"
                else current.get_right_lane()
            )

            if current is None:
                break

            if current.lane_type == carla.LaneType.Sidewalk:
                count += 1
                break

    return count


def rank_spawn_points(
    world: carla.World,
) -> list[tuple[float, int, carla.Transform, carla.Waypoint]]:
    carla_map = world.get_map()
    spawn_points = list(carla_map.get_spawn_points())

    if not spawn_points:
        raise RuntimeError("Town10 has no spawn points.")

    center_x = sum(
        transform.location.x
        for transform in spawn_points
    ) / len(spawn_points)
    center_y = sum(
        transform.location.y
        for transform in spawn_points
    ) / len(spawn_points)

    ranked: list[
        tuple[float, int, carla.Transform, carla.Waypoint]
    ] = []

    for index, transform in enumerate(spawn_points):
        waypoint = carla_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            continue

        sidewalks = nearby_sidewalk_count(waypoint)
        distance_from_center = math.hypot(
            transform.location.x - center_x,
            transform.location.y - center_y,
        )

        junction_penalty = 12.0 if waypoint.is_junction else 0.0
        lane_width_penalty = abs(waypoint.lane_width - 3.5) * 2.0

        score = (
            sidewalks * 18.0
            - distance_from_center * 0.07
            - junction_penalty
            - lane_width_penalty
        )

        ranked.append(
            (score, index, transform, waypoint)
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def resize_for_comparison(
    image: np.ndarray,
    width: int = 320,
    height: int = 180,
) -> np.ndarray:
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def crop_comparison_region(image: np.ndarray) -> np.ndarray:
    """
    Ignore a thin top/bottom strip so sky/exposure and the vehicle hood
    do not dominate route matching.
    """
    height = image.shape[0]
    top = int(round(height * 0.08))
    bottom = int(round(height * 0.94))
    return image[top:bottom, :]


def difference_hash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )
    return (small[:, 1:] > small[:, :-1]).flatten()


def hash_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_hash = difference_hash(first)
    second_hash = difference_hash(second)
    return float(np.mean(first_hash == second_hash))


def histogram_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)

    first_hist = cv2.calcHist(
        [first_hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )
    second_hist = cv2.calcHist(
        [second_hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(first_hist, first_hist)
    cv2.normalize(second_hist, second_hist)

    correlation = cv2.compareHist(
        first_hist,
        second_hist,
        cv2.HISTCMP_CORREL,
    )
    return float(max(0.0, min(1.0, (correlation + 1.0) / 2.0)))


def edge_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
    second_gray = cv2.GaussianBlur(second_gray, (5, 5), 0)

    first_edges = cv2.Canny(first_gray, 45, 130)
    second_edges = cv2.Canny(second_gray, 45, 130)

    first_float = first_edges.astype(np.float32) / 255.0
    second_float = second_edges.astype(np.float32) / 255.0

    numerator = float(
        np.sum(first_float * second_float)
    )
    denominator = math.sqrt(
        float(np.sum(first_float * first_float))
        * float(np.sum(second_float * second_float))
    )

    if denominator <= 1e-9:
        return 0.0

    return max(0.0, min(1.0, numerator / denominator))


def feature_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create(
        nfeatures=1400,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=10,
    )

    first_keypoints, first_descriptors = detector.detectAndCompute(
        first_gray,
        None,
    )
    second_keypoints, second_descriptors = detector.detectAndCompute(
        second_gray,
        None,
    )

    if (
        first_descriptors is None
        or second_descriptors is None
        or not first_keypoints
        or not second_keypoints
    ):
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )
    pairs = matcher.knnMatch(
        first_descriptors,
        second_descriptors,
        k=2,
    )

    good = [
        first_match
        for pair in pairs
        if len(pair) == 2
        for first_match, second_match in [pair]
        if first_match.distance < 0.75 * second_match.distance
    ]

    normalizer = max(
        20.0,
        min(len(first_keypoints), len(second_keypoints)) * 0.18,
    )
    return float(
        max(0.0, min(1.0, len(good) / normalizer))
    )


def compare_images(
    candidate: np.ndarray,
    references: list[np.ndarray],
) -> tuple[float, float, float, float, float]:
    candidate_small = crop_comparison_region(
        resize_for_comparison(candidate)
    )

    per_reference: list[
        tuple[float, float, float, float, float]
    ] = []

    for reference in references:
        reference_small = crop_comparison_region(
            resize_for_comparison(reference)
        )

        feature = feature_similarity(
            candidate_small,
            reference_small,
        )
        edge = edge_similarity(
            candidate_small,
            reference_small,
        )
        hash_score = hash_similarity(
            candidate_small,
            reference_small,
        )
        histogram = histogram_similarity(
            candidate_small,
            reference_small,
        )

        total = (
            feature * 0.52
            + edge * 0.23
            + hash_score * 0.17
            + histogram * 0.08
        )

        per_reference.append(
            (
                total,
                feature,
                edge,
                hash_score,
                histogram,
            )
        )

    # Consecutive reference frames show nearly the same location.
    # Use the strongest match while still benefiting from both frames.
    per_reference.sort(reverse=True)
    return per_reference[0]


def transform_to_dict(
    transform: carla.Transform,
) -> dict[str, object]:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def make_top_matches_sheet(
    results: list[MatchResult],
    output_path: Path,
) -> None:
    cells: list[np.ndarray] = []

    for display_rank, result in enumerate(results, start=1):
        frame = cv2.resize(
            result.frame,
            (640, 360),
            interpolation=cv2.INTER_AREA,
        )
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (640, 42),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(
            overlay,
            0.62,
            frame,
            0.38,
            0.0,
            frame,
        )
        cv2.putText(
            frame,
            (
                f"Match {display_rank} | spawn {result.spawn_index} "
                f"| score {result.score:.3f}"
            ),
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(frame)

    while len(cells) < 3:
        cells.append(
            np.zeros((360, 640, 3), dtype=np.uint8)
        )

    sheet = cv2.vconcat(cells[:3])
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Generated dataset root not found: {dataset_root}"
        )

    weather_inventory: dict[str, list[Path]] = {
        weather: discover_images(dataset_root / weather)
        for weather in WEATHER_FOLDERS
    }

    clear_images = weather_inventory["clear"]

    if not clear_images:
        raise RuntimeError(
            f"No reference images found in: "
            f"{dataset_root / 'clear'}"
        )

    reference_paths = evenly_spaced_paths(
        clear_images,
        args.max_clear_references,
    )

    references: list[np.ndarray] = []

    for path in reference_paths:
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read reference: {path}"
            )

        references.append(image)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    world = client.get_world()
    active_map = world.get_map().name

    if not active_map.endswith(f"/{args.town}"):
        raise RuntimeError(
            f"Active map is {active_map}; expected {args.town}."
        )

    ranked = rank_spawn_points(world)

    if args.scan_count > 0:
        ranked = ranked[: args.scan_count]

    if not ranked:
        raise RuntimeError("No spawn points selected for scanning.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_root.resolve()
        / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 92)
    print("LOCK TOWN10 REFERENCE ROUTE FROM GENERATED DATASET")
    print("=" * 92)
    print(f"Map:          {active_map}")
    print(f"Dataset root: {dataset_root}")
    print(
        "Images:       "
        + ", ".join(
            f"{weather}={len(paths)}"
            for weather, paths in weather_inventory.items()
        )
    )
    print(f"Clear refs:   {len(references)}")
    print(f"Spawn scans:  {len(ranked)}")
    print(f"Camera:       {args.width}x{args.height}")
    print("Method:       genuine sensor.camera.rgb visual matching")
    print("=" * 92)

    original_weather = world.get_weather()
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=0.0,
            precipitation_deposits=20.0,
            wetness=40.0,
            wind_intensity=5.0,
            fog_density=0.0,
            fog_distance=100.0,
            sun_altitude_angle=45.0,
        )
    )

    library = world.get_blueprint_library()
    vehicle_bp = library.find("vehicle.tesla.model3")

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute(
            "role_name",
            "route_matcher",
        )

    first_transform = ranked[0][2]
    ego: Optional[carla.Vehicle] = world.try_spawn_actor(
        vehicle_bp,
        first_transform,
    )

    if ego is None:
        for _, _, transform, _ in ranked[1:]:
            ego = world.try_spawn_actor(
                vehicle_bp,
                transform,
            )
            if ego is not None:
                break

    if ego is None:
        raise RuntimeError(
            "Unable to spawn the route-matching Tesla."
        )

    ego.set_simulate_physics(False)

    camera_bp = library.find("sensor.camera.rgb")
    camera_bp.set_attribute(
        "image_size_x",
        str(args.width),
    )
    camera_bp.set_attribute(
        "image_size_y",
        str(args.height),
    )
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute(
        "sensor_tick",
        f"{1.0 / args.sensor_fps:.6f}",
    )

    frame_queue: queue.Queue[
        tuple[int, np.ndarray]
    ] = queue.Queue(maxsize=4)

    def callback(image: carla.Image) -> None:
        item = (int(image.frame), image_to_bgr(image))

        try:
            frame_queue.put_nowait(item)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                pass

    camera: Optional[carla.Sensor] = world.spawn_actor(
        camera_bp,
        carla.Transform(
            carla.Location(x=1.5, z=2.4)
        ),
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )
    camera.listen(callback)

    results: list[MatchResult] = []

    try:
        # Warm up the camera before the first teleport.
        warmup_deadline = time.perf_counter() + 10.0

        while frame_queue.empty():
            if time.perf_counter() > warmup_deadline:
                raise RuntimeError(
                    "RGB camera did not produce a warm-up frame."
                )
            time.sleep(0.1)

        for scan_rank, (
            geometry_score,
            spawn_index,
            transform,
            waypoint,
        ) in enumerate(ranked, start=1):
            while True:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break

            ego.set_transform(transform)
            time.sleep(args.settle_seconds)

            latest_frame: Optional[np.ndarray] = None
            capture_deadline = time.perf_counter() + 4.0

            while time.perf_counter() < capture_deadline:
                try:
                    _, latest_frame = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Use one additional frame after asset streaming settles.
                try:
                    _, latest_frame = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    pass
                break

            if latest_frame is None:
                raise RuntimeError(
                    f"No RGB frame for spawn index {spawn_index}."
                )

            (
                total_score,
                feature_score,
                edge_score,
                hash_score,
                histogram_score,
            ) = compare_images(
                latest_frame,
                references,
            )

            results.append(
                MatchResult(
                    rank=scan_rank,
                    spawn_index=spawn_index,
                    road_id=int(waypoint.road_id),
                    lane_id=int(waypoint.lane_id),
                    score=total_score,
                    feature_score=feature_score,
                    edge_score=edge_score,
                    hash_score=hash_score,
                    histogram_score=histogram_score,
                    transform=transform,
                    frame=latest_frame.copy(),
                )
            )

            print(
                f"Scanned {scan_rank:02d}/{len(ranked)} | "
                f"spawn={spawn_index:03d} | "
                f"match={total_score:.3f}"
            )

            # Small rest limits sustained GPU pressure during map streaming.
            if scan_rank % 10 == 0:
                time.sleep(1.0)

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )
        top_results = results[:3]

        for display_rank, result in enumerate(
            top_results,
            start=1,
        ):
            cv2.imwrite(
                str(
                    run_dir
                    / (
                        f"match_{display_rank:02d}_"
                        f"spawn_{result.spawn_index:03d}.png"
                    )
                ),
                result.frame,
            )

        best = top_results[0]
        cv2.imwrite(
            str(run_dir / "best_match.png"),
            best.frame,
        )
        make_top_matches_sheet(
            top_results,
            run_dir / "top_3_matches.jpg",
        )

        payload = {
            "generated_at": datetime.now().isoformat(),
            "status": "REFERENCE_ROUTE_LOCKED",
            "map": active_map,
            "dataset_root": str(dataset_root),
            "weather_inventory": {
                weather: {
                    "folder": str(dataset_root / weather),
                    "image_count": len(paths),
                }
                for weather, paths in weather_inventory.items()
            },
            "clear_reference_images": [
                str(path)
                for path in reference_paths
            ],
            "scan_count": len(ranked),
            "selected": {
                "spawn_index": best.spawn_index,
                "road_id": best.road_id,
                "lane_id": best.lane_id,
                "match_score": best.score,
                "feature_score": best.feature_score,
                "edge_score": best.edge_score,
                "hash_score": best.hash_score,
                "histogram_score": best.histogram_score,
                "transform": transform_to_dict(
                    best.transform
                ),
            },
            "top_matches": [
                {
                    "display_rank": rank,
                    "spawn_index": result.spawn_index,
                    "road_id": result.road_id,
                    "lane_id": result.lane_id,
                    "match_score": result.score,
                    "feature_score": result.feature_score,
                    "edge_score": result.edge_score,
                    "hash_score": result.hash_score,
                    "histogram_score": result.histogram_score,
                    "transform": transform_to_dict(
                        result.transform
                    ),
                }
                for rank, result in enumerate(
                    top_results,
                    start=1,
                )
            ],
            "locked_camera": {
                "blueprint": "sensor.camera.rgb",
                "x": 1.5,
                "y": 0.0,
                "z": 2.4,
                "fov": 90.0,
            },
        }

        json_path = run_dir / "locked_town10_reference_route.json"
        json_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        print("=" * 92)
        print("FINAL STATUS: PASS")
        print(
            f"Best matching spawn: {best.spawn_index}"
        )
        print(f"Match score:         {best.score:.3f}")
        print(f"Best frame:          {run_dir / 'best_match.png'}")
        print(f"Locked route:        {json_path}")
        print("=" * 92)

    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

        if camera is not None:
            try:
                camera.destroy()
            except RuntimeError:
                pass

        if ego is not None:
            try:
                ego.destroy()
            except RuntimeError:
                pass

        try:
            world.set_weather(original_weather)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()


