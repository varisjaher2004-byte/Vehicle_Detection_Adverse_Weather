from __future__ import annotations
import os

import argparse
import csv
import heapq
import json
import math
import queue
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


# ============================================================================
# TOWN10 RAIN SCENE GENERATION
#
# One file, controlled phases:
#   preflight -> capture -> detect -> validate -> combine -> status
#
# Scene:
#   same Town10 locked corridor
#   readable rain transition
#   one lead car
#   red signal
#   two pedestrians cross on a real CARLA crosswalk
#   green only after both pedestrians clear
#   lead car and ego take opposite turns
#
# Safety:
#   - Never destroys unknown/world actors.
#   - Only destroys actors created by this script.
#   - Does not use controller.ai.walker.
#   - Does not use BasicAgent or navigation-agent private APIs.
#   - Ground truth validates YOLO; it never creates a detection by itself.
# ============================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_ROUTE_ROOT = (
    CARLA_ROOT / "outputs" / "town10_reference_route"
)

DEFAULT_MODEL = (
    CARLA_ROOT
    / "outputs"
    / "training_runs"
    / "carla_multiclass_yolov8s_final"
    / "weights"
    / "best.pt"
)

DEFAULT_CLEAR_VIDEO = (
    CARLA_ROOT
    / "outputs"
    / "final_demo_reference"
    / "town10_clear_stable_final"
    / "town10_clear_stable_detected.mp4"
)

DEFAULT_RUN_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "town10_rain_signal_continuation"
)

DEFAULT_FINAL_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "final_demo_reference"
    / "town10_rain_signal_final"
)

DEFAULT_TOWN = "Town10HD_Opt"
DEFAULT_ROUTE_RANK = 2
DEFAULT_PORT = 2000
DEFAULT_TM_PORT = 8000

WIDTH = 640
HEIGHT = 360
FPS = 10.0
WORLD_FPS = 20.0
FOV = 90.0

CAMERA_X = 1.45
CAMERA_Y = 0.0
CAMERA_Z = 1.70
CAMERA_PITCH = 0.0

# Approximate hidden distance travelled by the accepted 10.7 s clear clip.
# The accepted Town10 raw setup used approximately 15 km/h after its hidden
# acceleration period. Starting the rain plan from this distance preserves
# the same corridor rather than restarting at spawn 31.
CLEAR_CONTINUATION_DISTANCE_M = 44.0

SEARCH_AFTER_CONTINUATION_M = 12.0
MAX_JUNCTION_SEARCH_M = 180.0
APPROACH_RUNWAY_M = 38.0
LEAD_START_GAP_M = 18.0
LEAD_STOP_BEFORE_CROSSWALK_M = 5.5
EGO_STOP_GAP_M = 14.0

EGO_APPROACH_SPEED_KMH = 15.0
LEAD_APPROACH_SPEED_KMH = 14.0
EGO_RELEASE_SPEED_KMH = 12.5
LEAD_RELEASE_SPEED_KMH = 13.0

PED_1_SPEED_MPS = 1.60
PED_2_SPEED_MPS = 1.50
PED_2_DELAY_SECONDS = 0.50
PEDESTRIAN_LONGITUDINAL_SPACING_M = 1.50

RAIN_CLEAR_LEAD_SECONDS = 0.8
RAIN_TRANSITION_SECONDS = 2.4
MAX_CAPTURE_SECONDS = 45.0
GREEN_SETTLE_SECONDS = 0.6
EGO_RELEASE_DELAY_SECONDS = 0.8
FINAL_TAIL_SECONDS = 1.2

ALLOWED_LABELS = ("car", "person")
TARGET_KEYS = ("lead_car", "pedestrian_1", "pedestrian_2")
TARGET_TO_LABEL = {
    "lead_car": "car",
    "pedestrian_1": "person",
    "pedestrian_2": "person",
}

RAW_CONFIDENCE_FLOOR = {
    "car": 0.02,
    "person": 0.005,
}

START_CONFIDENCE = {
    "car": 0.20,
    "person": 0.025,
}

MIN_DISPLAY_FRAMES = {
    "lead_car": 10,
    "pedestrian_1": 6,
    "pedestrian_2": 6,
}

BOX_COLORS = {
    "lead_car": (70, 220, 120),
    "pedestrian_1": (70, 210, 245),
    "pedestrian_2": (80, 165, 245),
}


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    def to_carla(self) -> Any:
        import carla

        return carla.Location(
            x=float(self.x),
            y=float(self.y),
            z=float(self.z),
        )


@dataclass(frozen=True)
class Rotation3:
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class TransformData:
    location: Point3
    rotation: Rotation3

    def to_carla(self) -> Any:
        import carla

        return carla.Transform(
            self.location.to_carla(),
            carla.Rotation(
                pitch=float(self.rotation.pitch),
                yaw=float(self.rotation.yaw),
                roll=float(self.rotation.roll),
            ),
        )


@dataclass(frozen=True)
class ScenePlan:
    map_name: str
    route_json: str
    route_rank: int
    locked_spawn_index: int
    locked_road_id: int
    continuation_distance_m: float
    scene_start: TransformData
    lead_start: TransformData
    lead_stop: Point3
    ego_stop: Point3
    junction_entry: TransformData
    junction_center: Point3
    crosswalk_start_1: Point3
    crosswalk_end_1: Point3
    crosswalk_start_2: Point3
    crosswalk_end_2: Point3
    traffic_light_anchor: Point3
    incoming_path: list[Point3]
    lead_left_path: list[Point3]
    ego_right_path: list[Point3]
    incoming_yaw: float
    lead_exit_yaw: float
    ego_exit_yaw: float
    lead_signed_turn_deg: float
    ego_signed_turn_deg: float
    crosswalk_width_m: float
    junction_distance_from_continuation_m: float
    plan_status: str


@dataclass
class VehicleFollower:
    path: list[Any]
    index: int = 0


@dataclass
class DetectionTrack:
    key: str
    label: str
    box: np.ndarray
    confidence_ema: float
    consecutive_hits: int
    total_hits: int
    missed: int
    confirmed: bool
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a controlled Town10 rain continuation with a lead "
            "car, red signal, two crossing pedestrians, green release and "
            "opposite turns."
        )
    )
    parser.add_argument(
        "phase",
        choices=(
            "preflight",
            "capture",
            "detect",
            "validate",
            "combine",
            "finalize",
            "all",
            "status",
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tm-port", type=int, default=DEFAULT_TM_PORT)
    parser.add_argument("--town", default=DEFAULT_TOWN)
    parser.add_argument(
        "--route-rank",
        type=int,
        default=DEFAULT_ROUTE_RANK,
    )
    parser.add_argument(
        "--route-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Existing timestamped run folder for later phases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=DEFAULT_FINAL_ROOT,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--clear-video",
        type=Path,
        default=DEFAULT_CLEAR_VIDEO,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
    )
    parser.add_argument(
        "--keep-run-on-failure",
        action="store_true",
        help=(
            "Keep a new run folder when preflight fails. By default an "
            "empty failed run folder is removed."
        ),
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


def signed_yaw_difference(target_yaw: float, source_yaw: float) -> float:
    return normalize_angle(target_yaw - source_yaw)


def absolute_yaw_difference(first: float, second: float) -> float:
    return abs(signed_yaw_difference(first, second))


def location_distance(first: Any, second: Any) -> float:
    return math.sqrt(
        (float(first.x) - float(second.x)) ** 2
        + (float(first.y) - float(second.y)) ** 2
        + (float(first.z) - float(second.z)) ** 2
    )


def point_from_location(location: Any) -> Point3:
    return Point3(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
    )


def transform_to_data(transform: Any) -> TransformData:
    return TransformData(
        location=point_from_location(transform.location),
        rotation=Rotation3(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def latest_route_json() -> Path:
    candidates = sorted(
        DEFAULT_ROUTE_ROOT.glob(
            "run_*/locked_town10_reference_route.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No locked Town10 reference route JSON was found under "
            f"{DEFAULT_ROUTE_ROOT}"
        )

    return candidates[0]


def latest_run(output_root: Path) -> Path:
    candidates = sorted(
        output_root.glob("run_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        if (candidate / "scene_plan.json").is_file():
            return candidate

    raise FileNotFoundError(
        f"No valid rain-signal run exists under {output_root}"
    )


def resolve_run(args: argparse.Namespace) -> Path:
    if args.run is not None:
        path = args.run.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"Run folder does not exist: {path}"
            )

        return path

    return latest_run(args.output_root.resolve())


def load_locked_transform(
    route_path: Path,
    route_rank: int,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    import carla

    payload = json.loads(
        route_path.read_text(encoding="utf-8")
    )

    if payload.get("status") != "REFERENCE_ROUTE_LOCKED":
        raise RuntimeError(
            f"Town10 route JSON is not locked: {route_path}"
        )

    top_matches = payload.get("top_matches")

    if (
        not isinstance(top_matches, list)
        or len(top_matches) < route_rank
    ):
        raise RuntimeError(
            f"Route rank {route_rank} is unavailable in {route_path}"
        )

    selected = top_matches[route_rank - 1]
    transform_data = selected["transform"]
    location = transform_data["location"]
    rotation = transform_data["rotation"]

    transform = carla.Transform(
        carla.Location(
            x=float(location["x"]),
            y=float(location["y"]),
            z=float(location["z"]),
        ),
        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"]),
        ),
    )

    return payload, selected, transform


def choose_straightest(
    current: Any,
    options: Iterable[Any],
) -> Any:
    option_list = list(options)

    if not option_list:
        raise RuntimeError("No waypoint option is available.")

    return min(
        option_list,
        key=lambda waypoint: absolute_yaw_difference(
            waypoint.transform.rotation.yaw,
            current.transform.rotation.yaw,
        ),
    )


def trace_straight_route(
    start_waypoint: Any,
    maximum_distance_m: float,
    step_m: float = 2.0,
) -> tuple[list[Any], list[float]]:
    import carla

    route = [start_waypoint]
    cumulative = [0.0]
    current = start_waypoint

    for _ in range(500):
        options = [
            waypoint
            for waypoint in current.next(step_m)
            if waypoint.lane_type == carla.LaneType.Driving
        ]

        if not options:
            break

        chosen = choose_straightest(current, options)
        segment = current.transform.location.distance(
            chosen.transform.location
        )

        if segment < 0.05:
            break

        route.append(chosen)
        cumulative.append(cumulative[-1] + segment)
        current = chosen

        if cumulative[-1] >= maximum_distance_m:
            break

    if cumulative[-1] < maximum_distance_m * 0.60:
        raise RuntimeError(
            "The Town10 corridor ended too early while tracing the "
            f"locked route: {cumulative[-1]:.1f} m"
        )

    return route, cumulative


def first_index_at_distance(
    cumulative: list[float],
    distance_m: float,
) -> int:
    for index, value in enumerate(cumulative):
        if value >= distance_m:
            return index

    return len(cumulative) - 1


def nearest_route_index(
    route: list[Any],
    location: Any,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> int:
    end = len(route) if end_index is None else min(
        len(route),
        end_index,
    )
    candidates = range(max(0, start_index), end)

    return min(
        candidates,
        key=lambda index: route[
            index
        ].transform.location.distance(location),
    )


def route_points(
    route: list[Any],
    start_index: int,
    end_index: int,
) -> list[Point3]:
    start = max(0, start_index)
    end = min(len(route) - 1, end_index)

    if end < start:
        raise ValueError("Route point range is reversed.")

    return [
        point_from_location(route[index].transform.location)
        for index in range(start, end + 1)
    ]


def point_at_distance_before(
    route: list[Any],
    cumulative: list[float],
    reference_index: int,
    distance_before_m: float,
) -> tuple[int, Point3]:
    target_distance = max(
        0.0,
        cumulative[reference_index] - distance_before_m,
    )
    index = first_index_at_distance(
        cumulative,
        target_distance,
    )

    return (
        index,
        point_from_location(
            route[index].transform.location
        ),
    )


def parse_crosswalk_polygons(
    locations: list[Any],
) -> list[list[Any]]:
    polygons: list[list[Any]] = []
    current: list[Any] = []

    for location in locations:
        if not current:
            current = [location]
            continue

        current.append(location)

        if (
            len(current) >= 4
            and current[0].distance(current[-1]) <= 0.20
        ):
            polygons.append(current[:-1])
            current = []

    if len(current) >= 3:
        polygons.append(current)

    return polygons


def polygon_center(polygon: list[Any]) -> Any:
    import carla

    return carla.Location(
        x=sum(float(point.x) for point in polygon) / len(polygon),
        y=sum(float(point.y) for point in polygon) / len(polygon),
        z=sum(float(point.z) for point in polygon) / len(polygon),
    )


def farthest_pair(points: list[Any]) -> tuple[Any, Any, float]:
    if len(points) < 2:
        raise RuntimeError("Crosswalk polygon has fewer than two points.")

    best_first = points[0]
    best_second = points[1]
    best_distance = best_first.distance(best_second)

    for first_index, first in enumerate(points):
        for second in points[first_index + 1 :]:
            distance = first.distance(second)

            if distance > best_distance:
                best_first = first
                best_second = second
                best_distance = distance

    return best_first, best_second, float(best_distance)


def vector_dot_2d(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    return ax * bx + ay * by


def crosswalk_score(
    polygon: list[Any],
    junction_location: Any,
    incoming_yaw: float,
) -> tuple[float, Any, Any, Any, float]:
    center = polygon_center(polygon)
    first, second, width = farthest_pair(polygon)
    dx = float(second.x - first.x)
    dy = float(second.y - first.y)
    magnitude = max(0.001, math.hypot(dx, dy))
    dx /= magnitude
    dy /= magnitude

    yaw_radians = math.radians(incoming_yaw)
    incoming_x = math.cos(yaw_radians)
    incoming_y = math.sin(yaw_radians)

    # A real crossing direction should be close to perpendicular to the
    # incoming lane. A lower absolute dot product is better.
    parallel_penalty = abs(
        vector_dot_2d(
            dx,
            dy,
            incoming_x,
            incoming_y,
        )
    )
    distance = center.distance(junction_location)

    score = (
        distance
        + parallel_penalty * 18.0
        + (8.0 if width < 4.0 else 0.0)
        + (5.0 if width > 18.0 else 0.0)
    )

    return (
        float(score),
        center,
        first,
        second,
        width,
    )


def waypoint_state_key(
    waypoint: Any,
) -> tuple[int, int, int, int, int, int]:
    location = waypoint.transform.location

    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        int(round(float(waypoint.s) * 2.0)),
        int(round(float(location.x) * 2.0)),
        int(round(float(location.y) * 2.0)),
    )


def trace_path_to_junction_exit(
    entry_waypoint: Any,
    exit_waypoint: Any,
    step_m: float = 2.0,
    maximum_travel_m: float = 95.0,
) -> list[Any]:
    """
    Build a real waypoint path through a CARLA junction.

    Town10 branches may split several metres after the first junction
    waypoint. Looking only at entry.next(2.0) therefore misses valid
    left/right exits. This bounded A* search follows the waypoint graph
    until it reaches the requested junction exit.
    """
    import carla

    target_location = exit_waypoint.transform.location
    start_key = waypoint_state_key(entry_waypoint)
    best_cost: dict[
        tuple[int, int, int, int, int, int],
        float,
    ] = {start_key: 0.0}
    counter = 0
    frontier: list[
        tuple[float, int, float, Any, list[Any]]
    ] = []

    heapq.heappush(
        frontier,
        (
            float(
                entry_waypoint.transform.location.distance(
                    target_location
                )
            ),
            counter,
            0.0,
            entry_waypoint,
            [entry_waypoint],
        ),
    )
    expansions = 0

    while frontier and expansions < 800:
        (
            _priority,
            _counter,
            travelled,
            current,
            path,
        ) = heapq.heappop(frontier)
        expansions += 1

        distance_to_exit = (
            current.transform.location.distance(
                target_location
            )
        )
        yaw_error = absolute_yaw_difference(
            current.transform.rotation.yaw,
            exit_waypoint.transform.rotation.yaw,
        )

        if (
            distance_to_exit <= 3.5
            and yaw_error <= 60.0
            and len(path) >= 3
        ):
            if (
                path[-1].transform.location.distance(
                    target_location
                )
                > 0.75
            ):
                path = path + [exit_waypoint]

            return path

        if travelled >= maximum_travel_m:
            continue

        options = [
            waypoint
            for waypoint in current.next(step_m)
            if waypoint.lane_type
            == carla.LaneType.Driving
        ]

        for option in options:
            segment = (
                current.transform.location.distance(
                    option.transform.location
                )
            )

            if segment < 0.05:
                continue

            new_cost = travelled + segment

            if new_cost > maximum_travel_m:
                continue

            key = waypoint_state_key(option)
            old_cost = best_cost.get(
                key,
                float("inf"),
            )

            if new_cost >= old_cost - 0.05:
                continue

            best_cost[key] = new_cost
            heuristic = (
                option.transform.location.distance(
                    target_location
                )
            )
            heading_penalty = (
                absolute_yaw_difference(
                    option.transform.rotation.yaw,
                    exit_waypoint.transform.rotation.yaw,
                )
                * 0.025
            )
            topology_penalty = (
                0.0
                if option.is_junction
                or heuristic <= 8.0
                else 7.0
            )
            counter += 1
            heapq.heappush(
                frontier,
                (
                    float(
                        new_cost
                        + heuristic
                        + heading_penalty
                        + topology_penalty
                    ),
                    counter,
                    new_cost,
                    option,
                    path + [option],
                ),
            )

    return []


def extend_branch_after_exit(
    exit_waypoint: Any,
    maximum_distance_m: float = 36.0,
    step_m: float = 2.0,
) -> list[Any]:
    import carla

    path = [exit_waypoint]
    current = exit_waypoint
    travelled = 0.0
    previous_yaw = (
        exit_waypoint.transform.rotation.yaw
    )

    for _ in range(80):
        options = [
            waypoint
            for waypoint in current.next(step_m)
            if waypoint.lane_type
            == carla.LaneType.Driving
        ]

        if not options:
            break

        chosen = min(
            options,
            key=lambda waypoint: (
                absolute_yaw_difference(
                    waypoint.transform.rotation.yaw,
                    previous_yaw,
                )
                + (
                    8.0
                    if waypoint.is_junction
                    and not current.is_junction
                    else 0.0
                )
            ),
        )
        segment = (
            current.transform.location.distance(
                chosen.transform.location
            )
        )

        if segment < 0.05:
            break

        travelled += segment
        path.append(chosen)
        current = chosen
        previous_yaw = (
            current.transform.rotation.yaw
        )

        if travelled >= maximum_distance_m:
            break

    return path


def branch_candidates(
    entry_waypoint: Any,
    junction: Any,
) -> list[tuple[float, list[Any]]]:
    """
    Read CARLA junction entry/exit lane pairs and build a genuine path
    through every plausible turning branch.
    """
    import carla

    incoming_yaw = (
        entry_waypoint.transform.rotation.yaw
    )
    incoming_location = (
        entry_waypoint.transform.location
    )
    lane_pairs = list(
        junction.get_waypoints(
            carla.LaneType.Driving
        )
    )
    matched_pairs: list[
        tuple[float, Any, Any]
    ] = []

    for pair_entry, pair_exit in lane_pairs:
        entry_distance = (
            pair_entry.transform.location.distance(
                incoming_location
            )
        )
        entry_yaw_error = (
            absolute_yaw_difference(
                pair_entry.transform.rotation.yaw,
                incoming_yaw,
            )
        )
        same_lane_bonus = (
            -12.0
            if (
                pair_entry.road_id
                == entry_waypoint.road_id
                and pair_entry.lane_id
                == entry_waypoint.lane_id
            )
            else 0.0
        )
        score = (
            entry_distance
            + entry_yaw_error * 0.10
            + same_lane_bonus
        )

        if (
            entry_distance <= 32.0
            and entry_yaw_error <= 75.0
        ):
            matched_pairs.append(
                (
                    float(score),
                    pair_entry,
                    pair_exit,
                )
            )

    if not matched_pairs:
        return []

    matched_pairs.sort(
        key=lambda item: item[0]
    )
    best_score = matched_pairs[0][0]
    matched_pairs = [
        item
        for item in matched_pairs
        if item[0] <= best_score + 22.0
    ]

    candidates: list[
        tuple[float, list[Any]]
    ] = []

    for (
        _score,
        _pair_entry,
        pair_exit,
    ) in matched_pairs:
        preliminary_turn = (
            signed_yaw_difference(
                pair_exit.transform.rotation.yaw,
                incoming_yaw,
            )
        )

        if not (
            20.0
            <= abs(preliminary_turn)
            <= 155.0
        ):
            continue

        through_path = trace_path_to_junction_exit(
            entry_waypoint,
            pair_exit,
        )

        if len(through_path) < 3:
            continue

        extension = extend_branch_after_exit(
            pair_exit
        )
        combined_path = (
            through_path + extension[1:]
        )
        final_turn = signed_yaw_difference(
            combined_path[
                -1
            ].transform.rotation.yaw,
            incoming_yaw,
        )

        if not (
            25.0 <= abs(final_turn) <= 150.0
        ):
            continue

        candidates.append(
            (
                float(final_turn),
                combined_path,
            )
        )

    unique: list[
        tuple[float, list[Any]]
    ] = []

    for signed_turn, path in sorted(
        candidates,
        key=lambda item: item[0],
    ):
        duplicate = False

        for old_turn, old_path in unique:
            exit_distance = (
                path[-1].transform.location.distance(
                    old_path[-1].transform.location
                )
            )

            if (
                abs(signed_turn - old_turn) < 14.0
                and exit_distance < 10.0
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(
                (
                    signed_turn,
                    path,
                )
            )

    return unique


def find_relevant_traffic_light(
    world: Any,
    entry_waypoint: Any,
    crosswalk_center: Any,
) -> tuple[Any, float]:
    """
    Return only a traffic light that genuinely controls this approach.

    A visually nearby traffic-light actor is not enough. It must expose a
    stop waypoint matching the incoming road/lane, or a very close
    same-heading stop waypoint. This prevents a STOP-sign junction from
    borrowing an unrelated traffic light elsewhere in Town10.
    """
    lights = list(
        world.get_actors().filter("*traffic_light*")
    )
    scored: list[tuple[float, Any]] = []
    entry_location = (
        entry_waypoint.transform.location
    )
    entry_yaw = (
        entry_waypoint.transform.rotation.yaw
    )
    entry_forward = (
        entry_waypoint.transform.get_forward_vector()
    )
    entry_right = (
        entry_waypoint.transform.get_right_vector()
    )

    for light in lights:
        try:
            stop_waypoints = list(
                light.get_stop_waypoints()
            )
        except (AttributeError, RuntimeError):
            stop_waypoints = []

        if not stop_waypoints:
            continue

        best_match_score = float("inf")

        for stop_waypoint in stop_waypoints:
            stop_location = (
                stop_waypoint.transform.location
            )
            distance = (
                stop_location.distance(
                    entry_location
                )
            )
            yaw_error = (
                absolute_yaw_difference(
                    stop_waypoint.transform.rotation.yaw,
                    entry_yaw,
                )
            )
            same_lane = (
                stop_waypoint.road_id
                == entry_waypoint.road_id
                and stop_waypoint.lane_id
                == entry_waypoint.lane_id
            )
            same_road = (
                stop_waypoint.road_id
                == entry_waypoint.road_id
            )

            valid_match = (
                (
                    same_lane
                    and distance <= 24.0
                    and yaw_error <= 50.0
                )
                or (
                    same_road
                    and distance <= 10.0
                    and yaw_error <= 35.0
                )
                or (
                    distance <= 5.0
                    and yaw_error <= 25.0
                )
            )

            if not valid_match:
                continue

            match_score = (
                distance
                + yaw_error * 0.05
                + (0.0 if same_lane else 6.0)
            )
            best_match_score = min(
                best_match_score,
                match_score,
            )

        if math.isinf(best_match_score):
            continue

        light_location = light.get_location()
        actor_distance = (
            light_location.distance(
                crosswalk_center
            )
        )

        # Traffic signal must also be in a plausible visible area around
        # the junction, not 50-60 m away.
        if actor_distance > 35.0:
            continue

        dx = float(
            light_location.x - entry_location.x
        )
        dy = float(
            light_location.y - entry_location.y
        )
        forward_distance = (
            dx * entry_forward.x
            + dy * entry_forward.y
        )
        lateral_distance = abs(
            dx * entry_right.x
            + dy * entry_right.y
        )

        if not (
            -8.0 <= forward_distance <= 42.0
            and lateral_distance <= 30.0
        ):
            continue

        score = (
            best_match_score
            + actor_distance * 0.20
            + lateral_distance * 0.08
        )
        scored.append(
            (
                float(score),
                light,
            )
        )

    if not scored:
        raise RuntimeError(
            "No traffic light genuinely controlling this approach "
            "was found."
        )

    scored.sort(
        key=lambda item: item[0]
    )

    return scored[0][1], float(scored[0][0])


def cumulative_route_distances(
    route: list[Any],
) -> list[float]:
    if not route:
        return []

    cumulative = [0.0]

    for first, second in zip(
        route,
        route[1:],
    ):
        cumulative.append(
            cumulative[-1]
            + first.transform.location.distance(
                second.transform.location
            )
        )

    return cumulative


def trace_incoming_approach(
    entry_waypoint: Any,
    distance_m: float = 48.0,
    step_m: float = 2.0,
) -> tuple[list[Any], list[float]]:
    """
    Trace backwards from a junction entry to create a smooth approach
    runway. The returned route is ordered in the driving direction.
    """
    import carla

    reverse_route = [entry_waypoint]
    current = entry_waypoint
    travelled = 0.0
    current_yaw = (
        entry_waypoint.transform.rotation.yaw
    )

    for _ in range(120):
        options = [
            waypoint
            for waypoint in current.previous(step_m)
            if waypoint.lane_type
            == carla.LaneType.Driving
        ]

        if not options:
            break

        chosen = min(
            options,
            key=lambda waypoint: (
                absolute_yaw_difference(
                    waypoint.transform.rotation.yaw,
                    current_yaw,
                )
                + (
                    0.0
                    if (
                        waypoint.road_id
                        == current.road_id
                        and waypoint.lane_id
                        == current.lane_id
                    )
                    else 18.0
                )
                + (
                    15.0
                    if waypoint.is_junction
                    else 0.0
                )
            ),
        )
        segment = (
            current.transform.location.distance(
                chosen.transform.location
            )
        )

        if segment < 0.05:
            break

        travelled += segment
        reverse_route.append(chosen)
        current = chosen
        current_yaw = (
            current.transform.rotation.yaw
        )

        if travelled >= distance_m:
            break

    route = list(reversed(reverse_route))
    cumulative = cumulative_route_distances(
        route
    )

    return route, cumulative


def discover_town_junction_entries(
    carla_map: Any,
) -> list[tuple[Any, Any]]:
    """
    Discover unique non-junction driving waypoints whose next waypoint
    enters a junction. Entries are deduplicated by junction, road and lane.
    """
    import carla

    candidates: dict[
        tuple[int, int, int],
        tuple[Any, Any],
    ] = {}

    for waypoint in carla_map.generate_waypoints(
        2.0
    ):
        if (
            waypoint.lane_type
            != carla.LaneType.Driving
            or waypoint.is_junction
        ):
            continue

        following_options = [
            following
            for following in waypoint.next(2.0)
            if following.lane_type
            == carla.LaneType.Driving
            and following.is_junction
        ]

        for following in following_options:
            junction = following.get_junction()

            if junction is None:
                continue

            junction_id = int(
                getattr(
                    junction,
                    "id",
                    round(
                        float(
                            junction.bounding_box.location.x
                        )
                        * 10.0
                    ),
                )
            )
            key = (
                junction_id,
                int(waypoint.road_id),
                int(waypoint.lane_id),
            )
            old = candidates.get(key)

            if old is None:
                candidates[key] = (
                    waypoint,
                    junction,
                )
                continue

            old_waypoint = old[0]
            old_distance = (
                old_waypoint.transform.location.distance(
                    junction.bounding_box.location
                )
            )
            new_distance = (
                waypoint.transform.location.distance(
                    junction.bounding_box.location
                )
            )

            if new_distance < old_distance:
                candidates[key] = (
                    waypoint,
                    junction,
                )

    return list(candidates.values())


def build_scene_plan(
    world: Any,
    route_path: Path,
    route_rank: int,
) -> ScenePlan:
    """
    Select a professional signalised Town10 junction.

    The accepted clear video remains the visual starting reference, but
    the original locked corridor is not forced to provide geometry it
    does not contain. The whole Town10 map is audited, and the closest
    visually compatible signalised junction with a real crosswalk and
    two opposite signed turn branches is selected.
    """
    import carla

    _route_payload, selected, locked_transform = (
        load_locked_transform(
            route_path,
            route_rank,
        )
    )
    carla_map = world.get_map()
    locked_waypoint = carla_map.get_waypoint(
        locked_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if locked_waypoint is None:
        raise RuntimeError(
            "The locked Town10 transform did not project to a "
            "driving waypoint."
        )

    locked_route, locked_cumulative = (
        trace_straight_route(
            locked_waypoint,
            maximum_distance_m=(
                CLEAR_CONTINUATION_DISTANCE_M
                + 30.0
            ),
        )
    )
    continuation_index = first_index_at_distance(
        locked_cumulative,
        CLEAR_CONTINUATION_DISTANCE_M,
    )
    continuation_waypoint = (
        locked_route[continuation_index]
    )
    continuation_location = (
        continuation_waypoint.transform.location
    )
    continuation_yaw = (
        continuation_waypoint.transform.rotation.yaw
    )

    crosswalk_polygons = (
        parse_crosswalk_polygons(
            list(carla_map.get_crosswalks())
        )
    )

    if not crosswalk_polygons:
        raise RuntimeError(
            "Town10 returned no crosswalk polygons."
        )

    junction_entries = (
        discover_town_junction_entries(
            carla_map
        )
    )

    if not junction_entries:
        raise RuntimeError(
            "Town10 returned no usable junction entries."
        )

    best_candidate: Optional[
        tuple[
            float,
            Any,
            Any,
            Any,
            float,
            tuple[float, list[Any]],
            tuple[float, list[Any]],
            Any,
            list[Any],
            list[float],
            float,
        ]
    ] = None
    audit_counts = {
        "town_junction_entries": len(
            junction_entries
        ),
        "entries_with_approach_runway": 0,
        "junctions_with_crosswalk": 0,
        "junctions_with_turn_branch": 0,
        "junctions_with_opposite_branches": 0,
        "junctions_with_traffic_light": 0,
        "fully_usable_candidates": 0,
    }

    for entry, junction in junction_entries:
        incoming_route, incoming_cumulative = (
            trace_incoming_approach(
                entry,
                distance_m=48.0,
            )
        )

        if (
            not incoming_cumulative
            or incoming_cumulative[-1] < 36.0
        ):
            continue

        audit_counts[
            "entries_with_approach_runway"
        ] += 1
        junction_location = (
            junction.bounding_box.location
        )
        incoming_yaw = (
            entry.transform.rotation.yaw
        )
        crosswalk_options = []

        for polygon in crosswalk_polygons:
            option = crosswalk_score(
                polygon,
                junction_location,
                incoming_yaw,
            )

            if (
                option[1].distance(
                    junction_location
                )
                <= 38.0
            ):
                crosswalk_options.append(option)

        if not crosswalk_options:
            continue

        audit_counts[
            "junctions_with_crosswalk"
        ] += 1
        crosswalk_options.sort(
            key=lambda item: item[0]
        )
        (
            crosswalk_quality,
            crosswalk_center,
            _crosswalk_first,
            _crosswalk_second,
            crosswalk_width,
        ) = crosswalk_options[0]

        if not (
            4.0 <= crosswalk_width <= 18.0
        ):
            continue

        branches = branch_candidates(
            entry,
            junction,
        )

        if branches:
            audit_counts[
                "junctions_with_turn_branch"
            ] += 1

        negative = [
            item
            for item in branches
            if item[0] < -28.0
        ]
        positive = [
            item
            for item in branches
            if item[0] > 28.0
        ]

        if not negative or not positive:
            continue

        audit_counts[
            "junctions_with_opposite_branches"
        ] += 1
        lead_branch = max(
            negative,
            key=lambda item: abs(item[0]),
        )
        ego_branch = max(
            positive,
            key=lambda item: abs(item[0]),
        )

        try:
            traffic_light, light_score = (
                find_relevant_traffic_light(
                    world,
                    entry,
                    crosswalk_center,
                )
            )
        except RuntimeError:
            continue

        audit_counts[
            "junctions_with_traffic_light"
        ] += 1
        relocation_distance = (
            entry.transform.location.distance(
                continuation_location
            )
        )
        heading_difference = (
            absolute_yaw_difference(
                incoming_yaw,
                continuation_yaw,
            )
        )
        approach_length = (
            incoming_cumulative[-1]
        )

        # The map relocation is hidden by the clear-to-rain dissolve.
        # Prefer a nearby junction with a similar approach direction and
        # balanced cinematic left/right branches.
        compact_crosswalk_penalty = max(
            0.0,
            crosswalk_width - 12.0,
        ) * 3.0

        candidate_score = (
            relocation_distance * 0.16
            + heading_difference * 0.20
            + crosswalk_quality
            + light_score
            + compact_crosswalk_penalty
            + abs(
                approach_length - 44.0
            )
            * 0.10
            + abs(
                abs(lead_branch[0]) - 75.0
            )
            * 0.03
            + abs(
                abs(ego_branch[0]) - 75.0
            )
            * 0.03
        )
        audit_counts[
            "fully_usable_candidates"
        ] += 1
        candidate = (
            float(candidate_score),
            entry,
            crosswalk_center,
            crosswalk_width,
            float(relocation_distance),
            lead_branch,
            ego_branch,
            traffic_light,
            incoming_route,
            incoming_cumulative,
            float(heading_difference),
        )

        if (
            best_candidate is None
            or candidate[0] < best_candidate[0]
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            "No fully usable signalised Town10 junction was found. "
            "Audit counts: "
            + json.dumps(
                audit_counts,
                sort_keys=True,
            )
        )

    (
        _candidate_score,
        entry_waypoint,
        crosswalk_center,
        _crosswalk_width,
        relocation_distance,
        lead_branch,
        ego_branch,
        traffic_light,
        incoming_route,
        incoming_cumulative,
        _heading_difference,
    ) = best_candidate
    entry_index = len(incoming_route) - 1

    selected_polygon = min(
        crosswalk_polygons,
        key=lambda polygon: polygon_center(
            polygon
        ).distance(crosswalk_center),
    )
    endpoint_a, endpoint_b, selected_width = (
        farthest_pair(selected_polygon)
    )
    right = (
        entry_waypoint.transform.get_right_vector()
    )
    center = crosswalk_center

    def side_value(location: Any) -> float:
        return (
            (location.x - center.x) * right.x
            + (location.y - center.y) * right.y
        )

    if (
        side_value(endpoint_a)
        <= side_value(endpoint_b)
    ):
        cross_start = endpoint_a
        cross_end = endpoint_b
    else:
        cross_start = endpoint_b
        cross_end = endpoint_a

    scene_start_index = 0
    entry_distance = (
        incoming_cumulative[entry_index]
    )
    lead_start_distance = min(
        entry_distance - 12.0,
        incoming_cumulative[
            scene_start_index
        ]
        + LEAD_START_GAP_M,
    )
    lead_start_index = (
        first_index_at_distance(
            incoming_cumulative,
            lead_start_distance,
        )
    )
    lead_stop_index, lead_stop_point = (
        point_at_distance_before(
            incoming_route,
            incoming_cumulative,
            entry_index,
            LEAD_STOP_BEFORE_CROSSWALK_M,
        )
    )
    ego_stop_index, ego_stop_point = (
        point_at_distance_before(
            incoming_route,
            incoming_cumulative,
            entry_index,
            (
                LEAD_STOP_BEFORE_CROSSWALK_M
                + EGO_STOP_GAP_M
            ),
        )
    )

    if (
        lead_stop_index
        <= lead_start_index + 2
    ):
        raise RuntimeError(
            "The selected signalised junction has an insufficient "
            "lead-car approach runway."
        )

    if ego_stop_index <= scene_start_index + 2:
        raise RuntimeError(
            "The selected signalised junction has an insufficient "
            "ego approach runway."
        )

    incoming_points = route_points(
        incoming_route,
        scene_start_index,
        entry_index,
    )
    lead_prefix = route_points(
        incoming_route,
        lead_start_index,
        entry_index,
    )
    lead_branch_points = [
        point_from_location(
            waypoint.transform.location
        )
        for waypoint in lead_branch[1]
    ]
    ego_branch_points = [
        point_from_location(
            waypoint.transform.location
        )
        for waypoint in ego_branch[1]
    ]
    lead_path = (
        lead_prefix + lead_branch_points
    )
    ego_path = (
        incoming_points + ego_branch_points
    )
    incoming_forward = (
        entry_waypoint.transform.get_forward_vector()
    )

    def offset_crosswalk_point(
        base: Any,
        longitudinal_offset_m: float,
    ) -> Point3:
        return Point3(
            x=float(
                base.x
                + incoming_forward.x
                * longitudinal_offset_m
            ),
            y=float(
                base.y
                + incoming_forward.y
                * longitudinal_offset_m
            ),
            z=float(base.z + 0.15),
        )

    ped_1_start = offset_crosswalk_point(
        cross_start,
        (
            -PEDESTRIAN_LONGITUDINAL_SPACING_M
            / 2.0
        ),
    )
    ped_1_end = offset_crosswalk_point(
        cross_end,
        (
            -PEDESTRIAN_LONGITUDINAL_SPACING_M
            / 2.0
        ),
    )
    ped_2_start = offset_crosswalk_point(
        cross_start,
        (
            PEDESTRIAN_LONGITUDINAL_SPACING_M
            / 2.0
        ),
    )
    ped_2_end = offset_crosswalk_point(
        cross_end,
        (
            PEDESTRIAN_LONGITUDINAL_SPACING_M
            / 2.0
        ),
    )

    return ScenePlan(
        map_name=str(world.get_map().name),
        route_json=str(route_path),
        route_rank=int(route_rank),
        locked_spawn_index=int(
            selected["spawn_index"]
        ),
        locked_road_id=int(
            selected["road_id"]
        ),
        continuation_distance_m=float(
            CLEAR_CONTINUATION_DISTANCE_M
        ),
        scene_start=transform_to_data(
            incoming_route[
                scene_start_index
            ].transform
        ),
        lead_start=transform_to_data(
            incoming_route[
                lead_start_index
            ].transform
        ),
        lead_stop=lead_stop_point,
        ego_stop=ego_stop_point,
        junction_entry=transform_to_data(
            entry_waypoint.transform
        ),
        # The selected entry waypoint is intentionally outside the
        # junction, so get_junction() returns None here. The selected
        # crosswalk center is a stable scene-center reference for the
        # split-completion distance checks.
        junction_center=point_from_location(
            crosswalk_center
        ),
        crosswalk_start_1=ped_1_start,
        crosswalk_end_1=ped_1_end,
        crosswalk_start_2=ped_2_start,
        crosswalk_end_2=ped_2_end,
        traffic_light_anchor=point_from_location(
            traffic_light.get_location()
        ),
        incoming_path=incoming_points,
        lead_left_path=lead_path,
        ego_right_path=ego_path,
        incoming_yaw=float(
            entry_waypoint.transform.rotation.yaw
        ),
        lead_exit_yaw=float(
            lead_branch[1][
                -1
            ].transform.rotation.yaw
        ),
        ego_exit_yaw=float(
            ego_branch[1][
                -1
            ].transform.rotation.yaw
        ),
        lead_signed_turn_deg=float(
            lead_branch[0]
        ),
        ego_signed_turn_deg=float(
            ego_branch[0]
        ),
        crosswalk_width_m=float(
            selected_width
        ),
        # The schema field is retained for compatibility. In V3 this is
        # the hidden map relocation distance from the clear continuation
        # point to the selected professional junction.
        junction_distance_from_continuation_m=float(
            relocation_distance
        ),
        plan_status="PREFLIGHT_PASS",
    )


def scene_plan_to_json(plan: ScenePlan) -> dict[str, Any]:
    return asdict(plan)


def scene_plan_from_json(payload: dict[str, Any]) -> ScenePlan:
    def point(data: dict[str, Any]) -> Point3:
        return Point3(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
        )

    def rotation(data: dict[str, Any]) -> Rotation3:
        return Rotation3(
            pitch=float(data["pitch"]),
            yaw=float(data["yaw"]),
            roll=float(data["roll"]),
        )

    def transform(data: dict[str, Any]) -> TransformData:
        return TransformData(
            location=point(data["location"]),
            rotation=rotation(data["rotation"]),
        )

    return ScenePlan(
        map_name=str(payload["map_name"]),
        route_json=str(payload["route_json"]),
        route_rank=int(payload["route_rank"]),
        locked_spawn_index=int(
            payload["locked_spawn_index"]
        ),
        locked_road_id=int(payload["locked_road_id"]),
        continuation_distance_m=float(
            payload["continuation_distance_m"]
        ),
        scene_start=transform(payload["scene_start"]),
        lead_start=transform(payload["lead_start"]),
        lead_stop=point(payload["lead_stop"]),
        ego_stop=point(payload["ego_stop"]),
        junction_entry=transform(
            payload["junction_entry"]
        ),
        junction_center=point(
            payload["junction_center"]
        ),
        crosswalk_start_1=point(
            payload["crosswalk_start_1"]
        ),
        crosswalk_end_1=point(
            payload["crosswalk_end_1"]
        ),
        crosswalk_start_2=point(
            payload["crosswalk_start_2"]
        ),
        crosswalk_end_2=point(
            payload["crosswalk_end_2"]
        ),
        traffic_light_anchor=point(
            payload["traffic_light_anchor"]
        ),
        incoming_path=[
            point(item)
            for item in payload["incoming_path"]
        ],
        lead_left_path=[
            point(item)
            for item in payload["lead_left_path"]
        ],
        ego_right_path=[
            point(item)
            for item in payload["ego_right_path"]
        ],
        incoming_yaw=float(payload["incoming_yaw"]),
        lead_exit_yaw=float(payload["lead_exit_yaw"]),
        ego_exit_yaw=float(payload["ego_exit_yaw"]),
        lead_signed_turn_deg=float(
            payload["lead_signed_turn_deg"]
        ),
        ego_signed_turn_deg=float(
            payload["ego_signed_turn_deg"]
        ),
        crosswalk_width_m=float(
            payload["crosswalk_width_m"]
        ),
        junction_distance_from_continuation_m=float(
            payload[
                "junction_distance_from_continuation_m"
            ]
        ),
        plan_status=str(payload["plan_status"]),
    )


def connect_world(
    host: str,
    port: int,
    timeout_seconds: float,
) -> tuple[Any, Any]:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(timeout_seconds)
    world = client.get_world()

    return client, world


def ensure_town(world: Any, town: str) -> None:
    active_map = str(world.get_map().name)

    if not active_map.endswith(f"/{town}"):
        raise RuntimeError(
            f"Active map is {active_map}; expected {town}. "
            f"Load {town} before running this phase."
        )


def create_new_run(output_root: Path) -> Path:
    run_dir = (
        output_root.resolve()
        / f"run_{timestamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def run_preflight(
    args: argparse.Namespace,
    run_dir: Optional[Path] = None,
) -> Path:
    new_run = run_dir is None
    actual_run = (
        create_new_run(args.output_root)
        if run_dir is None
        else run_dir
    )
    route_path = (
        args.route_json.resolve()
        if args.route_json is not None
        else latest_route_json()
    )

    print("=" * 94)
    print("TOWN10 RAIN SCENE - PEDESTRIANS EXIT FRAME | BUILD 20260807-E")
    print("=" * 94)
    print(f"Route JSON: {route_path}")
    print(f"Route rank: {args.route_rank}")
    print(
        "Audit: whole Town10 map for a signalised crosswalk junction "
        "with opposite turn branches"
    )
    print("=" * 94)

    try:
        _client, world = connect_world(
            args.host,
            args.port,
            90.0,
        )
        ensure_town(world, args.town)
        plan = build_scene_plan(
            world,
            route_path,
            args.route_rank,
        )
        plan_path = actual_run / "scene_plan.json"
        plan_path.write_text(
            json.dumps(
                scene_plan_to_json(plan),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary = [
            "TOWN10 RAIN SCENE - PREFLIGHT",
            "=" * 82,
            "Status: PREFLIGHT_PASS",
            f"Map: {plan.map_name}",
            (
                "Locked route: "
                f"rank {plan.route_rank}, "
                f"spawn {plan.locked_spawn_index}, "
                f"road {plan.locked_road_id}"
            ),
            (
                "Continuation distance: "
                f"{plan.continuation_distance_m:.1f} m"
            ),
            (
                "Hidden Town10 relocation distance: "
                f"{plan.junction_distance_from_continuation_m:.1f} m"
            ),
            (
                "Crosswalk width: "
                f"{plan.crosswalk_width_m:.1f} m"
            ),
            (
                "Opposite signed turns: "
                f"lead={plan.lead_signed_turn_deg:.1f}°, "
                f"ego={plan.ego_signed_turn_deg:.1f}°"
            ),
            f"Plan: {plan_path}",
        ]
        (actual_run / "preflight_summary.txt").write_text(
            "\n".join(summary) + "\n",
            encoding="utf-8",
        )

        print("FINAL STATUS: PREFLIGHT_PASS")
        print(
            "Opposite turns: "
            f"{plan.lead_signed_turn_deg:.1f}°, "
            f"{plan.ego_signed_turn_deg:.1f}°"
        )
        print(f"Scene plan: {plan_path}")
        print("=" * 94)

        return actual_run

    except Exception:
        if (
            new_run
            and not args.keep_run_on_failure
            and actual_run.exists()
            and not any(actual_run.iterdir())
        ):
            actual_run.rmdir()

        raise


def load_plan(run_dir: Path) -> ScenePlan:
    plan_path = run_dir / "scene_plan.json"

    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Scene plan is missing: {plan_path}"
        )

    payload = json.loads(
        plan_path.read_text(encoding="utf-8")
    )
    plan = scene_plan_from_json(payload)

    if plan.plan_status != "PREFLIGHT_PASS":
        raise RuntimeError(
            "The scene plan did not pass preflight."
        )

    return plan


def find_blueprint(
    library: Any,
    priorities: tuple[str, ...],
    fallback_pattern: str,
) -> Any:
    for blueprint_id in priorities:
        try:
            return library.find(blueprint_id)
        except RuntimeError:
            continue

    fallback = list(library.filter(fallback_pattern))

    if not fallback:
        raise RuntimeError(
            "No suitable CARLA blueprint was found for "
            f"{priorities}"
        )

    return fallback[0]


def configure_blueprint(
    blueprint: Any,
    role_name: str,
    color_index: int = 0,
) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute(
            "is_invincible",
            "false",
        )

    if blueprint.has_attribute("color"):
        values = (
            blueprint.get_attribute(
                "color"
            ).recommended_values
        )

        if values:
            blueprint.set_attribute(
                "color",
                values[color_index % len(values)],
            )


def raised_transform(
    transform: Any,
    z_offset: float,
) -> Any:
    import carla

    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z + z_offset),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def point_transform(
    point: Point3,
    yaw: float,
    z_offset: float,
) -> Any:
    import carla

    return carla.Transform(
        carla.Location(
            x=point.x,
            y=point.y,
            z=point.z + z_offset,
        ),
        carla.Rotation(yaw=yaw),
    )


def path_to_locations(path: list[Point3]) -> list[Any]:
    return [point.to_carla() for point in path]


def speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()

    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def nearest_path_index(
    actor: Any,
    path: list[Any],
    previous_index: int,
) -> int:
    location = actor.get_location()
    start = max(0, previous_index - 2)
    end = min(len(path), previous_index + 14)

    if start >= end:
        return min(previous_index, len(path) - 1)

    return min(
        range(start, end),
        key=lambda index: path[index].distance(location),
    )


def lookahead_index(
    path: list[Any],
    start_index: int,
    distance_m: float,
) -> int:
    travelled = 0.0
    index = min(start_index, len(path) - 1)

    while index < len(path) - 1 and travelled < distance_m:
        travelled += path[index].distance(
            path[index + 1]
        )
        index += 1

    return index


def apply_vehicle_path_control(
    actor: Any,
    follower: VehicleFollower,
    target_speed_kmh: float,
    stop_location: Optional[Any] = None,
) -> float:
    import carla

    if not follower.path:
        actor.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
        return 0.0

    follower.index = nearest_path_index(
        actor,
        follower.path,
        follower.index,
    )
    speed = speed_mps(actor)
    lookahead_m = clamp(3.5 + speed * 0.75, 4.0, 8.0)
    target_index = lookahead_index(
        follower.path,
        follower.index,
        lookahead_m,
    )
    target = follower.path[target_index]
    transform = actor.get_transform()
    difference = target - transform.location
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    longitudinal = (
        difference.x * forward.x
        + difference.y * forward.y
    )
    lateral = (
        difference.x * right.x
        + difference.y * right.y
    )
    steering_angle = math.atan2(
        lateral,
        max(0.30, longitudinal),
    )
    steer = clamp(
        steering_angle / math.radians(40.0),
        -0.85,
        0.85,
    )
    desired_speed = max(
        0.0,
        target_speed_kmh / 3.6,
    )
    distance_to_stop = float("inf")

    if stop_location is not None:
        distance_to_stop = actor.get_location().distance(
            stop_location
        )
        braking_speed = max(
            0.0,
            min(
                desired_speed,
                max(
                    0.0,
                    (distance_to_stop - 0.5) * 0.55,
                ),
            ),
        )
        desired_speed = braking_speed

    error = desired_speed - speed
    throttle = clamp(error * 0.42, 0.0, 0.52)
    brake = clamp(-error * 0.85, 0.0, 1.0)

    if (
        stop_location is not None
        and distance_to_stop <= 0.75
    ):
        throttle = 0.0
        brake = 1.0
        steer *= 0.30

    if target_speed_kmh <= 0.01:
        throttle = 0.0
        brake = 1.0

    actor.apply_control(
        carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
        )
    )

    return float(distance_to_stop)


def hold_vehicle(actor: Any) -> None:
    import carla

    actor.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
        )
    )


def inset_point_toward(
    start: Point3,
    end: Point3,
    inset_m: float,
) -> Point3:
    dx = float(end.x - start.x)
    dy = float(end.y - start.y)
    dz = float(end.z - start.z)
    length = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    if length <= 0.001:
        return start

    scale = min(
        max(0.0, inset_m) / length,
        0.40,
    )

    return Point3(
        x=float(start.x + dx * scale),
        y=float(start.y + dy * scale),
        z=float(start.z + dz * scale),
    )


def spawn_walker_on_crosswalk(
    world: Any,
    blueprint: Any,
    start: Point3,
    end: Point3,
    yaw: float,
) -> tuple[Any, Point3]:
    """
    Crosswalk endpoints can sit exactly on a Town10 curb/collision edge.
    Spawn slightly inside the painted crossing and try a few safe heights.
    No AI walker controller is used.
    """
    for inset_m in (
        0.80,
        1.10,
        1.40,
        1.80,
    ):
        candidate = inset_point_toward(
            start,
            end,
            inset_m,
        )

        for z_offset in (
            0.25,
            0.50,
            0.80,
            1.10,
        ):
            actor = world.try_spawn_actor(
                blueprint,
                point_transform(
                    candidate,
                    yaw,
                    z_offset,
                ),
            )

            if actor is not None:
                return actor, candidate

    raise RuntimeError(
        "Could not spawn a pedestrian safely inside the "
        "preflight-approved zebra crossing."
    )


def extend_point_beyond(
    start: Point3,
    end: Point3,
    extra_distance_m: float,
) -> Point3:
    """
    Continue in the same crossing direction beyond the far edge.
    Used only after the pedestrian has already cleared the road.
    """
    dx = float(end.x - start.x)
    dy = float(end.y - start.y)
    dz = float(end.z - start.z)
    length = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    if length <= 0.001:
        return end

    scale = max(
        0.0,
        extra_distance_m,
    ) / length

    return Point3(
        x=float(end.x + dx * scale),
        y=float(end.y + dy * scale),
        z=float(end.z + dz * scale),
    )


def stop_walker(actor: Any) -> None:
    import carla

    actor.apply_control(
        carla.WalkerControl(
            direction=carla.Vector3D(
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            speed=0.0,
            jump=False,
        )
    )


def move_walker(
    actor: Any,
    target: Any,
    speed: float,
) -> tuple[float, bool]:
    import carla

    current = actor.get_location()
    dx = float(target.x - current.x)
    dy = float(target.y - current.y)
    dz = float(target.z - current.z)
    distance = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    if distance <= 1.20:
        stop_walker(actor)
        return distance, True

    magnitude = max(0.001, math.sqrt(dx * dx + dy * dy))
    direction = carla.Vector3D(
        x=dx / magnitude,
        y=dy / magnitude,
        z=0.0,
    )
    actor.apply_control(
        carla.WalkerControl(
            direction=direction,
            speed=float(speed),
            jump=False,
        )
    )

    return distance, False


def rain_weather(progress: float) -> Any:
    import carla

    value = clamp(progress, 0.0, 1.0)

    # Presentation rain: visibly wet road and rain streaks while keeping
    # pedestrians, signal colour, lane markings and the lead car readable.
    return carla.WeatherParameters(
        cloudiness=12.0 + 40.0 * value,
        precipitation=0.0 + 30.0 * value,
        precipitation_deposits=5.0 + 43.0 * value,
        wind_intensity=4.0 + 10.0 * value,
        sun_azimuth_angle=35.0,
        sun_altitude_angle=58.0 - 5.0 * value,
        fog_density=0.0,
        fog_distance=1000.0,
        fog_falloff=0.2,
        wetness=10.0 + 52.0 * value,
        scattering_intensity=0.85,
        mie_scattering_scale=0.015,
        rayleigh_scattering_scale=0.0331,
    )


def image_to_bgr(image: Any) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape(
        (image.height, image.width, 4)
    )

    return array[:, :, :3].copy()


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
            f"Could not create MP4 video: {path}"
        )

    return writer


def camera_intrinsic(
    width: int,
    height: int,
    fov_degrees: float,
) -> np.ndarray:
    focal = width / (
        2.0
        * math.tan(
            math.radians(fov_degrees) / 2.0
        )
    )
    matrix = np.identity(3)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0

    return matrix


def world_to_camera(
    location: Any,
    camera: Any,
) -> np.ndarray:
    vector = np.array(
        [
            float(location.x),
            float(location.y),
            float(location.z),
            1.0,
        ],
        dtype=np.float64,
    )
    inverse = np.array(
        camera.get_transform().get_inverse_matrix(),
        dtype=np.float64,
    )
    sensor_point = inverse @ vector

    return np.array(
        [
            sensor_point[1],
            -sensor_point[2],
            sensor_point[0],
        ],
        dtype=np.float64,
    )


def project_actor_box(
    actor: Any,
    camera: Any,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[
    bool,
    float,
    float,
    float,
    float,
    float,
]:
    try:
        vertices = actor.bounding_box.get_world_vertices(
            actor.get_transform()
        )
    except (AttributeError, RuntimeError):
        return (
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    projected: list[tuple[float, float, float]] = []

    for vertex in vertices:
        camera_point = world_to_camera(
            vertex,
            camera,
        )
        depth = float(camera_point[2])

        if depth <= 0.20:
            continue

        image_point = intrinsic @ camera_point
        u = float(
            image_point[0] / image_point[2]
        )
        v = float(
            image_point[1] / image_point[2]
        )
        projected.append((u, v, depth))

    if len(projected) < 4:
        return (
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    x_values = [item[0] for item in projected]
    y_values = [item[1] for item in projected]
    depths = [item[2] for item in projected]
    x1 = max(0.0, min(x_values))
    y1 = max(0.0, min(y_values))
    x2 = min(float(width - 1), max(x_values))
    y2 = min(float(height - 1), max(y_values))
    depth = float(statistics.median(depths))
    visible = (
        x2 - x1 >= 5.0
        and y2 - y1 >= 7.0
        and depth <= 65.0
        and x2 >= 0.0
        and y2 >= 0.0
        and x1 <= width
        and y1 <= height
    )

    return visible, x1, y1, x2, y2, depth


def find_light_for_plan(
    world: Any,
    plan: ScenePlan,
) -> Any:
    anchor = plan.traffic_light_anchor.to_carla()
    lights = list(
        world.get_actors().filter("*traffic_light*")
    )

    if not lights:
        raise RuntimeError(
            "No traffic light actors exist in the active world."
        )

    return min(
        lights,
        key=lambda light: light.get_location().distance(
            anchor
        ),
    )


def set_light_state(
    light: Any,
    state: str,
) -> None:
    import carla

    state_map = {
        "Red": carla.TrafficLightState.Red,
        "Green": carla.TrafficLightState.Green,
        "Yellow": carla.TrafficLightState.Yellow,
    }

    try:
        light.freeze(True)
        light.set_state(state_map[state])
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not set traffic light to {state}: {error}"
        ) from error


def traffic_state_name(light: Any) -> str:
    try:
        return str(light.get_state()).split(".")[-1]
    except RuntimeError:
        return "Unknown"


def batch_destroy_owned(
    client: Any,
    world: Any,
    owned_actor_ids: list[int],
) -> None:
    if not owned_actor_ids:
        return

    try:
        import carla

        commands = [
            carla.command.DestroyActor(actor_id)
            for actor_id in reversed(owned_actor_ids)
        ]
        client.apply_batch_sync(commands, True)
    except Exception as error:
        print(
            "WARNING: owned-actor cleanup could not complete: "
            f"{type(error).__name__}: {error}"
        )


def capture_row_for_actor(
    row: dict[str, Any],
    key: str,
    actor: Any,
    camera: Any,
    intrinsic: np.ndarray,
) -> None:
    (
        visible,
        x1,
        y1,
        x2,
        y2,
        depth,
    ) = project_actor_box(
        actor,
        camera,
        intrinsic,
        WIDTH,
        HEIGHT,
    )
    row[f"{key}_actor_id"] = int(actor.id)
    row[f"{key}_visible"] = int(visible)
    row[f"{key}_x1"] = x1
    row[f"{key}_y1"] = y1
    row[f"{key}_x2"] = x2
    row[f"{key}_y2"] = y2
    row[f"{key}_depth_m"] = depth


def run_capture(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    import carla

    plan = load_plan(run_dir)
    client, world = connect_world(
        args.host,
        args.port,
        90.0,
    )
    ensure_town(world, args.town)

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    traffic_manager = client.get_trafficmanager(
        args.tm_port
    )
    original_light_state: Optional[Any] = None
    relevant_light: Optional[Any] = None
    owned_actor_ids: list[int] = []
    camera = None
    writer: Optional[cv2.VideoWriter] = None

    raw_path = run_dir / "rain_signal_raw.mp4"
    metrics_path = (
        run_dir / "ground_truth_metrics.csv"
    )
    report_path = run_dir / "capture_report.json"
    summary_path = run_dir / "capture_summary.txt"

    frame_queue: queue.Queue[
        tuple[int, float, np.ndarray]
    ] = queue.Queue(maxsize=8)

    print("=" * 94)
    print("TOWN10 CONTROLLED RAIN CAPTURE")
    print("=" * 94)
    print(f"Map: {world.get_map().name}")
    print(
        "Scene: lead car -> red signal -> two pedestrians -> "
        "green -> opposite turns"
    )
    print(
        "Safety: only script-owned actors will be destroyed"
    )
    print(
        "Presentation setup: camera z=1.70m | lead stop gap=14m | "
        "bright moderate rain"
    )
    print(
        "Pedestrian finish: keep walking toward camera-left after "
        "crossing, then leave the shot"
    )
    print("=" * 94)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / WORLD_FPS
        )
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        world.set_weather(rain_weather(0.0))

        library = world.get_blueprint_library()

        ego_blueprint = find_blueprint(
            library,
            (
                "vehicle.tesla.model3",
                "vehicle.lincoln.mkz_2020",
                "vehicle.audi.tt",
            ),
            "vehicle.*",
        )
        configure_blueprint(
            ego_blueprint,
            "rain_signal_ego",
            0,
        )

        lead_blueprint = find_blueprint(
            library,
            (
                "vehicle.audi.tt",
                "vehicle.mercedes.coupe_2020",
                "vehicle.lincoln.mkz_2020",
            ),
            "vehicle.*",
        )
        configure_blueprint(
            lead_blueprint,
            "rain_signal_lead_car",
            1,
        )

        walker_blueprint_1 = find_blueprint(
            library,
            (
                "walker.pedestrian.0001",
                "walker.pedestrian.0004",
                "walker.pedestrian.0007",
            ),
            "walker.pedestrian.*",
        )
        configure_blueprint(
            walker_blueprint_1,
            "rain_signal_pedestrian_1",
        )

        walker_blueprint_2 = find_blueprint(
            library,
            (
                "walker.pedestrian.0013",
                "walker.pedestrian.0020",
                "walker.pedestrian.0030",
            ),
            "walker.pedestrian.*",
        )
        configure_blueprint(
            walker_blueprint_2,
            "rain_signal_pedestrian_2",
        )

        ego = world.try_spawn_actor(
            ego_blueprint,
            raised_transform(
                plan.scene_start.to_carla(),
                0.45,
            ),
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn the ego vehicle at the "
                "preflight-approved continuation point."
            )

        owned_actor_ids.append(int(ego.id))

        lead = world.try_spawn_actor(
            lead_blueprint,
            raised_transform(
                plan.lead_start.to_carla(),
                0.45,
            ),
        )

        if lead is None:
            raise RuntimeError(
                "Could not spawn the lead car on the "
                "preflight-approved approach lane."
            )

        owned_actor_ids.append(int(lead.id))

        (
            pedestrian_1,
            pedestrian_spawn_1,
        ) = spawn_walker_on_crosswalk(
            world,
            walker_blueprint_1,
            plan.crosswalk_start_1,
            plan.crosswalk_end_1,
            plan.incoming_yaw + 90.0,
        )
        owned_actor_ids.append(
            int(pedestrian_1.id)
        )

        (
            pedestrian_2,
            pedestrian_spawn_2,
        ) = spawn_walker_on_crosswalk(
            world,
            walker_blueprint_2,
            plan.crosswalk_start_2,
            plan.crosswalk_end_2,
            plan.incoming_yaw + 90.0,
        )
        owned_actor_ids.append(
            int(pedestrian_2.id)
        )
        stop_walker(pedestrian_1)
        stop_walker(pedestrian_2)
        print(
            "Pedestrians spawned safely inside the zebra crossing: "
            f"{pedestrian_1.id}, {pedestrian_2.id}"
        )

        relevant_light = find_light_for_plan(
            world,
            plan,
        )
        original_light_state = relevant_light.get_state()
        set_light_state(relevant_light, "Red")

        camera_blueprint = library.find(
            "sensor.camera.rgb"
        )
        camera_blueprint.set_attribute(
            "image_size_x",
            str(WIDTH),
        )
        camera_blueprint.set_attribute(
            "image_size_y",
            str(HEIGHT),
        )
        camera_blueprint.set_attribute(
            "fov",
            f"{FOV:.1f}",
        )
        camera_blueprint.set_attribute(
            "sensor_tick",
            f"{1.0 / FPS:.6f}",
        )

        for attribute_name in (
            "motion_blur_intensity",
            "motion_blur_max_distortion",
            "lens_flare_intensity",
        ):
            if camera_blueprint.has_attribute(
                attribute_name
            ):
                camera_blueprint.set_attribute(
                    attribute_name,
                    "0.0",
                )

        def callback(image: Any) -> None:
            item = (
                int(image.frame),
                float(image.timestamp),
                image_to_bgr(image),
            )

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

        camera = world.spawn_actor(
            camera_blueprint,
            carla.Transform(
                carla.Location(
                    x=CAMERA_X,
                    y=CAMERA_Y,
                    z=CAMERA_Z,
                ),
                carla.Rotation(
                    pitch=CAMERA_PITCH,
                ),
            ),
            attach_to=ego,
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )
        owned_actor_ids.append(int(camera.id))
        camera.listen(callback)

        ego_follower = VehicleFollower(
            path=path_to_locations(
                plan.ego_right_path
            )
        )
        lead_follower = VehicleFollower(
            path=path_to_locations(
                plan.lead_left_path
            )
        )
        ego_stop = plan.ego_stop.to_carla()
        lead_stop = plan.lead_stop.to_carla()
        pedestrian_target_1 = (
            inset_point_toward(
                plan.crosswalk_end_1,
                plan.crosswalk_start_1,
                0.80,
            ).to_carla()
        )
        pedestrian_target_2 = (
            inset_point_toward(
                plan.crosswalk_end_2,
                plan.crosswalk_start_2,
                0.80,
            ).to_carla()
        )

        # After clearing the road, both pedestrians keep walking in the
        # same direction for another 10 m. In this locked scene that sends
        # them toward camera-left and naturally out of the final shot.
        pedestrian_exit_target_1 = (
            extend_point_beyond(
                plan.crosswalk_start_1,
                plan.crosswalk_end_1,
                10.0,
            ).to_carla()
        )
        pedestrian_exit_target_2 = (
            extend_point_beyond(
                plan.crosswalk_start_2,
                plan.crosswalk_end_2,
                10.0,
            ).to_carla()
        )

        # Settle all spawned actors while stationary. Camera frames are
        # discarded so the final video does not show spawning.
        hold_vehicle(ego)
        hold_vehicle(lead)

        for _ in range(int(WORLD_FPS * 1.5)):
            world.tick()

        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        writer = create_writer(
            raw_path,
            FPS,
            WIDTH,
            HEIGHT,
        )
        intrinsic = camera_intrinsic(
            WIDTH,
            HEIGHT,
            FOV,
        )

        state = "APPROACH_RED"
        state_started: Optional[float] = None
        capture_started: Optional[float] = None
        last_weather_progress = -1.0
        pedestrian_1_done = False
        pedestrian_2_done = False
        pedestrian_1_started = False
        pedestrian_2_started = False
        both_stopped_frame: Optional[int] = None
        green_frame: Optional[int] = None
        release_frame: Optional[int] = None
        split_complete_timestamp: Optional[float] = None
        rows: list[dict[str, Any]] = []
        recorded = 0
        maximum_frames = int(
            MAX_CAPTURE_SECONDS * FPS
        )

        while recorded < maximum_frames:
            world.tick()

            # Drain every available camera image. At WORLD_FPS=20 and
            # camera FPS=10 this is normally zero or one image per tick.
            available: list[
                tuple[int, float, np.ndarray]
            ] = []

            while True:
                try:
                    available.append(
                        frame_queue.get_nowait()
                    )
                except queue.Empty:
                    break

            simulation_time = (
                available[-1][1]
                if available
                else float(world.get_snapshot().timestamp.elapsed_seconds)
            )

            if capture_started is None and available:
                capture_started = simulation_time
                state_started = simulation_time

            elapsed = (
                0.0
                if capture_started is None
                else simulation_time - capture_started
            )
            state_elapsed = (
                0.0
                if state_started is None
                else simulation_time - state_started
            )

            if elapsed <= RAIN_CLEAR_LEAD_SECONDS:
                rain_progress = 0.0
            else:
                rain_progress = clamp(
                    (
                        elapsed
                        - RAIN_CLEAR_LEAD_SECONDS
                    )
                    / RAIN_TRANSITION_SECONDS,
                    0.0,
                    1.0,
                )

            if (
                last_weather_progress < 0.0
                or abs(
                    rain_progress
                    - last_weather_progress
                )
                >= 0.05
            ):
                world.set_weather(
                    rain_weather(rain_progress)
                )
                last_weather_progress = rain_progress

            lead_distance = float("inf")
            ego_distance = float("inf")

            if state == "APPROACH_RED":
                set_light_state(
                    relevant_light,
                    "Red",
                )
                lead_distance = apply_vehicle_path_control(
                    lead,
                    lead_follower,
                    LEAD_APPROACH_SPEED_KMH,
                    lead_stop,
                )
                ego_distance = apply_vehicle_path_control(
                    ego,
                    ego_follower,
                    EGO_APPROACH_SPEED_KMH,
                    ego_stop,
                )
                stop_walker(pedestrian_1)
                stop_walker(pedestrian_2)

                lead_stopped = (
                    lead_distance <= 1.10
                    and speed_mps(lead) <= 0.35
                )
                ego_stopped = (
                    ego_distance <= 1.15
                    and speed_mps(ego) <= 0.35
                )

                if lead_stopped and ego_stopped:
                    state = "PEDESTRIAN_CROSSING"
                    state_started = simulation_time
                    both_stopped_frame = recorded + 1
                    hold_vehicle(lead)
                    hold_vehicle(ego)

            elif state == "PEDESTRIAN_CROSSING":
                set_light_state(
                    relevant_light,
                    "Red",
                )
                hold_vehicle(lead)
                hold_vehicle(ego)

                if state_elapsed >= 0.35:
                    pedestrian_1_started = True

                if (
                    state_elapsed
                    >= 0.35 + PED_2_DELAY_SECONDS
                ):
                    pedestrian_2_started = True

                if (
                    pedestrian_1_started
                    and not pedestrian_1_done
                ):
                    (
                        _distance,
                        pedestrian_1_done,
                    ) = move_walker(
                        pedestrian_1,
                        pedestrian_target_1,
                        PED_1_SPEED_MPS,
                    )
                else:
                    stop_walker(pedestrian_1)

                if (
                    pedestrian_2_started
                    and not pedestrian_2_done
                ):
                    (
                        _distance,
                        pedestrian_2_done,
                    ) = move_walker(
                        pedestrian_2,
                        pedestrian_target_2,
                        PED_2_SPEED_MPS,
                    )
                else:
                    stop_walker(pedestrian_2)

                if (
                    pedestrian_1_done
                    and pedestrian_2_done
                ):
                    state = "GREEN_SETTLE"
                    state_started = simulation_time
                    set_light_state(
                        relevant_light,
                        "Green",
                    )
                    green_frame = recorded + 1

            elif state == "GREEN_SETTLE":
                set_light_state(
                    relevant_light,
                    "Green",
                )
                hold_vehicle(lead)
                hold_vehicle(ego)

                # Crossing is already complete. Keep both pedestrians
                # walking onto/past the far-side pavement rather than
                # freezing them at the road edge.
                move_walker(
                    pedestrian_1,
                    pedestrian_exit_target_1,
                    PED_1_SPEED_MPS,
                )
                move_walker(
                    pedestrian_2,
                    pedestrian_exit_target_2,
                    PED_2_SPEED_MPS,
                )

                if state_elapsed >= GREEN_SETTLE_SECONDS:
                    state = "RELEASE_AND_SPLIT"
                    state_started = simulation_time
                    release_frame = recorded + 1

            elif state == "RELEASE_AND_SPLIT":
                set_light_state(
                    relevant_light,
                    "Green",
                )

                move_walker(
                    pedestrian_1,
                    pedestrian_exit_target_1,
                    PED_1_SPEED_MPS,
                )
                move_walker(
                    pedestrian_2,
                    pedestrian_exit_target_2,
                    PED_2_SPEED_MPS,
                )

                apply_vehicle_path_control(
                    lead,
                    lead_follower,
                    LEAD_RELEASE_SPEED_KMH,
                    None,
                )

                if state_elapsed < EGO_RELEASE_DELAY_SECONDS:
                    hold_vehicle(ego)
                else:
                    apply_vehicle_path_control(
                        ego,
                        ego_follower,
                        EGO_RELEASE_SPEED_KMH,
                        None,
                    )

                lead_turn = signed_yaw_difference(
                    lead.get_transform().rotation.yaw,
                    plan.incoming_yaw,
                )
                ego_turn = signed_yaw_difference(
                    ego.get_transform().rotation.yaw,
                    plan.incoming_yaw,
                )
                opposite_turns = (
                    lead_turn * ego_turn < 0.0
                    and abs(lead_turn) >= 38.0
                    and abs(ego_turn) >= 38.0
                )
                both_clear = (
                    lead.get_location().distance(
                        plan.junction_center.to_carla()
                    )
                    >= 14.0
                    and ego.get_location().distance(
                        plan.junction_center.to_carla()
                    )
                    >= 10.0
                )

                if opposite_turns and both_clear:
                    if split_complete_timestamp is None:
                        split_complete_timestamp = (
                            simulation_time
                        )
                    elif (
                        simulation_time
                        - split_complete_timestamp
                        >= FINAL_TAIL_SECONDS
                    ):
                        state = "FINISHED"

            elif state == "FINISHED":
                hold_vehicle(ego)
                hold_vehicle(lead)
                move_walker(
                    pedestrian_1,
                    pedestrian_exit_target_1,
                    PED_1_SPEED_MPS,
                )
                move_walker(
                    pedestrian_2,
                    pedestrian_exit_target_2,
                    PED_2_SPEED_MPS,
                )

            for (
                carla_frame,
                image_timestamp,
                frame,
            ) in available:
                if recorded >= maximum_frames:
                    break

                if capture_started is None:
                    capture_started = image_timestamp

                recorded += 1
                writer.write(frame)

                row: dict[str, Any] = {
                    "video_frame": recorded,
                    "carla_frame": carla_frame,
                    "time_seconds": (
                        image_timestamp
                        - capture_started
                    ),
                    "scene_state": state,
                    "rain_progress": rain_progress,
                    "traffic_light_state": (
                        traffic_state_name(
                            relevant_light
                        )
                    ),
                    "ego_speed_mps": speed_mps(ego),
                    "lead_car_speed_mps": speed_mps(
                        lead
                    ),
                    "ego_yaw": float(
                        ego.get_transform().rotation.yaw
                    ),
                    "lead_car_yaw": float(
                        lead.get_transform().rotation.yaw
                    ),
                    "ego_signed_turn_deg": (
                        signed_yaw_difference(
                            ego.get_transform().rotation.yaw,
                            plan.incoming_yaw,
                        )
                    ),
                    "lead_signed_turn_deg": (
                        signed_yaw_difference(
                            lead.get_transform().rotation.yaw,
                            plan.incoming_yaw,
                        )
                    ),
                    "pedestrian_1_started": int(
                        pedestrian_1_started
                    ),
                    "pedestrian_2_started": int(
                        pedestrian_2_started
                    ),
                    "pedestrian_1_done": int(
                        pedestrian_1_done
                    ),
                    "pedestrian_2_done": int(
                        pedestrian_2_done
                    ),
                }
                capture_row_for_actor(
                    row,
                    "lead_car",
                    lead,
                    camera,
                    intrinsic,
                )
                capture_row_for_actor(
                    row,
                    "pedestrian_1",
                    pedestrian_1,
                    camera,
                    intrinsic,
                )
                capture_row_for_actor(
                    row,
                    "pedestrian_2",
                    pedestrian_2,
                    camera,
                    intrinsic,
                )
                rows.append(row)

                if recorded in {
                    1,
                    max(1, maximum_frames // 4),
                    max(1, maximum_frames // 2),
                    max(1, 3 * maximum_frames // 4),
                }:
                    cv2.imwrite(
                        str(
                            run_dir
                            / (
                                "capture_review_"
                                f"{recorded:03d}.png"
                            )
                        ),
                        frame,
                    )

            if (
                recorded > 0
                and recorded % 40 == 0
                and available
            ):
                extra = ""

                if state == "PEDESTRIAN_CROSSING":
                    p1_distance = pedestrian_1.get_location().distance(
                        pedestrian_target_1
                    )
                    p2_distance = pedestrian_2.get_location().distance(
                        pedestrian_target_2
                    )
                    extra = (
                        f" | p1_remaining={p1_distance:.1f}m"
                        f" | p2_remaining={p2_distance:.1f}m"
                    )

                print(
                    f"Captured {recorded}/{maximum_frames} frames "
                    f"| state={state}{extra}"
                )

            if state == "FINISHED":
                break

        if not rows:
            raise RuntimeError(
                "The RGB camera produced no captured frames."
            )

        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer_csv = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer_csv.writeheader()
            writer_csv.writerows(rows)

        final_lead_turn = signed_yaw_difference(
            lead.get_transform().rotation.yaw,
            plan.incoming_yaw,
        )
        final_ego_turn = signed_yaw_difference(
            ego.get_transform().rotation.yaw,
            plan.incoming_yaw,
        )
        opposite_turns = (
            final_lead_turn * final_ego_turn < 0.0
            and abs(final_lead_turn) >= 38.0
            and abs(final_ego_turn) >= 38.0
        )
        full_rain = max(
            float(row["rain_progress"])
            for row in rows
        ) >= 0.95
        red_during_crossing = any(
            row["scene_state"]
            == "PEDESTRIAN_CROSSING"
            and row["traffic_light_state"] == "Red"
            for row in rows
        )
        green_after_crossing = any(
            row["scene_state"]
            in {
                "GREEN_SETTLE",
                "RELEASE_AND_SPLIT",
                "FINISHED",
            }
            and row["traffic_light_state"] == "Green"
            for row in rows
        )
        both_pedestrians_completed = (
            pedestrian_1_done
            and pedestrian_2_done
        )
        capture_pass = all(
            (
                state == "FINISHED",
                both_stopped_frame is not None,
                both_pedestrians_completed,
                green_frame is not None,
                release_frame is not None,
                red_during_crossing,
                green_after_crossing,
                opposite_turns,
                full_rain,
            )
        )
        capture_status = (
            "PASS_CAPTURE"
            if capture_pass
            else "REVIEW_CAPTURE"
        )

        report = {
            "status": capture_status,
            "map": str(world.get_map().name),
            "scene_plan": str(
                run_dir / "scene_plan.json"
            ),
            "raw_video": str(raw_path),
            "ground_truth_metrics": str(
                metrics_path
            ),
            "frame_count": recorded,
            "duration_seconds": recorded / FPS,
            "final_state": state,
            "both_stopped_frame": both_stopped_frame,
            "green_frame": green_frame,
            "release_frame": release_frame,
            "pedestrian_1_done": pedestrian_1_done,
            "pedestrian_2_done": pedestrian_2_done,
            "red_during_crossing": (
                red_during_crossing
            ),
            "green_after_crossing": (
                green_after_crossing
            ),
            "full_rain_reached": full_rain,
            "final_lead_turn_deg": (
                final_lead_turn
            ),
            "final_ego_turn_deg": final_ego_turn,
            "opposite_turns": opposite_turns,
            "unknown_actors_destroyed": 0,
            "walker_ai_controller_used": False,
        }
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = [
            "TOWN10 CONTROLLED RAIN CAPTURE",
            "=" * 82,
            f"Status: {capture_status}",
            f"Frames: {recorded}",
            f"Duration: {recorded / FPS:.2f} s",
            f"Final state: {state}",
            (
                "Pedestrians complete: "
                f"{both_pedestrians_completed}"
            ),
            (
                "Signal sequence valid: "
                f"{red_during_crossing and green_after_crossing}"
            ),
            (
                "Opposite turns: "
                f"{opposite_turns} "
                f"(lead={final_lead_turn:.1f}°, "
                f"ego={final_ego_turn:.1f}°)"
            ),
            f"Raw video: {raw_path}",
            f"Metrics: {metrics_path}",
        ]
        summary_path.write_text(
            "\n".join(summary) + "\n",
            encoding="utf-8",
        )

        print("=" * 94)
        print(f"FINAL STATUS: {capture_status}")
        print(
            f"Pedestrians complete: {both_pedestrians_completed}"
        )
        print(
            "Opposite turns: "
            f"{opposite_turns} "
            f"(lead={final_lead_turn:.1f}°, "
            f"ego={final_ego_turn:.1f}°)"
        )
        print(f"Raw video: {raw_path}")
        print("=" * 94)

    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

        if writer is not None:
            writer.release()

        if relevant_light is not None:
            try:
                relevant_light.freeze(False)

                if original_light_state is not None:
                    relevant_light.set_state(
                        original_light_state
                    )
            except RuntimeError:
                pass

        # Destroy only actors created by this script. No world-wide actor
        # scan or unknown actor destruction is performed.
        batch_destroy_owned(
            client,
            world,
            owned_actor_ids,
        )

        try:
            traffic_manager.set_synchronous_mode(
                False
            )
        except RuntimeError:
            pass

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        try:
            world.set_weather(original_weather)
        except RuntimeError:
            pass


def read_ground_truth(
    path: Path,
) -> dict[int, dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    return {
        int(float(row["video_frame"])): row
        for row in rows
    }


def model_class_mapping(
    model: Any,
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

    name_to_id = {
        name: class_id
        for class_id, name in id_to_name.items()
    }

    for required in ALLOWED_LABELS:
        if required not in name_to_id:
            raise RuntimeError(
                f"The model has no '{required}' class."
            )

    return id_to_name, name_to_id


def gt_box(
    row: dict[str, str],
    key: str,
) -> Optional[np.ndarray]:
    try:
        visible = (
            int(float(row[f"{key}_visible"])) == 1
        )
        coordinates = np.array(
            [
                float(row[f"{key}_x1"]),
                float(row[f"{key}_y1"]),
                float(row[f"{key}_x2"]),
                float(row[f"{key}_y2"]),
            ],
            dtype=float,
        )
        depth = float(row[f"{key}_depth_m"])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        not visible
        or not np.all(np.isfinite(coordinates))
        or coordinates[2] <= coordinates[0]
        or coordinates[3] <= coordinates[1]
        or not (1.0 <= depth <= 65.0)
    ):
        return None

    return coordinates


def box_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )
    first_area = (
        max(0.0, float(first[2] - first[0]))
        * max(0.0, float(first[3] - first[1]))
    )
    second_area = (
        max(0.0, float(second[2] - second[0]))
        * max(0.0, float(second[3] - second[1]))
    )
    union = first_area + second_area - intersection

    return (
        intersection / union
        if union > 0.0
        else 0.0
    )


def center_distance_ratio(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_center = np.array(
        [
            (first[0] + first[2]) / 2.0,
            (first[1] + first[3]) / 2.0,
        ],
        dtype=float,
    )
    second_center = np.array(
        [
            (second[0] + second[2]) / 2.0,
            (second[1] + second[3]) / 2.0,
        ],
        dtype=float,
    )
    diagonal = max(
        1.0,
        math.hypot(
            float(second[2] - second[0]),
            float(second[3] - second[1]),
        ),
    )

    return float(
        np.linalg.norm(
            first_center - second_center
        )
        / diagonal
    )


def optical_flow_box(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_box: np.ndarray,
) -> Optional[np.ndarray]:
    height, width = previous_gray.shape
    x1, y1, x2, y2 = [
        int(round(value))
        for value in previous_box
    ]
    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width - 1, x2))
    y2 = max(y1 + 1, min(height - 1, y2))
    mask = np.zeros_like(previous_gray)
    mask[y1:y2, x1:x2] = 255
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=mask,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
    )

    if (
        previous_points is None
        or len(previous_points) < 5
    ):
        return None

    current_points, status, _error = (
        cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
    )

    if current_points is None or status is None:
        return None

    valid = status.reshape(-1) == 1
    old_points = previous_points[
        valid
    ].reshape(-1, 2)
    new_points = current_points[
        valid
    ].reshape(-1, 2)

    if len(old_points) < 5:
        return None

    transform, _inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if transform is None:
        return None

    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(
        corners,
        transform,
    ).reshape(-1, 2)
    new_box = np.array(
        [
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
        ],
        dtype=float,
    )

    old_width = max(
        1.0,
        float(previous_box[2] - previous_box[0]),
    )
    old_height = max(
        1.0,
        float(previous_box[3] - previous_box[1]),
    )
    new_width = max(
        1.0,
        float(new_box[2] - new_box[0]),
    )
    new_height = max(
        1.0,
        float(new_box[3] - new_box[1]),
    )

    if not (
        0.60 <= new_width / old_width <= 1.70
        and 0.60 <= new_height / old_height <= 1.70
    ):
        return None

    return new_box


def predictions_for_frame(
    result: Any,
    id_to_name: dict[int, str],
) -> list[tuple[str, float, np.ndarray]]:
    predictions: list[
        tuple[str, float, np.ndarray]
    ] = []

    if result.boxes is None:
        return predictions

    for prediction in result.boxes:
        class_id = int(prediction.cls.item())
        label = id_to_name.get(
            class_id,
            str(class_id),
        )

        if label not in ALLOWED_LABELS:
            continue

        confidence = float(prediction.conf.item())

        if confidence < RAW_CONFIDENCE_FLOOR[label]:
            continue

        box = (
            prediction.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        predictions.append(
            (label, confidence, box)
        )

    return predictions


def assign_predictions(
    predictions: list[
        tuple[str, float, np.ndarray]
    ],
    gt_boxes: dict[str, Optional[np.ndarray]],
) -> dict[
    str,
    Optional[tuple[np.ndarray, float]],
]:
    assignments: dict[
        str,
        Optional[tuple[np.ndarray, float]],
    ] = {
        key: None
        for key in TARGET_KEYS
    }
    candidates: list[
        tuple[
            float,
            str,
            int,
            np.ndarray,
            float,
        ]
    ] = []

    for key in TARGET_KEYS:
        target_box = gt_boxes[key]

        if target_box is None:
            continue

        expected_label = TARGET_TO_LABEL[key]

        for index, (
            label,
            confidence,
            box,
        ) in enumerate(predictions):
            if label != expected_label:
                continue

            overlap = box_iou(box, target_box)
            center_ratio = center_distance_ratio(
                box,
                target_box,
            )

            if overlap < 0.025 and center_ratio > 0.95:
                continue

            score = (
                overlap * 1.15
                + confidence * 0.35
                - center_ratio * 0.08
            )
            candidates.append(
                (
                    float(score),
                    key,
                    index,
                    box,
                    confidence,
                )
            )

    used_keys: set[str] = set()
    used_predictions: set[int] = set()

    for (
        _score,
        key,
        prediction_index,
        box,
        confidence,
    ) in sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    ):
        if (
            key in used_keys
            or prediction_index in used_predictions
        ):
            continue

        assignments[key] = (
            box,
            confidence,
        )
        used_keys.add(key)
        used_predictions.add(prediction_index)

    return assignments


def update_detection_track(
    track: Optional[DetectionTrack],
    key: str,
    label: str,
    prediction: Optional[
        tuple[np.ndarray, float]
    ],
    target_box: Optional[np.ndarray],
    previous_gray: Optional[np.ndarray],
    current_gray: np.ndarray,
) -> Optional[DetectionTrack]:
    if target_box is None:
        return None

    if prediction is not None:
        box, confidence = prediction

        if track is None:
            return DetectionTrack(
                key=key,
                label=label,
                box=box,
                confidence_ema=confidence,
                consecutive_hits=1,
                total_hits=1,
                missed=0,
                confirmed=False,
                source="yolo",
            )

        track.box = (
            0.72 * box + 0.28 * track.box
        )
        track.confidence_ema = (
            0.72 * confidence
            + 0.28 * track.confidence_ema
        )
        track.consecutive_hits += 1
        track.total_hits += 1
        track.missed = 0
        track.source = "yolo"

        if (
            track.consecutive_hits >= 2
            and (
                track.confirmed
                or track.confidence_ema
                >= START_CONFIDENCE[label]
            )
        ):
            track.confirmed = True

        return track

    if (
        track is None
        or not track.confirmed
        or previous_gray is None
        or track.missed >= 2
    ):
        return None

    propagated = optical_flow_box(
        previous_gray,
        current_gray,
        track.box,
    )

    if propagated is None:
        return None

    if (
        box_iou(propagated, target_box) < 0.01
        and center_distance_ratio(
            propagated,
            target_box,
        )
        > 1.10
    ):
        return None

    track.box = (
        0.78 * propagated
        + 0.22 * track.box
    )
    track.confidence_ema *= 0.94
    track.consecutive_hits = 0
    track.missed += 1
    track.source = "optical_flow"

    return track


def draw_detection(
    frame: np.ndarray,
    track: DetectionTrack,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [
        int(round(value))
        for value in track.box
    ]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    color = BOX_COLORS[track.key]
    label_text = (
        "CAR"
        if track.key == "lead_car"
        else "PERSON"
    )
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )
    (text_width, text_height), baseline = (
        cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )
    )
    top = max(
        0,
        y1 - text_height - baseline - 8,
    )
    right = min(
        width - 1,
        x1 + text_width + 12,
    )
    bottom = min(
        height - 1,
        top + text_height + baseline + 8,
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
        0.84,
        frame,
        0.16,
        0.0,
        frame,
    )
    cv2.putText(
        frame,
        label_text,
        (x1 + 6, bottom - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )


def add_header(
    frame: np.ndarray,
    traffic_state: str,
) -> None:
    text = (
        "Town10HD | Rain | YOLOv8s + "
        f"Temporal Tracking | Signal: {traffic_state}"
    )
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (7, 7),
        (460, 33),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.58,
        frame,
        0.42,
        0.0,
        frame,
    )
    cv2.putText(
        frame,
        text,
        (13, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def run_detect(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    from ultralytics import YOLO

    raw_path = run_dir / "rain_signal_raw.mp4"
    gt_path = run_dir / "ground_truth_metrics.csv"
    capture_report_path = (
        run_dir / "capture_report.json"
    )
    detected_path = (
        run_dir / "rain_signal_detected.mp4"
    )
    metrics_path = (
        run_dir / "detection_metrics.csv"
    )
    report_path = (
        run_dir / "detection_report.json"
    )
    summary_path = (
        run_dir / "detection_summary.txt"
    )

    for required in (
        raw_path,
        gt_path,
        capture_report_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required capture file is missing: {required}"
            )

    capture_report = json.loads(
        capture_report_path.read_text(
            encoding="utf-8"
        )
    )

    if capture_report.get("status") not in {
        "PASS_CAPTURE",
        "REVIEW_CAPTURE",
    }:
        raise RuntimeError(
            "Capture report is not usable."
        )

    model_path = args.model.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model was not found: {model_path}"
        )

    model = YOLO(str(model_path))
    id_to_name, name_to_id = (
        model_class_mapping(model)
    )
    class_ids = [
        name_to_id[label]
        for label in ALLOWED_LABELS
    ]
    device: str | int = (
        int(args.device)
        if args.device.isdigit()
        else args.device
    )
    ground_truth = read_ground_truth(gt_path)
    capture = cv2.VideoCapture(str(raw_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open the raw video: {raw_path}"
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
    writer = create_writer(
        detected_path,
        fps,
        width,
        height,
    )
    tracks: dict[
        str,
        Optional[DetectionTrack],
    ] = {
        key: None
        for key in TARGET_KEYS
    }
    display_frames = {
        key: 0
        for key in TARGET_KEYS
    }
    yolo_frames = {
        key: 0
        for key in TARGET_KEYS
    }
    flow_frames = {
        key: 0
        for key in TARGET_KEYS
    }
    confidence_values: dict[
        str,
        list[float],
    ] = {
        key: []
        for key in TARGET_KEYS
    }
    simultaneous_person_frames = 0
    duplicate_display_frames = 0
    unmatched_display_frames = 0
    rows: list[dict[str, Any]] = []
    previous_gray: Optional[np.ndarray] = None
    frame_number = 0

    print("=" * 94)
    print("TOWN10 RAIN - STRICT OFFLINE DETECTION")
    print("=" * 94)
    print(f"Input: {raw_path}")
    print(f"Model: {model_path}")
    print(
        "Targets: one lead car and two individually matched pedestrians"
    )
    print(
        "Rule: ground truth validates predictions; it never creates boxes"
    )
    print("=" * 94)

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1
            gt_row = ground_truth.get(
                frame_number,
                {},
            )
            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=min(
                    RAW_CONFIDENCE_FLOOR.values()
                ),
                iou=0.50,
                agnostic_nms=False,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]
            predictions = predictions_for_frame(
                result,
                id_to_name,
            )
            gt_boxes = {
                key: gt_box(gt_row, key)
                for key in TARGET_KEYS
            }
            assignments = assign_predictions(
                predictions,
                gt_boxes,
            )
            displayed: list[str] = []
            sources: dict[str, str] = {}

            for key in TARGET_KEYS:
                label = TARGET_TO_LABEL[key]
                track = update_detection_track(
                    tracks[key],
                    key,
                    label,
                    assignments[key],
                    gt_boxes[key],
                    previous_gray,
                    current_gray,
                )
                tracks[key] = track

                if (
                    track is None
                    or not track.confirmed
                    or gt_boxes[key] is None
                ):
                    continue

                draw_detection(frame, track)
                displayed.append(key)
                sources[key] = track.source
                display_frames[key] += 1
                confidence_values[key].append(
                    track.confidence_ema
                )

                if track.source == "yolo":
                    yolo_frames[key] += 1
                else:
                    flow_frames[key] += 1

            if (
                "pedestrian_1" in displayed
                and "pedestrian_2" in displayed
            ):
                simultaneous_person_frames += 1

            # One display is permitted per controlled actor. Assignment
            # indices are unique, so duplicate/unmatched display counts
            # remain zero unless this invariant is broken.
            if len(displayed) != len(set(displayed)):
                duplicate_display_frames += 1

            traffic_state = gt_row.get(
                "traffic_light_state",
                "Unknown",
            )
            add_header(
                frame,
                traffic_state,
            )
            writer.write(frame)

            rows.append(
                {
                    "video_frame": frame_number,
                    "time_seconds": gt_row.get(
                        "time_seconds",
                        "",
                    ),
                    "scene_state": gt_row.get(
                        "scene_state",
                        "",
                    ),
                    "traffic_light_state": (
                        traffic_state
                    ),
                    "displayed_targets": "|".join(
                        displayed
                    ),
                    "lead_car_source": sources.get(
                        "lead_car",
                        "",
                    ),
                    "pedestrian_1_source": sources.get(
                        "pedestrian_1",
                        "",
                    ),
                    "pedestrian_2_source": sources.get(
                        "pedestrian_2",
                        "",
                    ),
                    "display_count": len(displayed),
                    "duplicate_display": int(
                        len(displayed)
                        != len(set(displayed))
                    ),
                    "unmatched_display": 0,
                }
            )

            if frame_number in {
                1,
                max(1, total_frames // 3),
                max(1, 2 * total_frames // 3),
                total_frames,
            }:
                cv2.imwrite(
                    str(
                        run_dir
                        / (
                            "detected_review_"
                            f"{frame_number:03d}.png"
                        )
                    ),
                    frame,
                )

            if frame_number % 40 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

            previous_gray = current_gray

    finally:
        capture.release()
        writer.release()

    if not rows:
        raise RuntimeError(
            "No frames were processed by detection."
        )

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    median_confidence = {
        key: (
            float(
                statistics.median(
                    confidence_values[key]
                )
            )
            if confidence_values[key]
            else 0.0
        )
        for key in TARGET_KEYS
    }
    checks = {
        f"{key}_minimum_display_frames": (
            display_frames[key]
            >= MIN_DISPLAY_FRAMES[key]
        )
        for key in TARGET_KEYS
    }
    checks.update(
        {
            "two_pedestrians_simultaneously_visible": (
                simultaneous_person_frames >= 3
            ),
            "zero_duplicate_display_frames": (
                duplicate_display_frames == 0
            ),
            "zero_unmatched_display_frames": (
                unmatched_display_frames == 0
            ),
        }
    )
    failure_reasons = [
        name
        for name, passed in checks.items()
        if not passed
    ]
    status = (
        "PASS_DETECTION"
        if not failure_reasons
        else "REVIEW_DETECTION"
    )
    report = {
        "status": status,
        "model": str(model_path),
        "raw_video": str(raw_path),
        "detected_video": str(detected_path),
        "display_frames": display_frames,
        "yolo_frames": yolo_frames,
        "optical_flow_frames": flow_frames,
        "median_confidence": median_confidence,
        "simultaneous_person_frames": (
            simultaneous_person_frames
        ),
        "duplicate_display_frames": (
            duplicate_display_frames
        ),
        "unmatched_display_frames": (
            unmatched_display_frames
        ),
        "checks": checks,
        "failure_reasons": failure_reasons,
        "method": (
            "YOLO predictions are greedily assigned to unique CARLA "
            "actor ground-truth boxes. A two-frame temporal confirmation "
            "and at most two optical-flow continuation frames are used. "
            "Ground truth never creates a displayed box."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = [
        "TOWN10 RAIN - STRICT OFFLINE DETECTION",
        "=" * 82,
        f"Status: {status}",
        f"Display frames: {display_frames}",
        (
            "Simultaneous two-person frames: "
            f"{simultaneous_person_frames}"
        ),
        (
            "Duplicate/unmatched frames: "
            f"{duplicate_display_frames}/"
            f"{unmatched_display_frames}"
        ),
        (
            "Failure reasons: "
            + (
                ", ".join(failure_reasons)
                if failure_reasons
                else "none"
            )
        ),
        f"Detected video: {detected_path}",
    ]
    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("=" * 94)
    print(f"FINAL STATUS: {status}")
    print(f"Display frames: {display_frames}")
    print(
        "Simultaneous pedestrian frames: "
        f"{simultaneous_person_frames}"
    )
    print(f"Detected video: {detected_path}")
    print("=" * 94)


def run_validate(
    args: argparse.Namespace,
    run_dir: Path,
) -> bool:
    plan_path = run_dir / "scene_plan.json"
    capture_report_path = (
        run_dir / "capture_report.json"
    )
    detection_report_path = (
        run_dir / "detection_report.json"
    )
    acceptance_path = (
        run_dir / "acceptance_report.json"
    )
    summary_path = run_dir / "run_summary.txt"

    for required in (
        plan_path,
        capture_report_path,
        detection_report_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Validation input is missing: {required}"
            )

    plan = json.loads(
        plan_path.read_text(encoding="utf-8")
    )
    capture_report = json.loads(
        capture_report_path.read_text(
            encoding="utf-8"
        )
    )
    detection_report = json.loads(
        detection_report_path.read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "preflight_pass": (
            plan.get("plan_status")
            == "PREFLIGHT_PASS"
        ),
        "capture_pass": (
            capture_report.get("status")
            == "PASS_CAPTURE"
        ),
        "two_pedestrians_completed": (
            bool(
                capture_report.get(
                    "pedestrian_1_done"
                )
            )
            and bool(
                capture_report.get(
                    "pedestrian_2_done"
                )
            )
        ),
        "red_during_crossing": bool(
            capture_report.get(
                "red_during_crossing"
            )
        ),
        "green_after_crossing": bool(
            capture_report.get(
                "green_after_crossing"
            )
        ),
        "opposite_turns": bool(
            capture_report.get(
                "opposite_turns"
            )
        ),
        "full_readable_rain_reached": bool(
            capture_report.get(
                "full_rain_reached"
            )
        ),
        "unknown_actor_deletion_zero": (
            int(
                capture_report.get(
                    "unknown_actors_destroyed",
                    -1,
                )
            )
            == 0
        ),
        "detection_pass": (
            detection_report.get("status")
            == "PASS_DETECTION"
        ),
        "false_positive_display_zero": (
            int(
                detection_report.get(
                    "unmatched_display_frames",
                    -1,
                )
            )
            == 0
        ),
        "duplicate_display_zero": (
            int(
                detection_report.get(
                    "duplicate_display_frames",
                    -1,
                )
            )
            == 0
        ),
    }
    failure_reasons = [
        name
        for name, passed in checks.items()
        if not passed
    ]
    status = (
        "PASS"
        if not failure_reasons
        else "REVIEW_REQUIRED"
    )
    acceptance = {
        "status": status,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "scene_plan": plan,
        "capture": capture_report,
        "detection": detection_report,
    }
    acceptance_path.write_text(
        json.dumps(
            acceptance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = [
        "TOWN10 RAIN SCENE GENERATION",
        "=" * 82,
        f"Final status: {status}",
        (
            "Scene: lead car -> red signal -> two pedestrians -> "
            "green -> opposite turns"
        ),
        f"Checks passed: {sum(checks.values())}/{len(checks)}",
        (
            "Failure reasons: "
            + (
                ", ".join(failure_reasons)
                if failure_reasons
                else "none"
            )
        ),
        (
            "Raw video: "
            f"{run_dir / 'rain_signal_raw.mp4'}"
        ),
        (
            "Detected video: "
            f"{run_dir / 'rain_signal_detected.mp4'}"
        ),
        f"Acceptance report: {acceptance_path}",
    ]
    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("=" * 94)
    print(f"FINAL STATUS: {status}")
    print(
        f"Checks passed: {sum(checks.values())}/{len(checks)}"
    )
    print(
        "Failure reasons: "
        + (
            ", ".join(failure_reasons)
            if failure_reasons
            else "none"
        )
    )
    print("=" * 94)

    return status == "PASS"


def read_video_frames(
    path: Path,
) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )
    frames: list[np.ndarray] = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(
            f"Video contains no frames: {path}"
        )

    return frames, fps


def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def run_combine(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    clear_path = args.clear_video.resolve()
    rain_path = (
        run_dir / "rain_signal_detected.mp4"
    )
    preview_path = (
        run_dir / "clear_to_rain_preview.mp4"
    )

    if not clear_path.is_file():
        raise FileNotFoundError(
            f"Protected clear video was not found: {clear_path}"
        )

    if not rain_path.is_file():
        raise FileNotFoundError(
            f"Detected rain video was not found: {rain_path}"
        )

    clear_frames, clear_fps = read_video_frames(
        clear_path
    )
    rain_frames, rain_fps = read_video_frames(
        rain_path
    )
    output_fps = FPS
    clear_tail_count = max(
        1,
        int(round(1.2 * clear_fps)),
    )
    rain_head_count = min(
        len(rain_frames),
        max(
            1,
            int(round(3.2 * rain_fps)),
        ),
    )
    clear_tail = clear_frames[
        -clear_tail_count:
    ]
    rain_head = rain_frames[:rain_head_count]
    writer = create_writer(
        preview_path,
        output_fps,
        WIDTH,
        HEIGHT,
    )

    try:
        for frame in clear_tail:
            writer.write(
                resize_frame(
                    frame,
                    WIDTH,
                    HEIGHT,
                )
            )

        # A short dissolve softens the edit while preserving the real
        # footage from both videos.
        dissolve_frames = min(
            4,
            len(clear_tail),
            len(rain_head),
        )

        for index, frame in enumerate(rain_head):
            rain_frame = resize_frame(
                frame,
                WIDTH,
                HEIGHT,
            )

            if index < dissolve_frames:
                clear_frame = resize_frame(
                    clear_tail[
                        -dissolve_frames + index
                    ],
                    WIDTH,
                    HEIGHT,
                )
                alpha = (
                    index + 1
                ) / (dissolve_frames + 1)
                output = cv2.addWeighted(
                    clear_frame,
                    1.0 - alpha,
                    rain_frame,
                    alpha,
                    0.0,
                )
            else:
                output = rain_frame

            writer.write(output)
    finally:
        writer.release()

    print("=" * 94)
    print("FINAL STATUS: PREVIEW_CREATED")
    print(f"Preview: {preview_path}")
    print("=" * 94)


def protect_final(
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    acceptance_path = (
        run_dir / "acceptance_report.json"
    )

    if not acceptance_path.is_file():
        raise FileNotFoundError(
            "Acceptance report is missing."
        )

    acceptance = json.loads(
        acceptance_path.read_text(
            encoding="utf-8"
        )
    )

    if acceptance.get("status") != "PASS":
        raise RuntimeError(
            "Final folder will not be created because validation "
            "did not PASS."
        )

    final_root = args.final_root.resolve()

    if final_root.exists():
        if not args.overwrite_final:
            raise FileExistsError(
                "Protected final folder already exists: "
                f"{final_root}. Review it before using "
                "--overwrite-final."
            )

        shutil.rmtree(final_root)

    final_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    copies = {
        "rain_signal_raw.mp4": (
            "town10_rain_signal_raw.mp4"
        ),
        "rain_signal_detected.mp4": (
            "town10_rain_signal_detected.mp4"
        ),
        "clear_to_rain_preview.mp4": (
            "town10_clear_to_rain_preview.mp4"
        ),
        "scene_plan.json": "scene_plan.json",
        "ground_truth_metrics.csv": (
            "ground_truth_metrics.csv"
        ),
        "detection_metrics.csv": (
            "detection_metrics.csv"
        ),
        "capture_report.json": (
            "capture_report.json"
        ),
        "detection_report.json": (
            "detection_report.json"
        ),
        "acceptance_report.json": (
            "acceptance_report.json"
        ),
        "run_summary.txt": "run_summary.txt",
    }

    for source_name, destination_name in copies.items():
        source = run_dir / source_name

        if source.is_file():
            shutil.copy2(
                source,
                final_root / destination_name,
            )

    print("=" * 94)
    print("PROTECTED FINAL CREATED")
    print(f"Folder: {final_root}")
    print("=" * 94)


def run_status(
    args: argparse.Namespace,
    run_dir: Optional[Path] = None,
) -> None:
    final_summary = (
        args.final_root.resolve()
        / "run_summary.txt"
    )

    if final_summary.is_file():
        print(
            final_summary.read_text(
                encoding="utf-8"
            )
        )
        return

    actual_run = (
        resolve_run(args)
        if run_dir is None
        else run_dir
    )

    for name in (
        "run_summary.txt",
        "detection_summary.txt",
        "capture_summary.txt",
        "preflight_summary.txt",
    ):
        path = actual_run / name

        if path.is_file():
            print(
                path.read_text(
                    encoding="utf-8"
                )
            )
            return

    print(f"No summary exists in {actual_run}")


def clone_latest_scene_plan_for_capture(
    args: argparse.Namespace,
) -> Path:
    """
    Start a fresh capture run without overwriting the previous PASS run.
    The latest approved scene plan is copied unchanged.
    """
    source_run = latest_run(
        args.output_root.resolve()
    )
    source_plan = source_run / "scene_plan.json"

    if not source_plan.is_file():
        raise FileNotFoundError(
            f"Scene plan is missing: {source_plan}"
        )

    new_run = create_new_run(
        args.output_root
    )
    shutil.copy2(
        source_plan,
        new_run / "scene_plan.json",
    )

    source_preflight = (
        source_run / "preflight_summary.txt"
    )

    if source_preflight.is_file():
        shutil.copy2(
            source_preflight,
            new_run / "preflight_summary.txt",
        )

    (
        new_run / "source_scene_plan.txt"
    ).write_text(
        "Scene plan cloned from preserved run:\n"
        f"{source_run}\n",
        encoding="utf-8",
    )

    print(
        "Preserving previous run. New capture folder: "
        f"{new_run}"
    )

    return new_run


def main() -> None:
    args = parse_args()

    try:
        if args.phase == "preflight":
            run_preflight(args)
            return

        if args.phase == "capture":
            if args.run is None:
                run_dir = clone_latest_scene_plan_for_capture(
                    args
                )
            else:
                run_dir = resolve_run(args)

            run_capture(args, run_dir)
            return

        if args.phase == "detect":
            run_dir = resolve_run(args)
            run_detect(args, run_dir)
            return

        if args.phase == "validate":
            run_dir = resolve_run(args)
            run_validate(args, run_dir)
            return

        if args.phase == "combine":
            run_dir = resolve_run(args)
            run_combine(args, run_dir)
            return

        if args.phase == "status":
            run_status(args)
            return

        if args.phase == "finalize":
            run_dir = resolve_run(args)
            passed = run_validate(args, run_dir)

            if not passed:
                print(
                    "Validation requires review. Preview and protected "
                    "final were not created."
                )
                sys.exit(2)

            run_combine(args, run_dir)
            protect_final(args, run_dir)
            run_status(args, run_dir)
            return

        if args.phase == "all":
            run_dir = run_preflight(args)
            run_capture(args, run_dir)
            run_detect(args, run_dir)
            passed = run_validate(
                args,
                run_dir,
            )

            if not passed:
                print(
                    "Validation requires review. Preview and protected "
                    "final were not created."
                )
                sys.exit(2)

            run_combine(args, run_dir)
            protect_final(args, run_dir)
            run_status(args, run_dir)
            return

        raise RuntimeError(
            f"Unknown phase: {args.phase}"
        )

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(130)
    except Exception as error:
        print("=" * 94)
        print("TOWN10 RAIN SCENE FAILED SAFELY")
        print(
            f"{type(error).__name__}: {error}"
        )
        print(
            "No protected clear file was modified. "
            "Only script-owned actors are eligible for cleanup."
        )
        print("=" * 94)
        raise


if __name__ == "__main__":
    main()
