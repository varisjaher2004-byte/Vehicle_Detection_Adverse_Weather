from __future__ import annotations
import os

import argparse
import csv
import json
import math
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import carla
import cv2
import numpy as np


# =============================================================================
# STEP 90 - TOWN10 DENSE FOG MULTI-VEHICLE RAW CAPTURE
#
# FINAL LOCKED SCENE
# ------------------
# Straight Town10 corridor. No pedestrians. No signal choreography. No turns.
#
# Same-direction / going away from ego:
#   ego -> motorcycle -> car
#
# Opposite / coming toward ego:
#   bus -> motorcycle behind bus
#
# Controlled targets:
#   1. forward_motorcycle
#   2. forward_car
#   3. oncoming_bus
#   4. oncoming_motorcycle
#
# Dense fog is steady for the full recorded clip.
#
# SAFETY
# ------
# - Never deletes unknown/world actors.
# - Only actors whose role_name starts with "fog90_" are eligible for cleanup.
# - Restores original CARLA world settings and weather in finally.
# - One RGB camera only.
#
# OUTPUT
# ------
# outputs\town10_fog_multivehicle\run_YYYYMMDD_HHMMSS\
#   fog_multivehicle_raw.mp4
#   ground_truth_metrics.csv
#   capture_report.json
#   scene_plan.json
#   capture_summary.txt
#   capture_review_*.png
#
# This is CAPTURE ONLY. Stable YOLO presentation detection is Step 91 after
# the raw fog scene has been visually approved.
# =============================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_ROUTE_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "town10_reference_route"
)

DEFAULT_OUTPUT_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "town10_fog_multivehicle"
)

ROLE_PREFIX = "fog90_"

WIDTH = 640
HEIGHT = 360
FPS = 10.0
WORLD_FPS = 20.0
FOV = 90.0

# Camera kept at the lower presentation height that worked well in the final
# rain scene, while retaining the same 90-degree Town10 perspective.
CAMERA_X = 1.50
CAMERA_Y = 0.00
CAMERA_Z = 1.70
CAMERA_PITCH = 0.0

# Distances are measured forward along the locked Town10 corridor BEFORE the
# hidden acceleration period. Same-direction actors retain their spacing.
FORWARD_MOTORCYCLE_DISTANCE_M = 18.0
FORWARD_CAR_DISTANCE_M = 36.0
ONCOMING_BUS_ROUTE_DISTANCE_M = 56.0
ONCOMING_MOTORCYCLE_ROUTE_DISTANCE_M = 78.0

# Speeds keep same-direction spacing calm while the two oncoming actors arrive
# one after another.
EGO_SPEED_KMH = 16.0
FORWARD_MOTORCYCLE_SPEED_KMH = 18.0
FORWARD_CAR_SPEED_KMH = 18.0
ONCOMING_BUS_SPEED_KMH = 18.0
ONCOMING_MOTORCYCLE_SPEED_KMH = 20.0

DEFAULT_DURATION_SECONDS = 12.0

TARGET_ORDER = (
    "forward_motorcycle",
    "forward_car",
    "oncoming_bus",
    "oncoming_motorcycle",
)

TARGET_DIRECTION = {
    "forward_motorcycle": "same_direction",
    "forward_car": "same_direction",
    "oncoming_bus": "oncoming",
    "oncoming_motorcycle": "oncoming",
}

TARGET_SPEED_KMH = {
    "forward_motorcycle": FORWARD_MOTORCYCLE_SPEED_KMH,
    "forward_car": FORWARD_CAR_SPEED_KMH,
    "oncoming_bus": ONCOMING_BUS_SPEED_KMH,
    "oncoming_motorcycle": ONCOMING_MOTORCYCLE_SPEED_KMH,
}

TARGET_BLUEPRINT_PREFERENCES = {
    "forward_motorcycle": (
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.vespa.zx125",
    ),
    "forward_car": (
        "vehicle.audi.tt",
        "vehicle.lincoln.mkz_2020",
        "vehicle.mercedes.coupe_2020",
        "vehicle.tesla.model3",
    ),
    "oncoming_bus": (
        "vehicle.mitsubishi.fusorosa",
    ),
    "oncoming_motorcycle": (
        "vehicle.yamaha.yzf",
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.vespa.zx125",
    ),
}


@dataclass
class TargetPlan:
    key: str
    direction: str
    blueprint_id: str
    route_distance_m: float
    spawn_waypoint: carla.Waypoint
    speed_kmh: float
    actor: Optional[carla.Vehicle] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a straight Town10 dense-fog raw scene with "
            "motorcycle + car ahead and bus + motorcycle oncoming."
        )
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town10HD_Opt")

    parser.add_argument(
        "--route-json",
        type=Path,
        default=None,
        help=(
            "Locked Town10 route JSON. Default: latest "
            "outputs/town10_reference_route/run_*/locked_town10_reference_route.json"
        ),
    )

    parser.add_argument(
        "--route-rank",
        type=int,
        default=2,
        choices=(1, 2, 3),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
    )

    parser.add_argument(
        "--fog-density",
        type=float,
        default=42.0,
        help="CARLA fog density. Default 42 = visibly dense but readable.",
    )

    parser.add_argument(
        "--fog-distance",
        type=float,
        default=10.0,
        help="Distance in metres before dense fog contribution begins.",
    )

    parser.add_argument(
        "--fog-falloff",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    return parser.parse_args()


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def angle_difference(
    first: float,
    second: float,
) -> float:
    return abs(
        normalize_angle(
            first - second
        )
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
            "No locked Town10 route JSON exists under "
            f"{DEFAULT_ROUTE_ROOT}"
        )

    return candidates[0]


def load_locked_transform(
    path: Path,
    route_rank: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    carla.Transform,
]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get("status")
        != "REFERENCE_ROUTE_LOCKED"
    ):
        raise RuntimeError(
            f"Town10 route JSON is not locked: {path}"
        )

    top_matches = payload.get(
        "top_matches"
    )

    if (
        not isinstance(
            top_matches,
            list,
        )
        or len(top_matches)
        < route_rank
    ):
        raise RuntimeError(
            f"Route rank {route_rank} is unavailable."
        )

    selected = top_matches[
        route_rank - 1
    ]

    transform_data = selected[
        "transform"
    ]
    location = transform_data[
        "location"
    ]
    rotation = transform_data[
        "rotation"
    ]

    transform = carla.Transform(
        carla.Location(
            x=float(location["x"]),
            y=float(location["y"]),
            z=float(location["z"]),
        ),
        carla.Rotation(
            pitch=float(
                rotation["pitch"]
            ),
            yaw=float(
                rotation["yaw"]
            ),
            roll=float(
                rotation["roll"]
            ),
        ),
    )

    return (
        payload,
        selected,
        transform,
    )


def dense_fog_weather(
    density: float,
    distance: float,
    falloff: float,
) -> carla.WeatherParameters:
    """
    Dense research fog while retaining enough ambient light for the custom
    detector to see the controlled vehicles.

    No rain, no wet-road dependency and no signal-specific lighting.
    """

    return carla.WeatherParameters(
        cloudiness=72.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=3.0,
        sun_azimuth_angle=35.0,
        sun_altitude_angle=38.0,
        fog_density=clamp(
            density,
            0.0,
            100.0,
        ),
        fog_distance=max(
            0.0,
            distance,
        ),
        fog_falloff=max(
            0.05,
            falloff,
        ),
        wetness=5.0,
        scattering_intensity=0.90,
        mie_scattering_scale=0.025,
        rayleigh_scattering_scale=0.0331,
    )


def advance_straight(
    start: carla.Waypoint,
    distance_m: float,
    step_m: float = 2.0,
    maximum_corridor_yaw_change: float = 18.0,
) -> carla.Waypoint:
    """
    Same straight-corridor idea as the accepted clear capture:
    always select the next driving waypoint closest to the initial yaw.
    """

    current = start
    travelled = 0.0
    start_yaw = float(
        start.transform.rotation.yaw
    )

    visited: list[
        carla.Location
    ] = [
        start.transform.location
    ]

    while travelled < distance_m:
        options = [
            waypoint
            for waypoint
            in current.next(step_m)
            if waypoint.lane_type
            == carla.LaneType.Driving
        ]

        if not options:
            raise RuntimeError(
                "Straight Town10 corridor ended after "
                f"{travelled:.1f} m; needed {distance_m:.1f} m."
            )

        next_waypoint = min(
            options,
            key=lambda waypoint: (
                angle_difference(
                    waypoint.transform.rotation.yaw,
                    start_yaw,
                )
            ),
        )

        yaw_change = angle_difference(
            next_waypoint.transform.rotation.yaw,
            start_yaw,
        )

        if (
            yaw_change
            > maximum_corridor_yaw_change
        ):
            raise RuntimeError(
                "Locked corridor would require a turn before "
                f"{distance_m:.1f} m. Observed yaw change "
                f"{yaw_change:.1f}°."
            )

        next_location = (
            next_waypoint.transform.location
        )

        if any(
            next_location.distance(
                old_location
            )
            < 1.0
            for old_location
            in visited[:-8]
        ):
            raise RuntimeError(
                "Route loop detected while building fog scene."
            )

        travelled += (
            current.transform.location.distance(
                next_location
            )
        )

        visited.append(
            next_location
        )
        current = next_waypoint

    return current


def find_oncoming_waypoint(
    all_waypoints: list[
        carla.Waypoint
    ],
    route_waypoint: carla.Waypoint,
) -> carla.Waypoint:
    """
    Find the nearest driving lane travelling approximately 180 degrees
    opposite to the locked corridor.
    """

    route_location = (
        route_waypoint.transform.location
    )
    route_yaw = float(
        route_waypoint.transform.rotation.yaw
    )

    candidates: list[
        tuple[
            float,
            carla.Waypoint,
        ]
    ] = []

    for waypoint in all_waypoints:
        if (
            waypoint.lane_type
            != carla.LaneType.Driving
        ):
            continue

        distance = (
            waypoint.transform.location.distance(
                route_location
            )
        )

        if distance > 10.0:
            continue

        yaw_difference = (
            angle_difference(
                waypoint.transform.rotation.yaw,
                route_yaw,
            )
        )

        if yaw_difference < 135.0:
            continue

        score = (
            distance
            + abs(
                180.0
                - yaw_difference
            )
            * 0.04
            + (
                6.0
                if waypoint.is_junction
                else 0.0
            )
        )

        candidates.append(
            (
                score,
                waypoint,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No safe oncoming driving lane found near "
            f"{route_location}."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


def raised_transform(
    transform: carla.Transform,
    z_offset: float = 0.45,
) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(
                transform.location.x
            ),
            y=float(
                transform.location.y
            ),
            z=float(
                transform.location.z
                + z_offset
            ),
        ),
        carla.Rotation(
            pitch=float(
                transform.rotation.pitch
            ),
            yaw=float(
                transform.rotation.yaw
            ),
            roll=float(
                transform.rotation.roll
            ),
        ),
    )


def blueprint_exists(
    library: carla.BlueprintLibrary,
    blueprint_id: str,
) -> bool:
    try:
        library.find(
            blueprint_id
        )
        return True
    except RuntimeError:
        return False


def choose_blueprint(
    library: carla.BlueprintLibrary,
    preferences: tuple[
        str,
        ...,
    ],
    fallback_pattern: str,
    used_ids: set[str],
) -> carla.ActorBlueprint:
    for blueprint_id in preferences:
        if (
            blueprint_id
            in used_ids
        ):
            continue

        if blueprint_exists(
            library,
            blueprint_id,
        ):
            used_ids.add(
                blueprint_id
            )
            return library.find(
                blueprint_id
            )

    fallback = [
        blueprint
        for blueprint
        in library.filter(
            fallback_pattern
        )
        if blueprint.id
        not in used_ids
    ]

    if not fallback:
        raise RuntimeError(
            "No usable CARLA blueprint found for "
            f"{preferences} / {fallback_pattern}"
        )

    blueprint = fallback[0]
    used_ids.add(
        blueprint.id
    )
    return blueprint


def configure_blueprint(
    blueprint: carla.ActorBlueprint,
    role_name: str,
    color_index: int = 0,
) -> None:
    if blueprint.has_attribute(
        "role_name"
    ):
        blueprint.set_attribute(
            "role_name",
            role_name,
        )

    if blueprint.has_attribute(
        "color"
    ):
        values = (
            blueprint.get_attribute(
                "color"
            ).recommended_values
        )

        if values:
            blueprint.set_attribute(
                "color",
                values[
                    color_index
                    % len(values)
                ],
            )


def actor_role_name(
    actor: carla.Actor,
) -> str:
    return str(
        actor.attributes.get(
            "role_name",
            "",
        )
    )


def cleanup_own_leftovers(
    world: carla.World,
) -> int:
    """
    Safe cleanup only. Never destroy unknown actors.
    """

    destroyed = 0

    for actor in world.get_actors():
        if not actor_role_name(
            actor
        ).startswith(
            ROLE_PREFIX
        ):
            continue

        try:
            actor.destroy()
            destroyed += 1
        except RuntimeError:
            pass

    return destroyed


def configure_autopilot(
    traffic_manager: carla.TrafficManager,
    actor: carla.Vehicle,
    speed_kmh: float,
    tm_port: int,
) -> None:
    actor.set_autopilot(
        True,
        tm_port,
    )

    traffic_manager.ignore_lights_percentage(
        actor,
        100.0,
    )

    traffic_manager.ignore_signs_percentage(
        actor,
        100.0,
    )

    traffic_manager.auto_lane_change(
        actor,
        False,
    )

    traffic_manager.distance_to_leading_vehicle(
        actor,
        7.0,
    )

    try:
        traffic_manager.set_desired_speed(
            actor,
            speed_kmh,
        )
    except (
        AttributeError,
        RuntimeError,
    ):
        # Compatibility fallback. Normal execution on CARLA 0.9.16 should use
        # set_desired_speed above.
        traffic_manager.vehicle_percentage_speed_difference(
            actor,
            0.0,
        )


def speed_mps(
    actor: carla.Actor,
) -> float:
    velocity = actor.get_velocity()

    return math.sqrt(
        float(
            velocity.x
        )
        ** 2
        + float(
            velocity.y
        )
        ** 2
        + float(
            velocity.z
        )
        ** 2
    )


def image_to_bgr(
    image: carla.Image,
) -> np.ndarray:
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )

    return array[
        :,
        :,
        :3,
    ].copy()


def create_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
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
            f"Could not create MP4: {path}"
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
            math.radians(
                fov_degrees
            )
            / 2.0
        )
    )

    matrix = np.identity(
        3,
        dtype=np.float64,
    )

    matrix[
        0,
        0,
    ] = focal
    matrix[
        1,
        1,
    ] = focal
    matrix[
        0,
        2,
    ] = width / 2.0
    matrix[
        1,
        2,
    ] = height / 2.0

    return matrix


def world_to_camera(
    location: carla.Location,
    camera: carla.Sensor,
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

    sensor_point = (
        inverse
        @ vector
    )

    return np.array(
        [
            sensor_point[1],
            -sensor_point[2],
            sensor_point[0],
        ],
        dtype=np.float64,
    )


def project_actor_box(
    actor: carla.Actor,
    camera: carla.Sensor,
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
    float,
    float,
]:
    """
    Project the full CARLA actor bounding box.

    Returns:
        visible, x1, y1, x2, y2, u, v, depth_m
    """

    try:
        vertices = (
            actor.bounding_box.get_world_vertices(
                actor.get_transform()
            )
        )
    except (
        AttributeError,
        RuntimeError,
    ):
        nan = float("nan")
        return (
            False,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
        )

    projected: list[
        tuple[
            float,
            float,
            float,
        ]
    ] = []

    depths = []

    for vertex in vertices:
        camera_point = (
            world_to_camera(
                vertex,
                camera,
            )
        )

        depth = float(
            camera_point[2]
        )

        if depth <= 0.10:
            continue

        pixel = (
            intrinsic
            @ camera_point
        )

        u = float(
            pixel[0]
            / pixel[2]
        )

        v = float(
            pixel[1]
            / pixel[2]
        )

        projected.append(
            (
                u,
                v,
                depth,
            )
        )
        depths.append(
            depth
        )

    if len(projected) < 4:
        nan = float("nan")
        return (
            False,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
        )

    xs = [
        item[0]
        for item in projected
    ]

    ys = [
        item[1]
        for item in projected
    ]

    raw_x1 = min(xs)
    raw_y1 = min(ys)
    raw_x2 = max(xs)
    raw_y2 = max(ys)

    # Visibility means at least part of the projected actor intersects the
    # camera image and the actor is not extremely distant.
    median_depth = float(
        np.median(
            np.asarray(
                depths,
                dtype=np.float64,
            )
        )
    )

    intersects = (
        raw_x2 >= 0.0
        and raw_x1 <= width - 1.0
        and raw_y2 >= 0.0
        and raw_y1 <= height - 1.0
    )

    visible = (
        intersects
        and median_depth <= 80.0
    )

    x1 = clamp(
        raw_x1,
        0.0,
        width - 1.0,
    )

    y1 = clamp(
        raw_y1,
        0.0,
        height - 1.0,
    )

    x2 = clamp(
        raw_x2,
        0.0,
        width - 1.0,
    )

    y2 = clamp(
        raw_y2,
        0.0,
        height - 1.0,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        visible = False

    u = (
        x1 + x2
    ) / 2.0

    v = (
        y1 + y2
    ) / 2.0

    return (
        bool(visible),
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        float(u),
        float(v),
        median_depth,
    )


def destroy_actor(
    actor: Optional[
        carla.Actor
    ],
) -> None:
    if actor is None:
        return

    try:
        actor.destroy()
    except RuntimeError:
        pass


def create_target_plans(
    carla_map: carla.Map,
    start_waypoint: carla.Waypoint,
    library: carla.BlueprintLibrary,
) -> list[TargetPlan]:
    """
    Build exactly:
      forward motorcycle @ 18m
      forward car        @ 36m
      oncoming bus       near route point 56m
      oncoming motorcycle behind bus near route point 78m
    """

    all_waypoints = (
        carla_map.generate_waypoints(
            2.0
        )
    )

    used_blueprint_ids: set[
        str
    ] = set()

    forward_motorcycle_wp = (
        advance_straight(
            start_waypoint,
            FORWARD_MOTORCYCLE_DISTANCE_M,
        )
    )

    forward_car_wp = (
        advance_straight(
            start_waypoint,
            FORWARD_CAR_DISTANCE_M,
        )
    )

    bus_route_wp = (
        advance_straight(
            start_waypoint,
            ONCOMING_BUS_ROUTE_DISTANCE_M,
        )
    )

    oncoming_motorcycle_route_wp = (
        advance_straight(
            start_waypoint,
            ONCOMING_MOTORCYCLE_ROUTE_DISTANCE_M,
        )
    )

    oncoming_bus_wp = (
        find_oncoming_waypoint(
            all_waypoints,
            bus_route_wp,
        )
    )

    oncoming_motorcycle_wp = (
        find_oncoming_waypoint(
            all_waypoints,
            oncoming_motorcycle_route_wp,
        )
    )

    blueprint_patterns = {
        "forward_motorcycle": "vehicle.*",
        "forward_car": "vehicle.*",
        "oncoming_bus": "vehicle.*",
        "oncoming_motorcycle": "vehicle.*",
    }

    waypoint_by_key = {
        "forward_motorcycle": (
            forward_motorcycle_wp
        ),
        "forward_car": (
            forward_car_wp
        ),
        "oncoming_bus": (
            oncoming_bus_wp
        ),
        "oncoming_motorcycle": (
            oncoming_motorcycle_wp
        ),
    }

    distance_by_key = {
        "forward_motorcycle": (
            FORWARD_MOTORCYCLE_DISTANCE_M
        ),
        "forward_car": (
            FORWARD_CAR_DISTANCE_M
        ),
        "oncoming_bus": (
            ONCOMING_BUS_ROUTE_DISTANCE_M
        ),
        "oncoming_motorcycle": (
            ONCOMING_MOTORCYCLE_ROUTE_DISTANCE_M
        ),
    }

    plans = []

    for key in TARGET_ORDER:
        blueprint = (
            choose_blueprint(
                library,
                TARGET_BLUEPRINT_PREFERENCES[
                    key
                ],
                blueprint_patterns[
                    key
                ],
                used_blueprint_ids,
            )
        )

        plans.append(
            TargetPlan(
                key=key,
                direction=TARGET_DIRECTION[
                    key
                ],
                blueprint_id=blueprint.id,
                route_distance_m=(
                    distance_by_key[
                        key
                    ]
                ),
                spawn_waypoint=(
                    waypoint_by_key[
                        key
                    ]
                ),
                speed_kmh=TARGET_SPEED_KMH[
                    key
                ],
            )
        )

    # Geometry sanity checks.
    forward_gap = (
        forward_motorcycle_wp.transform.location.distance(
            forward_car_wp.transform.location
        )
    )

    oncoming_gap = (
        oncoming_bus_wp.transform.location.distance(
            oncoming_motorcycle_wp.transform.location
        )
    )

    if forward_gap < 12.0:
        raise RuntimeError(
            "Forward motorcycle/car spacing is too small: "
            f"{forward_gap:.1f} m."
        )

    if oncoming_gap < 12.0:
        raise RuntimeError(
            "Oncoming bus/motorcycle spacing is too small: "
            f"{oncoming_gap:.1f} m."
        )

    return plans


def plan_to_json(
    route_json: Path,
    route_rank: int,
    driving_route: dict[str, Any],
    start_waypoint: carla.Waypoint,
    targets: list[TargetPlan],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "status": "FOG_SCENE_PLAN_READY",
        "map": "Town10HD_Opt",
        "route_json": str(
            route_json
        ),
        "route_rank": route_rank,
        "driving_spawn_index": (
            driving_route.get(
                "spawn_index"
            )
        ),
        "driving_road_id": (
            driving_route.get(
                "road_id"
            )
        ),
        "scene": (
            "straight dense fog; ego -> motorcycle -> car; "
            "oncoming bus -> motorcycle behind"
        ),
        "no_pedestrians": True,
        "no_signal_choreography": True,
        "no_turn_choreography": True,
        "camera": {
            "x": CAMERA_X,
            "y": CAMERA_Y,
            "z": CAMERA_Z,
            "pitch": CAMERA_PITCH,
            "fov": FOV,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
        },
        "fog": {
            "density": args.fog_density,
            "distance": args.fog_distance,
            "falloff": args.fog_falloff,
        },
        "ego": {
            "speed_kmh": EGO_SPEED_KMH,
            "spawn": {
                "x": float(
                    start_waypoint.transform.location.x
                ),
                "y": float(
                    start_waypoint.transform.location.y
                ),
                "z": float(
                    start_waypoint.transform.location.z
                ),
                "yaw": float(
                    start_waypoint.transform.rotation.yaw
                ),
            },
        },
        "targets": [
            {
                "key": plan.key,
                "direction": plan.direction,
                "blueprint": (
                    plan.blueprint_id
                ),
                "route_distance_m": (
                    plan.route_distance_m
                ),
                "speed_kmh": (
                    plan.speed_kmh
                ),
                "spawn": {
                    "x": float(
                        plan.spawn_waypoint.transform.location.x
                    ),
                    "y": float(
                        plan.spawn_waypoint.transform.location.y
                    ),
                    "z": float(
                        plan.spawn_waypoint.transform.location.z
                    ),
                    "yaw": float(
                        plan.spawn_waypoint.transform.rotation.yaw
                    ),
                    "road_id": int(
                        plan.spawn_waypoint.road_id
                    ),
                    "lane_id": int(
                        plan.spawn_waypoint.lane_id
                    ),
                },
            }
            for plan in targets
        ],
    }


def main() -> None:
    args = parse_args()

    if args.duration <= 1.0:
        raise ValueError(
            "--duration must be greater than 1 second."
        )

    route_json = (
        args.route_json.resolve()
        if args.route_json
        else latest_route_json()
    )

    if not route_json.is_file():
        raise FileNotFoundError(
            f"Locked route JSON not found: {route_json}"
        )

    (
        route_payload,
        driving_route,
        locked_transform,
    ) = load_locked_transform(
        route_json,
        args.route_rank,
    )

    client = carla.Client(
        args.host,
        args.port,
    )
    client.set_timeout(
        90.0
    )

    world = client.get_world()
    active_map = (
        world.get_map().name
    )

    if not active_map.endswith(
        f"/{args.town}"
    ):
        raise RuntimeError(
            f"Active CARLA map is {active_map}; "
            f"expected {args.town}."
        )

    original_settings = (
        world.get_settings()
    )
    original_weather = (
        world.get_weather()
    )

    traffic_manager = (
        client.get_trafficmanager(
            args.tm_port
        )
    )

    ego: Optional[
        carla.Vehicle
    ] = None

    camera: Optional[
        carla.Sensor
    ] = None

    target_plans: list[
        TargetPlan
    ] = []

    writer: Optional[
        cv2.VideoWriter
    ] = None

    frame_queue: queue.Queue[
        tuple[
            int,
            float,
            np.ndarray,
        ]
    ] = queue.Queue(
        maxsize=12
    )

    unknown_actors_destroyed = 0
    own_leftovers_destroyed = 0

    run_dir: Optional[
        Path
    ] = None

    print("=" * 100)
    print(
        "STEP 90 - TOWN10 DENSE FOG MULTI-VEHICLE CAPTURE"
    )
    print("=" * 100)
    print(
        f"Map:            {active_map}"
    )
    print(
        f"Route JSON:     {route_json}"
    )
    print(
        f"Route rank:     {args.route_rank}"
    )
    print(
        "Scene:          straight only | no signal | no pedestrians"
    )
    print(
        "Forward:        ego -> motorcycle -> car"
    )
    print(
        "Oncoming:       bus -> motorcycle behind bus"
    )
    print(
        "Fog:            "
        f"density={args.fog_density:.1f} | "
        f"distance={args.fog_distance:.1f}m | "
        f"falloff={args.fog_falloff:.2f}"
    )
    print(
        f"Camera:         z={CAMERA_Z:.2f}m | FOV={FOV:.0f} | "
        f"{WIDTH}x{HEIGHT}@{FPS:.0f}"
    )
    print(
        "Safety:         only fog90_* actors may be destroyed"
    )
    print("=" * 100)

    try:
        # Clean only leftovers created by this exact fog script on a prior
        # interrupted attempt.
        own_leftovers_destroyed = (
            cleanup_own_leftovers(
                world
            )
        )

        if own_leftovers_destroyed:
            print(
                "Removed own fog90_* leftovers: "
                f"{own_leftovers_destroyed}"
            )

        settings = (
            world.get_settings()
        )
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0
            / WORLD_FPS
        )
        settings.no_rendering_mode = False
        world.apply_settings(
            settings
        )

        traffic_manager.set_synchronous_mode(
            True
        )
        traffic_manager.set_random_device_seed(
            9007
        )

        world.set_weather(
            dense_fog_weather(
                args.fog_density,
                args.fog_distance,
                args.fog_falloff,
            )
        )

        carla_map = (
            world.get_map()
        )

        start_waypoint = (
            carla_map.get_waypoint(
                locked_transform.location,
                project_to_road=True,
                lane_type=(
                    carla.LaneType.Driving
                ),
            )
        )

        if start_waypoint is None:
            raise RuntimeError(
                "Locked Town10 transform could not be projected "
                "to a driving waypoint."
            )

        # Validate enough straight corridor for the whole staging layout.
        _ = advance_straight(
            start_waypoint,
            92.0,
        )

        library = (
            world.get_blueprint_library()
        )

        target_plans = (
            create_target_plans(
                carla_map,
                start_waypoint,
                library,
            )
        )

        # Create output only AFTER geometry preflight passes.
        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        run_dir = (
            args.output_root.resolve()
            / f"run_{timestamp}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        video_path = (
            run_dir
            / "fog_multivehicle_raw.mp4"
        )

        metrics_path = (
            run_dir
            / "ground_truth_metrics.csv"
        )

        report_path = (
            run_dir
            / "capture_report.json"
        )

        plan_path = (
            run_dir
            / "scene_plan.json"
        )

        summary_path = (
            run_dir
            / "capture_summary.txt"
        )

        plan_payload = (
            plan_to_json(
                route_json,
                args.route_rank,
                driving_route,
                start_waypoint,
                target_plans,
                args,
            )
        )

        plan_path.write_text(
            json.dumps(
                plan_payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # Spawn ego
        # ------------------------------------------------------------------
        ego_blueprint = (
            choose_blueprint(
                library,
                (
                    "vehicle.tesla.model3",
                    "vehicle.lincoln.mkz_2020",
                    "vehicle.audi.tt",
                ),
                "vehicle.*",
                used_ids=set(),
            )
        )

        configure_blueprint(
            ego_blueprint,
            f"{ROLE_PREFIX}ego",
            color_index=0,
        )

        ego = world.try_spawn_actor(
            ego_blueprint,
            raised_transform(
                start_waypoint.transform,
                0.45,
            ),
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn fog ego vehicle on locked route."
            )

        # ------------------------------------------------------------------
        # Spawn exactly four controlled target actors
        # ------------------------------------------------------------------
        for index, plan in enumerate(
            target_plans,
            start=1,
        ):
            blueprint = (
                library.find(
                    plan.blueprint_id
                )
            )

            configure_blueprint(
                blueprint,
                f"{ROLE_PREFIX}{plan.key}",
                color_index=index,
            )

            actor = world.try_spawn_actor(
                blueprint,
                raised_transform(
                    plan.spawn_waypoint.transform,
                    0.45,
                ),
            )

            if actor is None:
                raise RuntimeError(
                    f"Could not spawn {plan.key} "
                    f"({plan.blueprint_id})."
                )

            plan.actor = actor

        # ------------------------------------------------------------------
        # RGB camera
        # ------------------------------------------------------------------
        camera_blueprint = (
            library.find(
                "sensor.camera.rgb"
            )
        )

        if camera_blueprint.has_attribute(
            "role_name"
        ):
            camera_blueprint.set_attribute(
                "role_name",
                f"{ROLE_PREFIX}camera",
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

        def callback(
            image: carla.Image,
        ) -> None:
            item = (
                int(
                    image.frame
                ),
                float(
                    image.timestamp
                ),
                image_to_bgr(
                    image
                ),
            )

            try:
                frame_queue.put_nowait(
                    item
                )
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    frame_queue.put_nowait(
                        item
                    )
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

        camera.listen(
            callback
        )

        # A few ticks settle spawn physics. Motorcycles are then immediately
        # moved by Traffic Manager; no long stationary motorcycle warm-up.
        for _ in range(6):
            world.tick()

        configure_autopilot(
            traffic_manager,
            ego,
            EGO_SPEED_KMH,
            args.tm_port,
        )

        for plan in target_plans:
            assert (
                plan.actor
                is not None
            )

            configure_autopilot(
                traffic_manager,
                plan.actor,
                plan.speed_kmh,
                args.tm_port,
            )

        # Hidden one-second motion period. The first recorded frame therefore
        # starts naturally in motion and the first oncoming bus is still at a
        # useful fog distance.
        hidden_ticks = int(
            WORLD_FPS
            * 1.0
        )

        for _ in range(
            hidden_ticks
        ):
            world.tick()

        # Flush all hidden camera frames.
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        writer = create_writer(
            video_path,
            FPS,
            WIDTH,
            HEIGHT,
        )

        intrinsic = (
            camera_intrinsic(
                WIDTH,
                HEIGHT,
                FOV,
            )
        )

        required_frames = int(
            round(
                args.duration
                * FPS
            )
        )

        rows: list[
            dict[
                str,
                Any,
            ]
        ] = []

        visible_counts = {
            key: 0
            for key
            in TARGET_ORDER
        }

        encounter_counts = {
            key: 0
            for key
            in TARGET_ORDER
        }

        first_visible_frame: dict[
            str,
            Optional[int],
        ] = {
            key: None
            for key
            in TARGET_ORDER
        }

        first_encounter_frame: dict[
            str,
            Optional[int],
        ] = {
            key: None
            for key
            in TARGET_ORDER
        }

        simultaneous_forward_oncoming_frames = 0
        two_or_more_visible_frames = 0
        three_or_more_visible_frames = 0

        initial_ego_yaw = float(
            ego.get_transform().rotation.yaw
        )

        maximum_ego_yaw_change = 0.0

        recorded = 0
        capture_start_timestamp: Optional[
            float
        ] = None

        review_frames = {
            1,
            max(
                1,
                required_frames // 4,
            ),
            max(
                1,
                required_frames // 2,
            ),
            max(
                1,
                3
                * required_frames
                // 4,
            ),
            required_frames,
        }

        while recorded < required_frames:
            world.tick()

            available = []

            while True:
                try:
                    available.append(
                        frame_queue.get_nowait()
                    )
                except queue.Empty:
                    break

            if not available:
                continue

            current_ego_yaw = float(
                ego.get_transform().rotation.yaw
            )

            maximum_ego_yaw_change = max(
                maximum_ego_yaw_change,
                angle_difference(
                    current_ego_yaw,
                    initial_ego_yaw,
                ),
            )

            for (
                carla_frame,
                image_timestamp,
                frame,
            ) in available:
                if recorded >= required_frames:
                    break

                if (
                    capture_start_timestamp
                    is None
                ):
                    capture_start_timestamp = (
                        image_timestamp
                    )

                recorded += 1

                writer.write(
                    frame
                )

                row: dict[
                    str,
                    Any,
                ] = {
                    "video_frame": (
                        recorded
                    ),
                    "carla_frame": (
                        carla_frame
                    ),
                    "time_seconds": (
                        image_timestamp
                        - capture_start_timestamp
                    ),
                    "scene_state": (
                        "DENSE_FOG_STRAIGHT"
                    ),
                    "fog_density": (
                        args.fog_density
                    ),
                    "fog_distance_m": (
                        args.fog_distance
                    ),
                    "fog_falloff": (
                        args.fog_falloff
                    ),
                    "ego_speed_mps": (
                        speed_mps(
                            ego
                        )
                    ),
                    "ego_yaw": (
                        current_ego_yaw
                    ),
                }

                visible_now = []
                forward_visible = False
                oncoming_visible = False

                for plan in target_plans:
                    assert (
                        plan.actor
                        is not None
                    )

                    (
                        visible,
                        x1,
                        y1,
                        x2,
                        y2,
                        u,
                        v,
                        depth,
                    ) = project_actor_box(
                        plan.actor,
                        camera,
                        intrinsic,
                        WIDTH,
                        HEIGHT,
                    )

                    prefix = (
                        plan.key
                    )

                    row[
                        f"{prefix}_visible"
                    ] = int(
                        visible
                    )

                    row[
                        f"{prefix}_x1"
                    ] = x1

                    row[
                        f"{prefix}_y1"
                    ] = y1

                    row[
                        f"{prefix}_x2"
                    ] = x2

                    row[
                        f"{prefix}_y2"
                    ] = y2

                    row[
                        f"{prefix}_u"
                    ] = u

                    row[
                        f"{prefix}_v"
                    ] = v

                    row[
                        f"{prefix}_depth_m"
                    ] = depth

                    row[
                        f"{prefix}_speed_mps"
                    ] = (
                        speed_mps(
                            plan.actor
                        )
                    )

                    row[
                        f"{prefix}_yaw"
                    ] = float(
                        plan.actor.get_transform().rotation.yaw
                    )

                    # Meaningful encounter: actor lies inside image and is at
                    # a distance where dense-fog detection is a useful test.
                    encounter = (
                        visible
                        and 3.0
                        <= depth
                        <= 45.0
                        and 0.0
                        <= u
                        <= WIDTH
                        and 0.0
                        <= v
                        <= HEIGHT
                        and (
                            x2 - x1
                        )
                        >= 3.0
                        and (
                            y2 - y1
                        )
                        >= 4.0
                    )

                    row[
                        f"{prefix}_encounter"
                    ] = int(
                        encounter
                    )

                    if visible:
                        visible_counts[
                            prefix
                        ] += 1

                        visible_now.append(
                            prefix
                        )

                        if (
                            first_visible_frame[
                                prefix
                            ]
                            is None
                        ):
                            first_visible_frame[
                                prefix
                            ] = (
                                recorded
                            )

                        if (
                            plan.direction
                            == "same_direction"
                        ):
                            forward_visible = True
                        else:
                            oncoming_visible = True

                    if encounter:
                        encounter_counts[
                            prefix
                        ] += 1

                        if (
                            first_encounter_frame[
                                prefix
                            ]
                            is None
                        ):
                            first_encounter_frame[
                                prefix
                            ] = (
                                recorded
                            )

                visible_count = len(
                    visible_now
                )

                row[
                    "controlled_visible_count"
                ] = visible_count

                row[
                    "forward_and_oncoming_simultaneous"
                ] = int(
                    forward_visible
                    and oncoming_visible
                )

                if (
                    forward_visible
                    and oncoming_visible
                ):
                    simultaneous_forward_oncoming_frames += 1

                if visible_count >= 2:
                    two_or_more_visible_frames += 1

                if visible_count >= 3:
                    three_or_more_visible_frames += 1

                rows.append(
                    row
                )

                if (
                    recorded
                    in review_frames
                ):
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
                    recorded
                    % 30
                    == 0
                ):
                    print(
                        f"Captured {recorded}/{required_frames} frames | "
                        f"visible={visible_now}"
                    )

        if not rows:
            raise RuntimeError(
                "RGB camera produced no fog frames."
            )

        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer_csv = (
                csv.DictWriter(
                    handle,
                    fieldnames=list(
                        rows[0].keys()
                    ),
                )
            )

            writer_csv.writeheader()
            writer_csv.writerows(
                rows
            )

        minimum_encounter_frames = max(
            5,
            int(
                FPS
                * 0.5
            ),
        )

        all_targets_encountered = all(
            encounter_counts[
                key
            ]
            >= minimum_encounter_frames
            for key
            in TARGET_ORDER
        )

        simultaneous_pass = (
            simultaneous_forward_oncoming_frames
            >= max(
                6,
                int(
                    FPS
                    * 0.8
                ),
            )
        )

        straight_pass = (
            maximum_ego_yaw_change
            <= 20.0
        )

        sufficient_frames = (
            recorded
            == required_frames
        )

        checks = {
            "all_four_targets_encountered": (
                all_targets_encountered
            ),
            "forward_and_oncoming_visible_together": (
                simultaneous_pass
            ),
            "at_least_two_controlled_visible": (
                two_or_more_visible_frames
                >= max(
                    10,
                    int(
                        FPS
                        * 1.5
                    ),
                )
            ),
            "straight_no_turn": (
                straight_pass
            ),
            "full_frame_count": (
                sufficient_frames
            ),
            "unknown_actor_deletion_zero": (
                unknown_actors_destroyed
                == 0
            ),
        }

        failure_reasons = [
            name
            for (
                name,
                passed,
            )
            in checks.items()
            if not passed
        ]

        status = (
            "PASS_CAPTURE"
            if not failure_reasons
            else "REVIEW_CAPTURE"
        )

        report = {
            "status": status,
            "map": active_map,
            "scene": (
                "straight dense fog; no pedestrians; no signal; "
                "forward motorcycle + car; oncoming bus + motorcycle"
            ),
            "route_json": str(
                route_json
            ),
            "route_rank": (
                args.route_rank
            ),
            "driving_spawn_index": (
                driving_route.get(
                    "spawn_index"
                )
            ),
            "driving_road_id": (
                driving_route.get(
                    "road_id"
                )
            ),
            "resolution": [
                WIDTH,
                HEIGHT,
            ],
            "fps": FPS,
            "world_fps": (
                WORLD_FPS
            ),
            "duration_seconds": (
                recorded
                / FPS
            ),
            "frame_count": (
                recorded
            ),
            "fog": {
                "density": (
                    args.fog_density
                ),
                "distance_m": (
                    args.fog_distance
                ),
                "falloff": (
                    args.fog_falloff
                ),
            },
            "camera": {
                "x": CAMERA_X,
                "y": CAMERA_Y,
                "z": CAMERA_Z,
                "pitch": (
                    CAMERA_PITCH
                ),
                "fov": FOV,
            },
            "ego_speed_kmh": (
                EGO_SPEED_KMH
            ),
            "first_visible_frame": (
                first_visible_frame
            ),
            "visible_frame_counts": (
                visible_counts
            ),
            "first_encounter_frame": (
                first_encounter_frame
            ),
            "encounter_frame_counts": (
                encounter_counts
            ),
            "simultaneous_forward_oncoming_frames": (
                simultaneous_forward_oncoming_frames
            ),
            "two_or_more_visible_frames": (
                two_or_more_visible_frames
            ),
            "three_or_more_visible_frames": (
                three_or_more_visible_frames
            ),
            "maximum_ego_yaw_change_deg": (
                maximum_ego_yaw_change
            ),
            "own_leftovers_destroyed": (
                own_leftovers_destroyed
            ),
            "unknown_actors_destroyed": (
                unknown_actors_destroyed
            ),
            "checks": checks,
            "failure_reasons": (
                failure_reasons
            ),
            "targets": [
                {
                    "key": (
                        plan.key
                    ),
                    "direction": (
                        plan.direction
                    ),
                    "blueprint": (
                        plan.blueprint_id
                    ),
                    "speed_kmh": (
                        plan.speed_kmh
                    ),
                    "route_distance_m": (
                        plan.route_distance_m
                    ),
                }
                for plan
                in target_plans
            ],
            "raw_video": str(
                video_path
            ),
            "ground_truth_metrics": str(
                metrics_path
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

        summary_lines = [
            "STEP 90 - TOWN10 DENSE FOG MULTI-VEHICLE CAPTURE",
            "=" * 84,
            f"Status: {status}",
            f"Map: {active_map}",
            (
                "Scene: straight; ego -> motorcycle -> car; "
                "oncoming bus -> motorcycle"
            ),
            (
                "Fog: "
                f"density={args.fog_density:.1f}, "
                f"distance={args.fog_distance:.1f}m, "
                f"falloff={args.fog_falloff:.2f}"
            ),
            (
                f"Video: {WIDTH}x{HEIGHT} @ {FPS:.1f} FPS"
            ),
            (
                f"Duration: {recorded / FPS:.1f}s"
            ),
            (
                f"First encounters: {first_encounter_frame}"
            ),
            (
                f"Encounter counts: {encounter_counts}"
            ),
            (
                "Forward+oncoming simultaneous frames: "
                f"{simultaneous_forward_oncoming_frames}"
            ),
            (
                "Two-or-more visible frames: "
                f"{two_or_more_visible_frames}"
            ),
            (
                "Three-or-more visible frames: "
                f"{three_or_more_visible_frames}"
            ),
            (
                "Maximum ego yaw change: "
                f"{maximum_ego_yaw_change:.1f}°"
            ),
            (
                "Unknown actors destroyed: "
                f"{unknown_actors_destroyed}"
            ),
            (
                "Failure reasons: "
                + (
                    ", ".join(
                        failure_reasons
                    )
                    if failure_reasons
                    else "none"
                )
            ),
            f"Raw video: {video_path}",
            f"Metrics: {metrics_path}",
            f"Report: {report_path}",
        ]

        summary_path.write_text(
            "\n".join(
                summary_lines
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 100)
        print(
            f"FINAL STATUS: {status}"
        )
        print(
            f"First encounters: {first_encounter_frame}"
        )
        print(
            f"Encounter counts: {encounter_counts}"
        )
        print(
            "Forward+oncoming simultaneous frames: "
            f"{simultaneous_forward_oncoming_frames}"
        )
        print(
            "Two-or-more visible frames: "
            f"{two_or_more_visible_frames}"
        )
        print(
            "Maximum ego yaw change: "
            f"{maximum_ego_yaw_change:.1f}°"
        )
        print(
            f"Raw video: {video_path}"
        )
        print(
            f"Report: {report_path}"
        )
        print("=" * 100)

    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

        if writer is not None:
            writer.release()

        # Only objects created/owned by this fog run.
        destroy_actor(
            camera
        )

        for plan in (
            target_plans
        ):
            destroy_actor(
                plan.actor
            )

        destroy_actor(
            ego
        )

        try:
            traffic_manager.set_synchronous_mode(
                False
            )
        except RuntimeError:
            pass

        try:
            world.apply_settings(
                original_settings
            )
        except RuntimeError:
            pass

        try:
            world.set_weather(
                original_weather
            )
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()

