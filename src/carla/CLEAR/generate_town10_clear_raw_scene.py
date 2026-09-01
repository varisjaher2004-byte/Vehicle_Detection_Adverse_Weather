from __future__ import annotations
import os

import argparse
import csv
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


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

DEFAULT_ROUTE_ROOT = (
    CARLA_ROOT / "outputs" / "town10_reference_route"
)

DEFAULT_OUTPUT_ROOT = (
    CARLA_ROOT / "outputs" / "town10_professional_raw_demo"
)

TARGET_BLUEPRINTS = (
    ("car", "vehicle.audi.tt"),
    ("truck", "vehicle.carlamotors.carlacola"),
    ("bus", "vehicle.mitsubishi.fusorosa"),
)


@dataclass
class TargetPlan:
    label: str
    blueprint_id: str
    route_distance_m: float
    route_waypoint: carla.Waypoint
    spawn_waypoint: carla.Waypoint
    actor: Optional[carla.Vehicle] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a clean Town10HD_Opt raw demonstration on the "
            "locked reference route: Tesla ego plus Audi, truck and bus."
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
            "Locked route JSON. When omitted, the latest valid route "
            "JSON under outputs/town10_reference_route is used."
        ),
    )
    parser.add_argument(
        "--route-rank",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help=(
            "Top-match route rank used as the driving start. "
            "Rank 2 is the longer cinematic runway on the same corridor."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--world-fps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--ego-speed-kmh", type=float, default=15.0)
    parser.add_argument(
        "--target-speed-kmh",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--weather",
        choices=(
            "reference_clear",
            "light_rain",
            "heavy_rain",
            "light_fog",
            "dense_fog",
        ),
        default="reference_clear",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


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
            "No locked Town10 route JSON was found under: "
            f"{DEFAULT_ROUTE_ROOT}"
        )

    return candidates[0]


def load_locked_transform(
    path: Path,
    route_rank: int,
) -> tuple[dict[str, object], dict[str, object], carla.Transform]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("status") != "REFERENCE_ROUTE_LOCKED":
        raise RuntimeError(
            f"Route JSON is not locked: {path}"
        )

    top_matches = payload.get("top_matches")

    if (
        not isinstance(top_matches, list)
        or len(top_matches) < route_rank
    ):
        raise RuntimeError(
            f"Route JSON does not contain route rank {route_rank}."
        )

    selected = top_matches[route_rank - 1]

    if not isinstance(selected, dict):
        raise RuntimeError(
            f"Route rank {route_rank} is invalid."
        )

    transform_data = selected.get("transform")

    if not isinstance(transform_data, dict):
        raise RuntimeError(
            "Route JSON has no selected transform."
        )

    location_data = transform_data["location"]
    rotation_data = transform_data["rotation"]

    transform = carla.Transform(
        carla.Location(
            x=float(location_data["x"]),
            y=float(location_data["y"]),
            z=float(location_data["z"]),
        ),
        carla.Rotation(
            pitch=float(rotation_data["pitch"]),
            yaw=float(rotation_data["yaw"]),
            roll=float(rotation_data["roll"]),
        ),
    )

    return payload, selected, transform


def weather_parameters(
    name: str,
) -> carla.WeatherParameters:
    presets = {
        "reference_clear": carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=0.0,
            precipitation_deposits=20.0,
            wetness=40.0,
            wind_intensity=5.0,
            fog_density=0.0,
            fog_distance=100.0,
            sun_altitude_angle=45.0,
        ),
        "light_rain": carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=30.0,
            precipitation_deposits=35.0,
            wetness=55.0,
            wind_intensity=25.0,
            fog_density=2.0,
            fog_distance=80.0,
            sun_altitude_angle=35.0,
        ),
        "heavy_rain": carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=75.0,
            precipitation_deposits=80.0,
            wetness=95.0,
            wind_intensity=60.0,
            fog_density=7.0,
            fog_distance=45.0,
            sun_altitude_angle=20.0,
        ),
        "light_fog": carla.WeatherParameters(
            cloudiness=65.0,
            precipitation=0.0,
            precipitation_deposits=10.0,
            wetness=25.0,
            wind_intensity=5.0,
            fog_density=18.0,
            fog_distance=28.0,
            fog_falloff=1.5,
            sun_altitude_angle=32.0,
        ),
        "dense_fog": carla.WeatherParameters(
            cloudiness=85.0,
            precipitation=0.0,
            precipitation_deposits=15.0,
            wetness=35.0,
            wind_intensity=3.0,
            fog_density=48.0,
            fog_distance=8.0,
            fog_falloff=1.1,
            sun_altitude_angle=22.0,
        ),
    }
    return presets[name]


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def angle_difference(first: float, second: float) -> float:
    return abs(normalize_angle(first - second))


def advance_straight(
    start: carla.Waypoint,
    distance_m: float,
    step_m: float = 2.0,
    maximum_corridor_yaw_change: float = 18.0,
) -> carla.Waypoint:
    current = start
    travelled = 0.0
    start_yaw = start.transform.rotation.yaw
    visited: list[carla.Location] = [
        start.transform.location
    ]

    while travelled < distance_m:
        options = [
            waypoint
            for waypoint in current.next(step_m)
            if waypoint.lane_type == carla.LaneType.Driving
        ]

        if not options:
            raise RuntimeError(
                f"Straight corridor ended after {travelled:.1f} m; "
                f"needed {distance_m:.1f} m."
            )

        next_waypoint = min(
            options,
            key=lambda waypoint: angle_difference(
                waypoint.transform.rotation.yaw,
                start_yaw,
            ),
        )

        corridor_yaw_change = angle_difference(
            next_waypoint.transform.rotation.yaw,
            start_yaw,
        )

        if corridor_yaw_change > maximum_corridor_yaw_change:
            raise RuntimeError(
                "Requested target would leave the locked straight "
                f"corridor at {travelled:.1f} m."
            )

        next_location = next_waypoint.transform.location

        if any(
            next_location.distance(old_location) < 1.0
            for old_location in visited[:-8]
        ):
            raise RuntimeError(
                "Route loop detected while building target plan."
            )

        travelled += current.transform.location.distance(
            next_location
        )
        visited.append(next_location)
        current = next_waypoint

    return current


def find_oncoming_waypoint(
    all_waypoints: list[carla.Waypoint],
    route_waypoint: carla.Waypoint,
) -> carla.Waypoint:
    route_location = route_waypoint.transform.location
    route_yaw = route_waypoint.transform.rotation.yaw

    candidates: list[
        tuple[float, carla.Waypoint]
    ] = []

    for waypoint in all_waypoints:
        if waypoint.lane_type != carla.LaneType.Driving:
            continue

        distance = waypoint.transform.location.distance(
            route_location
        )

        if distance > 10.0:
            continue

        yaw_difference = angle_difference(
            waypoint.transform.rotation.yaw,
            route_yaw,
        )

        if yaw_difference < 135.0:
            continue

        score = (
            distance
            + abs(180.0 - yaw_difference) * 0.04
            + (6.0 if waypoint.is_junction else 0.0)
        )
        candidates.append((score, waypoint))

    if not candidates:
        raise RuntimeError(
            "No safe oncoming driving lane was found near "
            f"route point at {route_location}."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def raised_transform(
    transform: carla.Transform,
    z_offset: float = 0.35,
) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + z_offset,
        ),
        transform.rotation,
    )


def image_to_bgr(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
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
            f"Could not create video: {path}"
        )

    return writer


def set_blueprint_attributes(
    blueprint: carla.ActorBlueprint,
    role_name: str,
) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute(
            "role_name",
            role_name,
        )

    if blueprint.has_attribute("color"):
        values = (
            blueprint.get_attribute("color")
            .recommended_values
        )

        if values:
            blueprint.set_attribute(
                "color",
                values[0],
            )


def configure_autopilot(
    traffic_manager: carla.TrafficManager,
    actor: carla.Vehicle,
    speed_kmh: float,
    tm_port: int,
) -> None:
    actor.set_autopilot(True, tm_port)
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
        2.5,
    )

    try:
        traffic_manager.set_desired_speed(
            actor,
            speed_kmh,
        )
    except (AttributeError, RuntimeError):
        percentage_difference = max(
            -100.0,
            min(100.0, 100.0 - speed_kmh / 0.3),
        )
        traffic_manager.vehicle_percentage_speed_difference(
            actor,
            percentage_difference,
        )


def speed_mps(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        velocity.x * velocity.x
        + velocity.y * velocity.y
        + velocity.z * velocity.z
    )


def camera_intrinsic(
    width: int,
    height: int,
    fov_degrees: float,
) -> np.ndarray:
    focal = width / (
        2.0 * math.tan(
            math.radians(fov_degrees) / 2.0
        )
    )

    matrix = np.identity(3)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def actor_projection(
    actor: carla.Actor,
    camera: carla.Sensor,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[bool, float, float, float]:
    world_point = actor.get_transform().location
    world_vector = np.array(
        [
            world_point.x,
            world_point.y,
            world_point.z
            + actor.bounding_box.location.z,
            1.0,
        ],
        dtype=np.float64,
    )

    world_to_camera = np.array(
        camera.get_transform().get_inverse_matrix(),
        dtype=np.float64,
    )
    sensor_point = world_to_camera @ world_vector

    carla_camera = np.array(
        [
            sensor_point[1],
            -sensor_point[2],
            sensor_point[0],
        ],
        dtype=np.float64,
    )

    depth = float(carla_camera[2])

    if depth <= 0.1:
        return False, float("nan"), float("nan"), depth

    image_point = intrinsic @ carla_camera
    u = float(image_point[0] / image_point[2])
    v = float(image_point[1] / image_point[2])

    visible = (
        -30.0 <= u <= width + 30.0
        and -30.0 <= v <= height + 30.0
        and depth <= 120.0
    )
    return visible, u, v, depth


def destroy_actor(actor: Optional[carla.Actor]) -> None:
    if actor is None:
        return

    try:
        actor.destroy()
    except RuntimeError:
        pass


def main() -> None:
    args = parse_args()

    if args.fps <= 0 or args.world_fps <= 0:
        raise ValueError(
            "--fps and --world-fps must be positive."
        )

    if args.fps > args.world_fps:
        raise ValueError(
            "--fps cannot exceed --world-fps."
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

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        args.output_root.resolve()
        / f"run_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    video_path = run_dir / "town10_professional_raw.mp4"
    metrics_path = run_dir / "frame_metrics.csv"
    summary_path = run_dir / "run_summary.txt"
    metadata_path = run_dir / "run_metadata.json"

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    world = client.get_world()
    active_map = world.get_map().name

    if not active_map.endswith(f"/{args.town}"):
        raise RuntimeError(
            f"Active map is {active_map}; expected {args.town}."
        )

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    traffic_manager = client.get_trafficmanager(
        args.tm_port
    )

    ego: Optional[carla.Vehicle] = None
    camera: Optional[carla.Sensor] = None
    target_plans: list[TargetPlan] = []
    writer: Optional[cv2.VideoWriter] = None

    frame_queue: queue.Queue[
        tuple[int, float, np.ndarray]
    ] = queue.Queue(maxsize=8)

    print("=" * 92)
    print("TOWN10 CLEAR RAW SCENE")
    print("=" * 92)
    print(f"Map:          {active_map}")
    print(f"Route JSON:   {route_json}")
    print(
        f"Driving route: rank {args.route_rank}, "
        f"spawn {driving_route['spawn_index']}, "
        f"road {driving_route['road_id']}"
    )
    print(
        f"Video:        {args.width}x{args.height} "
        f"@ {args.fps:.1f} FPS"
    )
    print(f"Duration:     {args.duration:.1f} seconds")
    print(f"Weather:      {args.weather}")
    print("Sequence:     car -> truck -> bus")
    print("=" * 92)

    try:
        # Remove leftovers from prior demo attempts.
        leftovers = list(
            world.get_actors().filter("vehicle.*")
        ) + list(
            world.get_actors().filter("sensor.camera.rgb")
        )

        for actor in leftovers:
            try:
                actor.destroy()
            except RuntimeError:
                pass

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / args.world_fps
        )
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(6803)
        world.set_weather(
            weather_parameters(args.weather)
        )

        carla_map = world.get_map()
        start_waypoint = carla_map.get_waypoint(
            locked_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if start_waypoint is None:
            raise RuntimeError(
                "Locked route could not be projected "
                "to a driving waypoint."
            )

        all_waypoints = carla_map.generate_waypoints(
            2.0
        )

        target_distances = (34.0, 64.0, 88.0)

        for (
            (label, blueprint_id),
            distance_m,
        ) in zip(
            TARGET_BLUEPRINTS,
            target_distances,
        ):
            route_waypoint = advance_straight(
                start_waypoint,
                distance_m,
            )
            oncoming_waypoint = find_oncoming_waypoint(
                all_waypoints,
                route_waypoint,
            )
            target_plans.append(
                TargetPlan(
                    label=label,
                    blueprint_id=blueprint_id,
                    route_distance_m=distance_m,
                    route_waypoint=route_waypoint,
                    spawn_waypoint=oncoming_waypoint,
                )
            )

        library = world.get_blueprint_library()

        ego_bp = library.find(
            "vehicle.tesla.model3"
        )
        set_blueprint_attributes(
            ego_bp,
            "town10_demo_ego",
        )
        ego = world.try_spawn_actor(
            ego_bp,
            raised_transform(
                start_waypoint.transform,
                z_offset=0.45,
            ),
        )

        if ego is None:
            raise RuntimeError(
                "Tesla could not spawn on locked route."
            )

        for plan in target_plans:
            blueprint = library.find(
                plan.blueprint_id
            )
            set_blueprint_attributes(
                blueprint,
                f"town10_demo_{plan.label}",
            )
            actor = world.try_spawn_actor(
                blueprint,
                raised_transform(
                    plan.spawn_waypoint.transform,
                    z_offset=0.45,
                ),
            )

            if actor is None:
                raise RuntimeError(
                    f"Could not spawn {plan.label} "
                    f"({plan.blueprint_id})."
                )

            plan.actor = actor

        camera_bp = library.find(
            "sensor.camera.rgb"
        )
        camera_bp.set_attribute(
            "image_size_x",
            str(args.width),
        )
        camera_bp.set_attribute(
            "image_size_y",
            str(args.height),
        )
        camera_bp.set_attribute("fov", "90")

        # Epic quality is retained, but strong motion blur is removed
        # because it looks excessive at a 10 FPS research-demo output.
        if camera_bp.has_attribute("motion_blur_intensity"):
            camera_bp.set_attribute(
                "motion_blur_intensity",
                "0.0",
            )

        if camera_bp.has_attribute("motion_blur_max_distortion"):
            camera_bp.set_attribute(
                "motion_blur_max_distortion",
                "0.0",
            )

        if camera_bp.has_attribute("lens_flare_intensity"):
            camera_bp.set_attribute(
                "lens_flare_intensity",
                "0.0",
            )

        camera_bp.set_attribute(
            "sensor_tick",
            f"{1.0 / args.fps:.6f}",
        )

        def callback(image: carla.Image) -> None:
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
            camera_bp,
            carla.Transform(
                carla.Location(x=1.5, z=2.4)
            ),
            attach_to=ego,
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )
        camera.listen(callback)

        # Static warm-up prevents visible spawning and lets assets settle.
        for _ in range(
            max(20, int(args.world_fps * 2.0))
        ):
            world.tick()

        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        configure_autopilot(
            traffic_manager,
            ego,
            args.ego_speed_kmh,
            args.tm_port,
        )

        for plan in target_plans:
            assert plan.actor is not None
            configure_autopilot(
                traffic_manager,
                plan.actor,
                args.target_speed_kmh,
                args.tm_port,
            )

        # Short hidden acceleration period. The first recorded frame now
        # begins with smooth road motion instead of a stationary launch.
        acceleration_ticks = 0
        maximum_acceleration_ticks = int(
            args.world_fps * 1.5
        )

        while (
            speed_mps(ego) < 3.2
            and acceleration_ticks < maximum_acceleration_ticks
        ):
            world.tick()
            acceleration_ticks += 1

        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        writer = create_writer(
            video_path,
            args.fps,
            args.width,
            args.height,
        )

        required_frames = int(
            round(args.duration * args.fps)
        )
        intrinsic = camera_intrinsic(
            args.width,
            args.height,
            90.0,
        )

        first_visible_frame: dict[
            str, Optional[int]
        ] = {
            plan.label: None
            for plan in target_plans
        }
        visible_counts: dict[str, int] = {
            plan.label: 0
            for plan in target_plans
        }
        first_encounter_frame: dict[
            str, Optional[int]
        ] = {
            plan.label: None
            for plan in target_plans
        }
        encounter_counts: dict[str, int] = {
            plan.label: 0
            for plan in target_plans
        }

        metrics_rows: list[dict[str, object]] = []
        recorded = 0
        start_simulation_time: Optional[float] = None

        while recorded < required_frames:
            world.tick()

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

            if not available:
                continue

            for (
                carla_frame,
                timestamp_seconds,
                frame,
            ) in available:
                if recorded >= required_frames:
                    break

                if start_simulation_time is None:
                    start_simulation_time = (
                        timestamp_seconds
                    )

                recorded += 1
                writer.write(frame)

                row: dict[str, object] = {
                    "video_frame": recorded,
                    "carla_frame": carla_frame,
                    "time_seconds": (
                        timestamp_seconds
                        - start_simulation_time
                    ),
                    "ego_speed_mps": speed_mps(ego),
                }

                for plan in target_plans:
                    assert plan.actor is not None
                    visible, u, v, depth = (
                        actor_projection(
                            plan.actor,
                            camera,
                            intrinsic,
                            args.width,
                            args.height,
                        )
                    )

                    row[
                        f"{plan.label}_visible"
                    ] = int(visible)
                    row[f"{plan.label}_u"] = u
                    row[f"{plan.label}_v"] = v
                    row[f"{plan.label}_depth_m"] = depth
                    row[
                        f"{plan.label}_speed_mps"
                    ] = speed_mps(plan.actor)

                    if visible:
                        visible_counts[plan.label] += 1

                        if (
                            first_visible_frame[
                                plan.label
                            ]
                            is None
                        ):
                            first_visible_frame[
                                plan.label
                            ] = recorded

                    # Sequence validation uses a meaningful close
                    # encounter rather than counting tiny or occluded
                    # projections over 100 metres away.
                    effective_encounter = (
                        visible
                        and 3.0 <= depth <= 35.0
                        and 0.0 <= u <= args.width
                        and 0.0 <= v <= args.height
                    )
                    row[
                        f"{plan.label}_encounter"
                    ] = int(effective_encounter)

                    if effective_encounter:
                        encounter_counts[plan.label] += 1

                        if (
                            first_encounter_frame[
                                plan.label
                            ]
                            is None
                        ):
                            first_encounter_frame[
                                plan.label
                            ] = recorded

                metrics_rows.append(row)

                if recorded in {
                    1,
                    max(1, required_frames // 3),
                    max(1, 2 * required_frames // 3),
                    required_frames,
                }:
                    cv2.imwrite(
                        str(
                            run_dir
                            / f"frame_{recorded:03d}.png"
                        ),
                        frame,
                    )

                if recorded % max(
                    1,
                    int(args.fps * 3),
                ) == 0:
                    print(
                        f"Recorded {recorded}/"
                        f"{required_frames} frames"
                    )

        fieldnames = list(metrics_rows[0].keys())

        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer_csv = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer_csv.writeheader()
            writer_csv.writerows(metrics_rows)

        ordered_first_frames = [
            first_encounter_frame["car"],
            first_encounter_frame["truck"],
            first_encounter_frame["bus"],
        ]

        sequence_valid = (
            all(
                value is not None
                for value in ordered_first_frames
            )
            and ordered_first_frames[0]
            < ordered_first_frames[1]
            < ordered_first_frames[2]
        )

        sufficient_visibility = all(
            encounter_counts[label]
            >= max(3, int(args.fps * 0.5))
            for label in ("car", "truck", "bus")
        )

        status = (
            "PASS"
            if sequence_valid
            and sufficient_visibility
            else "REVIEW_REQUIRED"
        )

        metadata = {
            "status": status,
            "map": active_map,
            "route_json": str(route_json),
            "reference_best_spawn_index": (
                route_payload["selected"][
                    "spawn_index"
                ]
            ),
            "driving_route_rank": args.route_rank,
            "driving_spawn_index": (
                driving_route["spawn_index"]
            ),
            "driving_road_id": (
                driving_route["road_id"]
            ),
            "capture_method": (
                "CARLA sensor.camera.rgb attached rigidly "
                "to Tesla Model 3"
            ),
            "resolution": [
                args.width,
                args.height,
            ],
            "fps": args.fps,
            "world_fps": args.world_fps,
            "duration_seconds": (
                required_frames / args.fps
            ),
            "frame_count": required_frames,
            "weather": args.weather,
            "ego_speed_kmh": args.ego_speed_kmh,
            "target_speed_kmh": (
                args.target_speed_kmh
            ),
            "first_visible_frame": (
                first_visible_frame
            ),
            "visible_frame_counts": visible_counts,
            "first_encounter_frame": (
                first_encounter_frame
            ),
            "encounter_frame_counts": (
                encounter_counts
            ),
            "sequence_valid": sequence_valid,
            "sufficient_visibility": (
                sufficient_visibility
            ),
            "targets": [
                {
                    "label": plan.label,
                    "blueprint": (
                        plan.blueprint_id
                    ),
                    "route_distance_m": (
                        plan.route_distance_m
                    ),
                    "spawn_transform": {
                        "x": (
                            plan.spawn_waypoint
                            .transform.location.x
                        ),
                        "y": (
                            plan.spawn_waypoint
                            .transform.location.y
                        ),
                        "z": (
                            plan.spawn_waypoint
                            .transform.location.z
                        ),
                        "yaw": (
                            plan.spawn_waypoint
                            .transform.rotation.yaw
                        ),
                    },
                }
                for plan in target_plans
            ],
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        summary_lines = [
            "TOWN10 CLEAR RAW SCENE",
            "=" * 76,
            f"Status: {status}",
            f"Map: {active_map}",
            (
                "Capture: genuine CARLA "
                "sensor.camera.rgb"
            ),
            (
                f"Resolution: {args.width}x"
                f"{args.height}"
            ),
            f"FPS: {args.fps}",
            (
                f"Duration: "
                f"{required_frames / args.fps:.2f} s"
            ),
            f"Frames: {required_frames}",
            f"Weather: {args.weather}",
            (
                "First projected frames: "
                f"car={first_visible_frame['car']}, "
                f"truck={first_visible_frame['truck']}, "
                f"bus={first_visible_frame['bus']}"
            ),
            (
                "First encounter frames: "
                f"car={first_encounter_frame['car']}, "
                f"truck={first_encounter_frame['truck']}, "
                f"bus={first_encounter_frame['bus']}"
            ),
            (
                "Encounter counts: "
                f"car={encounter_counts['car']}, "
                f"truck={encounter_counts['truck']}, "
                f"bus={encounter_counts['bus']}"
            ),
            (
                f"Sequence valid: {sequence_valid}"
            ),
            (
                "Sufficient visibility: "
                f"{sufficient_visibility}"
            ),
            f"Raw video: {video_path}",
            f"Metrics: {metrics_path}",
        ]

        summary_path.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("=" * 92)
        print(f"FINAL STATUS: {status}")
        print(
            "First encounter frames: "
            f"{first_encounter_frame}"
        )
        print(
            f"Encounter counts: {encounter_counts}"
        )
        print(
            f"Raw video: {video_path}"
        )
        print(
            f"Metrics:   {metrics_path}"
        )
        print("=" * 92)

    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

        if writer is not None:
            writer.release()

        destroy_actor(camera)

        for plan in target_plans:
            destroy_actor(plan.actor)

        destroy_actor(ego)

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


if __name__ == "__main__":
    main()
