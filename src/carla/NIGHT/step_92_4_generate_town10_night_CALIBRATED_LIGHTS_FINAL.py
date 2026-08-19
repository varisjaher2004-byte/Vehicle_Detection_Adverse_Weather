from __future__ import annotations
import os

import argparse
import csv
import gc
import json
import math
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import carla
import cv2
import numpy as np

# Reuse the exact helper functions from the accepted Fog capture script.
import sys
FOG_DIR = Path(__file__).resolve().parents[1] / "FOG"
if str(FOG_DIR) not in sys.path:
    sys.path.insert(0, str(FOG_DIR))
import step_90_generate_town10_fog_multivehicle as fogbase


# ======================================================================================
# STEP 92.4 FINAL - TOWN10 NIGHT WITH CALIBRATED ACTOR LIGHTS
#
# WHY THIS EXISTS
# ---------------
# The accepted CLEAR / RAIN / FOG Town10 captures all used a positive sun angle.
# Native Town10 negative-sun night repeatedly crashes this PC's UE4/D3D renderer.
#
# Therefore:
#   1) CARLA runs in the proven positive-sun rendering regime.
#   2) Scene / route / vehicles / pedestrians / RGB frames are genuine CARLA.
#   3) The night appearance is created CPU-side from each genuine RGB frame.
#
# This avoids the UE4 D3D device-loss path while preserving the exact Town10 scene.
#
# Scene:
#   roadside: PERSON + PERSON
#   same direction: EGO -> MOTORCYCLE -> CAR
#   opposite: BUS -> TRUCK
#   no signal, no crossing, no turn
#
# This is a qualitative presentation capture. The report explicitly records that
# the night visual grading is post-processing, not native negative-sun rendering.
# ======================================================================================


CARLA_ROOT = Path(os.environ.get("CARLA_RESEARCH_ROOT", ".")).resolve()

OUTPUT_ROOT = (
    CARLA_ROOT
    / "outputs"
    / "town10_night_actor_lights_final"
)

ROLE_PREFIX = "night924calibrated_"

SCRIPT_VERSION = "92.4"
NIGHT_GRADE_VERSION = "cpu_day_for_night_v5_calibrated_actor_lights"
SOURCE_SUN_ALTITUDE_DEG = 38.0
DEFAULT_RANDOM_SEED = 9007

WIDTH = 640
HEIGHT = 360
FPS = 10.0
WORLD_FPS = 20.0
FOV = 90.0

CAMERA_X = 1.50
CAMERA_Y = 0.00
CAMERA_Z = 1.70
CAMERA_PITCH = 0.0

DURATION_SECONDS = 14.0

# A projected box can be visible to CARLA but still be too small, too distant,
# or edge-clipped for fair object-detection evaluation.  Keep the raw projected
# visibility in the CSV and record a separate label-quality decision.
MIN_LABEL_DEPTH_M = 2.0
MAX_LABEL_DEPTH_M = 48.0
MIN_LABEL_WIDTH_PX = 6.0
MIN_LABEL_HEIGHT_PX = 8.0
MIN_LABEL_AREA_PX2 = 96.0
FRAME_EDGE_MARGIN_PX = 1.0

EGO_SPEED_KMH = 16.0
FORWARD_MOTORCYCLE_SPEED_KMH = 18.0
FORWARD_CAR_SPEED_KMH = 18.0
ONCOMING_BUS_SPEED_KMH = 18.0
ONCOMING_TRUCK_SPEED_KMH = 18.0

FORWARD_MOTORCYCLE_DISTANCE_M = 18.0
FORWARD_CAR_DISTANCE_M = 36.0
ONCOMING_BUS_ROUTE_DISTANCE_M = 56.0
ONCOMING_TRUCK_ROUTE_DISTANCE_M = 78.0

PERSON_1_ROUTE_DISTANCE_M = 23.0
PERSON_2_ROUTE_DISTANCE_M = 42.0

VEHICLE_KEYS = (
    "forward_motorcycle",
    "forward_car",
    "oncoming_bus",
    "oncoming_truck",
)

PERSON_KEYS = (
    "person_1",
    "person_2",
)

ALL_TARGET_KEYS = VEHICLE_KEYS + PERSON_KEYS

TARGET_DIRECTION = {
    "forward_motorcycle": "same_direction",
    "forward_car": "same_direction",
    "oncoming_bus": "oncoming",
    "oncoming_truck": "oncoming",
}

TARGET_SPEED_KMH = {
    "forward_motorcycle": FORWARD_MOTORCYCLE_SPEED_KMH,
    "forward_car": FORWARD_CAR_SPEED_KMH,
    "oncoming_bus": ONCOMING_BUS_SPEED_KMH,
    "oncoming_truck": ONCOMING_TRUCK_SPEED_KMH,
}

TARGET_BLUEPRINTS = {
    # Motorcycle proven in accepted fog.
    "forward_motorcycle": "vehicle.kawasaki.ninja",
    # Car, truck and bus proven in accepted clear / fog Town10 demos.
    "forward_car": "vehicle.audi.tt",
    "oncoming_bus": "vehicle.mitsubishi.fusorosa",
    "oncoming_truck": "vehicle.carlamotors.carlacola",
}

# Blueprint-specific lamp calibration. CARLA bounding boxes describe the full
# body/cargo envelope, which is wider than the visible front lamp spacing on
# the Mitsubishi bus and especially the Carla-Cola truck cab.
LAMP_LATERAL_FRACTION = {
    "forward_motorcycle": 0.0,
    "forward_car": 0.56,
    "oncoming_bus": 0.38,
    "oncoming_truck": 0.35,
}

# Local-y centre correction, expressed as a fraction of bounding-box half
# width. Negative values move the projected pair toward the visible cab centre
# for the locked Town10 approach direction.
LAMP_CENTER_Y_BIAS = {
    "forward_motorcycle": 0.0,
    "forward_car": 0.0,
    "oncoming_bus": -0.07,
    "oncoming_truck": -0.24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final Town10 night-looking scene using the "
            "accepted Fog capture engine and CPU-only night grading."
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
        default=DURATION_SECONDS,
    )
    parser.add_argument(
        "--night-strength",
        type=float,
        default=1.0,
        help="Day-for-night strength in the inclusive range 0.60-1.35.",
    )
    parser.add_argument(
        "--sensor-noise",
        type=float,
        default=0.35,
        help=(
            "Deterministic low-light sensor-noise strength in the range "
            "0.0-1.5. Use 0 to disable it."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Reproducible Traffic Manager and CPU-grading seed.",
    )
    parser.add_argument(
        "--disable-light-cues",
        dest="actor_anchored_light_cues",
        action="store_false",
        default=True,
        help=(
            "Disable the subtle 3D actor-anchored lamp enhancement. Genuine "
            "CARLA Position + LowBeam lights remain enabled."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args()


def stable_source_weather() -> carla.WeatherParameters:
    """
    Deliberately stays in the same positive-sun renderer regime as the
    accepted Fog scene (+38 degrees), with fog removed.
    """
    return carla.WeatherParameters(
        cloudiness=72.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=3.0,
        sun_azimuth_angle=35.0,
        sun_altitude_angle=SOURCE_SUN_ALTITUDE_DEG,
        fog_density=0.0,
        fog_distance=1000.0,
        fog_falloff=0.2,
        wetness=5.0,
        scattering_intensity=0.90,
        mie_scattering_scale=0.025,
        rayleigh_scattering_scale=0.0331,
    )


def night_grade_frame(
    frame: np.ndarray,
    strength: float,
    frame_index: int,
    sensor_noise: float,
    seed: int,
) -> np.ndarray:
    """
    Create a deterministic detector-friendly day-for-night image.

    Version 2 deliberately avoids strong full-frame CLAHE because it made the
    trees and buildings look daylight-bright.  It remaps luminance first,
    applies a restrained cool shadow balance, uses vertical falloff plus a
    vignette, and adds low-amplitude seeded sensor noise.  Geometry is never
    changed, so projected ground truth remains aligned.
    """
    s = fogbase.clamp(
        float(strength),
        0.60,
        1.35,
    )
    noise_amount = fogbase.clamp(
        float(sensor_noise),
        0.0,
        1.5,
    )

    if (
        frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError(
            "night_grade_frame expects a BGR HxWx3 image."
        )

    # A small local-contrast blend preserves road texture without giving every
    # leaf the bright, crunchy appearance produced by full-strength CLAHE.
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB,
    )
    l_chan, a_chan, b_chan = cv2.split(
        lab
    )
    clahe = cv2.createCLAHE(
        clipLimit=1.18,
        tileGridSize=(8, 8),
    )
    l_chan = clahe.apply(
        l_chan
    )
    locally_enhanced = cv2.cvtColor(
        cv2.merge(
            (
                l_chan,
                a_chan,
                b_chan,
            )
        ),
        cv2.COLOR_LAB2BGR,
    )

    image = cv2.addWeighted(
        frame,
        0.86,
        locally_enhanced,
        0.14,
        0.0,
    ).astype(
        np.float32
    ) / 255.0

    source_luma = (
        image[:, :, 0] * 0.114
        + image[:, :, 1] * 0.587
        + image[:, :, 2] * 0.299
    )

    # Remap luminance rather than multiplying all colours equally.  Highlights
    # stay readable while midtones and daylight-lit foliage fall away.
    exposure = (
        0.50
        - 0.08
        * (
            s
            - 1.0
        )
    )
    exposure = fogbase.clamp(
        exposure,
        0.44,
        0.54,
    )

    gamma = (
        1.22
        + 0.14
        * (
            s
            - 1.0
        )
    )
    gamma = fogbase.clamp(
        gamma,
        1.16,
        1.30,
    )

    night_luma = exposure * np.power(
        np.clip(
            source_luma,
            0.0,
            1.0,
        ),
        gamma,
    )

    luminance_gain = night_luma / np.maximum(
        source_luma,
        1.0e-4,
    )
    image *= luminance_gain[:, :, None]

    # Cool shadows without turning the whole frame cyan.
    shadow_weight = np.clip(
        1.0
        - night_luma / 0.42,
        0.0,
        1.0,
    )
    image[:, :, 0] *= (
        1.08
        + 0.07 * shadow_weight
    )
    image[:, :, 1] *= (
        0.96
        + 0.01 * shadow_weight
    )
    image[:, :, 2] *= (
        0.82
        - 0.05 * shadow_weight
    )

    # Low-light scenes carry less colour in their shadows.
    neutral = night_luma[:, :, None]
    image = (
        image * 0.90
        + neutral * 0.10
    )

    height, width = (
        image.shape[:2]
    )

    y_normalized = np.linspace(
        0.0,
        1.0,
        height,
        dtype=np.float32,
    )
    vertical_falloff = (
        0.84
        + 0.16
        * np.exp(
            -(
                (
                    y_normalized
                    - 0.54
                )
                / 0.43
            )
            ** 2
        )
    )
    image *= vertical_falloff[:, None, None]

    # Mild optical vignette.  It is intentionally weaker than the old grade so
    # that edge targets are not hidden by post-processing.
    yy, xx = np.mgrid[
        0:height,
        0:width,
    ].astype(
        np.float32
    )

    cx = (
        width
        - 1
    ) * 0.5
    cy = (
        height
        - 1
    ) * 0.52

    nx = (
        xx
        - cx
    ) / max(
        1.0,
        width
        * 0.62,
    )

    ny = (
        yy
        - cy
    ) / max(
        1.0,
        height
        * 0.72,
    )

    radius = np.sqrt(
        nx * nx
        + ny * ny
    )

    vignette = (
        1.0
        - np.clip(
            radius,
            0.0,
            1.0,
        )
        * 0.13
    )

    image *= (
        vignette[
            :,
            :,
            None,
        ]
    )

    # A very small blue-black floor avoids crushed compression blocks without
    # creating the grey shadows seen in the earlier version.
    image += np.array(
        [0.0045, 0.0030, 0.0020],
        dtype=np.float32,
    )[None, None, :]

    if noise_amount > 0.0:
        rng = np.random.default_rng(
            int(seed)
            + int(frame_index) * 104729
        )
        darkness = np.clip(
            1.0 - night_luma,
            0.0,
            1.0,
        )
        sigma = (
            0.0015
            + 0.0035 * darkness
        ) * noise_amount
        luminance_noise = rng.normal(
            0.0,
            1.0,
            size=(height, width),
        ).astype(
            np.float32
        ) * sigma
        chroma_noise = rng.normal(
            0.0,
            0.0009 * noise_amount,
            size=image.shape,
        ).astype(
            np.float32
        )
        image += luminance_noise[:, :, None]
        image += chroma_noise

    return np.clip(
        image
        * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )


def frame_visual_metrics(
    frame: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Return compact reproducible image-quality diagnostics for one frame."""
    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )
    p01, p50, p99 = np.percentile(
        grey,
        (1.0, 50.0, 99.0),
    )
    return {
        f"{prefix}_luma_mean": float(
            np.mean(grey)
        ),
        f"{prefix}_luma_p01": float(p01),
        f"{prefix}_luma_p50": float(p50),
        f"{prefix}_luma_p99": float(p99),
        f"{prefix}_shadow_fraction": float(
            np.mean(grey < 16)
        ),
        f"{prefix}_highlight_fraction": float(
            np.mean(grey > 235)
        ),
        f"{prefix}_sharpness_laplacian": float(
            cv2.Laplacian(
                grey,
                cv2.CV_32F,
            ).var()
        ),
    }


def _valid_box(
    row: dict[str, Any],
    key: str,
) -> Optional[
    tuple[
        float,
        float,
        float,
        float,
    ]
]:
    try:
        values = (
            float(
                row[
                    f"{key}_x1"
                ]
            ),
            float(
                row[
                    f"{key}_y1"
                ]
            ),
            float(
                row[
                    f"{key}_x2"
                ]
            ),
            float(
                row[
                    f"{key}_y2"
                ]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        math.isfinite(
            value
        )
        for value
        in values
    ):
        return None

    x1, y1, x2, y2 = (
        values
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return values


def enable_genuine_vehicle_lights(
    actors: dict[str, carla.Vehicle],
) -> dict[str, str]:
    """
    Enable CARLA-rendered position and dipped-beam lights.

    These lights are part of each supported vehicle asset and therefore follow
    its real lamp geometry, perspective, occlusion and motion.  Unsupported
    assets are left usable; their status is recorded instead of replacing the
    lamps with fixed image-space circles.
    """
    status: dict[str, str] = {}
    try:
        requested_bits = (
            int(carla.VehicleLightState.NONE)
            | int(carla.VehicleLightState.Position)
            | int(carla.VehicleLightState.LowBeam)
        )
        # Older Boost.Python CARLA builds return a plain int after bitwise OR.
        # Explicitly reconstruct the enum required by Vehicle.set_light_state.
        requested_state = carla.VehicleLightState(requested_bits)
    except Exception as exc:
        message = f"native light enum unavailable: {type(exc).__name__}: {exc}"
        return {key: message for key in actors}

    for key, actor in actors.items():
        try:
            actor.set_light_state(requested_state)
            status[key] = str(actor.get_light_state())
        except Exception as exc:
            # Native lamps are a best-effort enhancement. The rigid 3D
            # actor-anchored cues remain available, so do not abort capture.
            status[key] = (
                f"unavailable: {type(exc).__name__}: {exc}"
            )
    return status


def _light_box(
    row: dict[str, Any],
    key: str,
) -> Optional[
    tuple[float, float, float, float]
]:
    """Reject distant/tiny projected boxes before adding presentation cues."""
    if not bool(
        row.get(
            f"{key}_visible",
            0,
        )
    ):
        return None

    box = _valid_box(
        row,
        key,
    )
    if box is None:
        return None

    try:
        depth = float(
            row[f"{key}_depth_m"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    x1, y1, x2, y2 = box
    area = max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )

    if (
        not math.isfinite(depth)
        or depth > 52.0
        or area < 72.0
    ):
        return None

    return box


def _legacy_add_bbox_light_cues(
    frame: np.ndarray,
    row: dict[str, Any],
) -> np.ndarray:
    """
    Retained only for audit comparison with Step 92.1. This bounding-box based
    method is never called because it can drift across a vehicle in side view.
    """
    output = frame.copy()
    bloom = np.zeros_like(
        output
    )
    cores: list[
        tuple[
            int,
            int,
            int,
            tuple[int, int, int],
        ]
    ] = []

    # Oncoming = white headlights.
    for key in (
        "oncoming_bus",
        "oncoming_truck",
    ):
        box = _light_box(
            row,
            key,
        )

        if box is None:
            continue

        x1, y1, x2, y2 = (
            box
        )

        width = (
            x2
            - x1
        )
        height = (
            y2
            - y1
        )

        y = int(
            round(
                y1
                + 0.68
                * height
            )
        )

        for fraction in (
            0.33,
            0.67,
        ):
            x = int(
                round(
                    x1
                    + fraction
                    * width
                )
            )

            radius = max(
                2,
                min(
                    7,
                    int(
                        round(
                            width
                            * 0.035
                        )
                    ),
                ),
            )

            cv2.circle(
                bloom,
                (
                    x,
                    y,
                ),
                radius * 4,
                (
                    70,
                    105,
                    145,
                ),
                -1,
                cv2.LINE_AA,
            )
            cores.append(
                (
                    x,
                    y,
                    radius,
                    (225, 242, 255),
                )
            )

    # Same direction = red tail-light cues.
    for key in (
        "forward_motorcycle",
        "forward_car",
    ):
        box = _light_box(
            row,
            key,
        )

        if box is None:
            continue

        x1, y1, x2, y2 = (
            box
        )

        width = (
            x2
            - x1
        )
        height = (
            y2
            - y1
        )

        y = int(
            round(
                y1
                + 0.72
                * height
            )
        )

        fractions = (
            (0.50,)
            if key
            == "forward_motorcycle"
            else (
                0.34,
                0.66,
            )
        )

        for fraction in fractions:
            x = int(
                round(
                    x1
                    + fraction
                    * width
                )
            )

            radius = max(
                2,
                min(
                    5,
                    int(
                        round(
                            width
                            * 0.03
                        )
                    ),
                ),
            )

            cv2.circle(
                bloom,
                (
                    x,
                    y,
                ),
                radius * 3,
                (
                    12,
                    18,
                    90,
                ),
                -1,
                cv2.LINE_AA,
            )
            cores.append(
                (
                    x,
                    y,
                    radius,
                    (35, 55, 230),
                )
            )

    if cores:
        bloom = cv2.GaussianBlur(
            bloom,
            (0, 0),
            sigmaX=4.0,
            sigmaY=4.0,
        )
        cv2.addWeighted(
            output,
            1.0,
            bloom,
            0.82,
            0.0,
            output,
        )

    for x, y, radius, colour in cores:
        cv2.circle(
            output,
            (x, y),
            radius,
            colour,
            -1,
            cv2.LINE_AA,
        )

    return output


def _project_actor_local_point(
    actor: carla.Vehicle,
    camera: carla.Sensor,
    intrinsic: np.ndarray,
    local_point: tuple[float, float, float],
) -> Optional[tuple[float, float, float]]:
    """Project a rigid actor-local 3D point into the RGB camera image."""
    local = np.asarray(
        [local_point[0], local_point[1], local_point[2], 1.0],
        dtype=float,
    )
    actor_to_world = np.asarray(
        actor.get_transform().get_matrix(),
        dtype=float,
    )
    world_to_camera = np.asarray(
        camera.get_transform().get_inverse_matrix(),
        dtype=float,
    )
    sensor_point = world_to_camera @ (actor_to_world @ local)

    # CARLA sensor coordinates (x forward, y right, z up) to conventional
    # camera coordinates (x right, y down, z forward).
    camera_point = np.asarray(
        [sensor_point[1], -sensor_point[2], sensor_point[0]],
        dtype=float,
    )
    depth = float(camera_point[2])
    if not math.isfinite(depth) or depth <= 0.10:
        return None

    pixel = intrinsic @ camera_point
    u = float(pixel[0] / pixel[2])
    v = float(pixel[1] / pixel[2])
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return u, v, depth


def _vehicle_front_facing_score(
    actor: carla.Vehicle,
    camera: carla.Sensor,
) -> float:
    """Return +1 front-facing, -1 rear-facing, and 0 for a side view."""
    actor_transform = actor.get_transform()
    camera_location = camera.get_transform().location
    actor_location = actor_transform.location
    to_camera = np.asarray(
        [
            camera_location.x - actor_location.x,
            camera_location.y - actor_location.y,
            camera_location.z - actor_location.z,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(to_camera))
    if norm <= 1.0e-6:
        return 0.0
    forward = actor_transform.get_forward_vector()
    forward_vector = np.asarray(
        [forward.x, forward.y, forward.z],
        dtype=float,
    )
    return float(np.dot(forward_vector, to_camera / norm))


def _inside_actor_projection(
    row: dict[str, Any],
    key: str,
    u: float,
    v: float,
) -> bool:
    box = _valid_box(row, key)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    margin = 4.0
    return bool(
        x1 - margin <= u <= x2 + margin
        and y1 - margin <= v <= y2 + margin
    )


def add_actor_anchored_light_cues(
    frame: np.ndarray,
    row: dict[str, Any],
    vehicles: dict[str, carla.Vehicle],
    camera: carla.Sensor,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """
    Add restrained lamp highlights at rigid 3D points on each vehicle.

    Unlike Step 92.1, lamp locations are transformed with the actor before
    projection. Headlights are hidden in side/rear views and tail lights are
    hidden in side/front views, preventing floating lights after a vehicle
    passes the camera.
    """
    output = frame.copy()
    bloom = np.zeros_like(output)
    lamps: list[tuple[int, int, int, tuple[int, int, int], bool]] = []

    for key in VEHICLE_KEYS:
        if not bool(row.get(f"{key}_visible", 0)):
            continue
        actor = vehicles[key]
        score = _vehicle_front_facing_score(actor, camera)
        is_headlight = key in {"oncoming_bus", "oncoming_truck"}
        if is_headlight and score < 0.28:
            continue
        if not is_headlight and score > -0.28:
            continue

        box = actor.bounding_box
        centre = box.location
        extent = box.extent
        anchor_x = (
            centre.x + 1.01 * extent.x
            if is_headlight
            else centre.x - 1.01 * extent.x
        )
        anchor_z = centre.z - 0.34 * extent.z
        lateral_fraction = LAMP_LATERAL_FRACTION[key]
        anchor_center_y = (
            centre.y
            + LAMP_CENTER_Y_BIAS[key] * extent.y
        )
        lateral_offsets = (
            (0.0,)
            if key == "forward_motorcycle"
            else (-lateral_fraction * extent.y, lateral_fraction * extent.y)
        )

        projected: list[tuple[float, float, float]] = []
        for offset_y in lateral_offsets:
            point = _project_actor_local_point(
                actor,
                camera,
                intrinsic,
                (anchor_x, anchor_center_y + offset_y, anchor_z),
            )
            if point is None:
                continue
            u, v, depth = point
            if depth > 55.0 or not _inside_actor_projection(row, key, u, v):
                continue
            if not (-3.0 <= u < WIDTH + 3.0 and -3.0 <= v < HEIGHT + 3.0):
                continue
            projected.append(point)

        if not projected:
            continue

        if len(projected) >= 2:
            separation = math.hypot(
                projected[0][0] - projected[1][0],
                projected[0][1] - projected[1][1],
            )
            radius = int(np.clip(round(separation * 0.055), 1, 3))
        else:
            radius = 1

        core_colour = (220, 238, 255) if is_headlight else (35, 55, 225)
        glow_colour = (45, 62, 78) if is_headlight else (8, 12, 58)
        for u, v, _ in projected:
            x = int(round(u))
            y = int(round(v))
            cv2.ellipse(
                bloom,
                (x, y),
                (radius * 3, max(2, radius * 2)),
                0.0,
                0.0,
                360.0,
                glow_colour,
                -1,
                cv2.LINE_AA,
            )
            lamps.append((x, y, radius, core_colour, is_headlight))

    if lamps:
        bloom = cv2.GaussianBlur(
            bloom,
            (0, 0),
            sigmaX=2.2,
            sigmaY=2.2,
        )
        cv2.addWeighted(output, 1.0, bloom, 0.58, 0.0, output)

    for x, y, radius, colour, is_headlight in lamps:
        axes = (max(1, radius + 1), max(1, radius))
        cv2.ellipse(
            output,
            (x, y),
            axes,
            0.0,
            0.0,
            360.0,
            colour,
            -1,
            cv2.LINE_AA,
        )

    return output


def annotate_review_frame(
    frame: np.ndarray,
    row: dict[str, Any],
) -> np.ndarray:
    """Create a diagnostic review image without modifying the clean video."""
    output = frame.copy()
    cv2.rectangle(
        output,
        (0, 0),
        (WIDTH - 1, 24),
        (10, 10, 10),
        -1,
    )
    cv2.putText(
        output,
        "GREEN=label eligible | AMBER=projected but ignored",
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    for key in ALL_TARGET_KEYS:
        if not bool(
            row.get(
                f"{key}_visible",
                0,
            )
        ):
            continue

        box = _valid_box(
            row,
            key,
        )
        if box is None:
            continue

        eligible = bool(
            row.get(
                f"{key}_label_eligible",
                0,
            )
        )
        colour = (
            (70, 220, 80)
            if eligible
            else (0, 180, 255)
        )
        x1, y1, x2, y2 = box
        left = int(
            round(
                fogbase.clamp(
                    x1,
                    0.0,
                    WIDTH - 1.0,
                )
            )
        )
        top = int(
            round(
                fogbase.clamp(
                    y1,
                    0.0,
                    HEIGHT - 1.0,
                )
            )
        )
        right = int(
            round(
                fogbase.clamp(
                    x2,
                    0.0,
                    WIDTH - 1.0,
                )
            )
        )
        bottom = int(
            round(
                fogbase.clamp(
                    y2,
                    0.0,
                    HEIGHT - 1.0,
                )
            )
        )
        cv2.rectangle(
            output,
            (left, top),
            (right, bottom),
            colour,
            1,
            cv2.LINE_AA,
        )

        try:
            depth_text = (
                f"{float(row[f'{key}_depth_m']):.1f}m"
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            depth_text = "?m"

        label = (
            f"{key} {depth_text} "
            + (
                "LABEL"
                if eligible
                else "IGNORE"
            )
        )
        text_y = max(
            38,
            top - 4,
        )
        cv2.putText(
            output,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            colour,
            1,
            cv2.LINE_AA,
        )

    return output


def sidewalk_transform(
    carla_map: carla.Map,
    anchor: carla.Waypoint,
    preferred_side: float,
) -> carla.Transform:
    base = (
        anchor.transform
    )
    right = (
        base.get_right_vector()
    )

    offsets = (
        preferred_side
        * 4.5,
        preferred_side
        * 5.3,
        preferred_side
        * 6.1,
        -preferred_side
        * 4.5,
        -preferred_side
        * 5.3,
        -preferred_side
        * 6.1,
    )

    for lateral in offsets:
        probe = carla.Location(
            x=float(
                base.location.x
                + right.x
                * lateral
            ),
            y=float(
                base.location.y
                + right.y
                * lateral
            ),
            z=float(
                base.location.z
                + 0.20
            ),
        )

        sidewalk = (
            carla_map.get_waypoint(
                probe,
                project_to_road=True,
                lane_type=(
                    carla.LaneType.Sidewalk
                ),
            )
        )

        if sidewalk is None:
            continue

        if (
            sidewalk.transform.location.distance(
                base.location
            )
            > 12.0
        ):
            continue

        location = (
            sidewalk.transform.location
        )

        yaw = float(
            base.rotation.yaw
            + (
                90.0
                if lateral
                > 0.0
                else -90.0
            )
        )

        return carla.Transform(
            carla.Location(
                x=float(
                    location.x
                ),
                y=float(
                    location.y
                ),
                z=float(
                    location.z
                    + 0.35
                ),
            ),
            carla.Rotation(
                yaw=yaw,
            ),
        )

    raise RuntimeError(
        "No safe Town10 sidewalk found for standing pedestrian."
    )


def spawn_static_person(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    transform: carla.Transform,
    role_name: str,
) -> carla.Walker:
    if blueprint.has_attribute(
        "role_name"
    ):
        blueprint.set_attribute(
            "role_name",
            role_name,
        )

    if blueprint.has_attribute(
        "is_invincible"
    ):
        blueprint.set_attribute(
            "is_invincible",
            "false",
        )

    actor = world.try_spawn_actor(
        blueprint,
        transform,
    )

    if actor is None:
        retry_transform = (
            carla.Transform(
                carla.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=(
                        transform.location.z
                        + 0.30
                    ),
                ),
                transform.rotation,
            )
        )

        actor = (
            world.try_spawn_actor(
                blueprint,
                retry_transform,
            )
        )

    if actor is None:
        raise RuntimeError(
            f"Could not spawn {role_name}."
        )

    try:
        actor.apply_control(
            carla.WalkerControl(
                direction=(
                    carla.Vector3D(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                    )
                ),
                speed=0.0,
                jump=False,
            )
        )
    except RuntimeError:
        pass

    # They only need to stand beside the road.
    try:
        actor.set_simulate_physics(
            False
        )
    except (
        AttributeError,
        RuntimeError,
    ):
        pass

    return actor


def write_actor_gt(
    row: dict[str, Any],
    key: str,
    actor: carla.Actor,
    camera: carla.Sensor,
    intrinsic: np.ndarray,
) -> bool:
    (
        visible,
        x1,
        y1,
        x2,
        y2,
        u,
        v,
        depth,
    ) = fogbase.project_actor_box(
        actor,
        camera,
        intrinsic,
        WIDTH,
        HEIGHT,
    )

    values = (
        x1,
        y1,
        x2,
        y2,
        u,
        v,
        depth,
    )
    try:
        finite_projection = all(
            math.isfinite(
                float(value)
            )
            for value in values
        )
    except (
        TypeError,
        ValueError,
    ):
        finite_projection = False
    valid_box = (
        finite_projection
        and x2 > x1
        and y2 > y1
    )
    projected_visible = bool(
        visible
        and valid_box
    )

    row[
        f"{key}_visible"
    ] = int(
        projected_visible
    )
    row[
        f"{key}_x1"
    ] = x1
    row[
        f"{key}_y1"
    ] = y1
    row[
        f"{key}_x2"
    ] = x2
    row[
        f"{key}_y2"
    ] = y2
    row[
        f"{key}_u"
    ] = u
    row[
        f"{key}_v"
    ] = v
    row[
        f"{key}_depth_m"
    ] = depth

    box_width = (
        float(x2 - x1)
        if valid_box
        else 0.0
    )
    box_height = (
        float(y2 - y1)
        if valid_box
        else 0.0
    )
    box_area = (
        box_width * box_height
    )

    clipped_x1 = fogbase.clamp(
        float(x1) if finite_projection else 0.0,
        0.0,
        float(WIDTH),
    )
    clipped_y1 = fogbase.clamp(
        float(y1) if finite_projection else 0.0,
        0.0,
        float(HEIGHT),
    )
    clipped_x2 = fogbase.clamp(
        float(x2) if finite_projection else 0.0,
        0.0,
        float(WIDTH),
    )
    clipped_y2 = fogbase.clamp(
        float(y2) if finite_projection else 0.0,
        0.0,
        float(HEIGHT),
    )
    inside_area = max(
        0.0,
        clipped_x2 - clipped_x1,
    ) * max(
        0.0,
        clipped_y2 - clipped_y1,
    )
    inside_fraction = (
        inside_area / box_area
        if box_area > 0.0
        else 0.0
    )
    edge_touching = bool(
        projected_visible
        and (
            x1 <= FRAME_EDGE_MARGIN_PX
            or y1 <= FRAME_EDGE_MARGIN_PX
            or x2 >= WIDTH - FRAME_EDGE_MARGIN_PX
            or y2 >= HEIGHT - FRAME_EDGE_MARGIN_PX
            or inside_fraction < 0.995
        )
    )
    centre_inside = bool(
        finite_projection
        and 0.0 <= u < WIDTH
        and 0.0 <= v < HEIGHT
    )

    reasons: list[str] = []
    if not projected_visible:
        reasons.append(
            "not_projected_visible"
        )
    else:
        if not (
            MIN_LABEL_DEPTH_M
            <= depth
            <= MAX_LABEL_DEPTH_M
        ):
            reasons.append(
                "depth_outside_policy"
            )
        if not centre_inside:
            reasons.append(
                "centre_outside_frame"
            )
        if box_width < MIN_LABEL_WIDTH_PX:
            reasons.append(
                "bbox_too_narrow"
            )
        if box_height < MIN_LABEL_HEIGHT_PX:
            reasons.append(
                "bbox_too_short"
            )
        if box_area < MIN_LABEL_AREA_PX2:
            reasons.append(
                "bbox_area_too_small"
            )
        if edge_touching:
            reasons.append(
                "edge_touching_or_truncated"
            )

    label_eligible = bool(
        projected_visible
        and not reasons
    )

    row[f"{key}_bbox_width_px"] = box_width
    row[f"{key}_bbox_height_px"] = box_height
    row[f"{key}_bbox_area_px2"] = box_area
    row[f"{key}_bbox_inside_fraction"] = inside_fraction
    row[f"{key}_edge_touching"] = int(
        edge_touching
    )
    row[f"{key}_label_eligible"] = int(
        label_eligible
    )
    row[f"{key}_ignore_for_detection"] = int(
        projected_visible
        and not label_eligible
    )
    row[f"{key}_label_reason"] = (
        "eligible"
        if label_eligible
        else "|".join(reasons)
    )

    # Backward-compatible alias: an encounter is now a fair detection label,
    # not merely any projected point inside the camera image.
    row[
        f"{key}_encounter"
    ] = int(
        label_eligible
    )

    return bool(
        label_eligible
    )


def summarize_image_quality(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate source/final diagnostics for the JSON report."""
    summary: dict[
        str,
        dict[str, float],
    ] = {}
    metric_names = (
        "luma_mean",
        "luma_p01",
        "luma_p50",
        "luma_p99",
        "shadow_fraction",
        "highlight_fraction",
        "sharpness_laplacian",
    )

    for prefix in (
        "source",
        "night",
    ):
        summary[prefix] = {}
        for metric_name in metric_names:
            key = f"{prefix}_{metric_name}"
            values = np.asarray(
                [
                    float(row[key])
                    for row in rows
                ],
                dtype=np.float64,
            )
            summary[prefix][
                f"mean_{metric_name}"
            ] = float(
                np.mean(values)
            )
            summary[prefix][
                f"median_{metric_name}"
            ] = float(
                np.median(values)
            )

    return summary


def summarize_target_quality(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarise projected, ignored, edge-touching, and eligible target boxes."""
    summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for key in ALL_TARGET_KEYS:
        projected_rows = [
            row
            for row in rows
            if bool(
                row.get(
                    f"{key}_visible",
                    0,
                )
            )
        ]
        eligible_rows = [
            row
            for row in rows
            if bool(
                row.get(
                    f"{key}_label_eligible",
                    0,
                )
            )
        ]
        areas = np.asarray(
            [
                float(
                    row[f"{key}_bbox_area_px2"]
                )
                for row in projected_rows
            ],
            dtype=np.float64,
        )
        depths = np.asarray(
            [
                float(
                    row[f"{key}_depth_m"]
                )
                for row in projected_rows
            ],
            dtype=np.float64,
        )

        summary[key] = {
            "projected_visible_frames": len(
                projected_rows
            ),
            "label_eligible_frames": len(
                eligible_rows
            ),
            "ignored_projected_frames": sum(
                int(
                    row.get(
                        f"{key}_ignore_for_detection",
                        0,
                    )
                )
                for row in rows
            ),
            "edge_touching_frames": sum(
                int(
                    row.get(
                        f"{key}_edge_touching",
                        0,
                    )
                )
                for row in rows
            ),
            "projected_bbox_area_px2": (
                {
                    "min": float(np.min(areas)),
                    "median": float(np.median(areas)),
                    "max": float(np.max(areas)),
                }
                if areas.size
                else None
            ),
            "projected_depth_m": (
                {
                    "min": float(np.min(depths)),
                    "median": float(np.median(depths)),
                    "max": float(np.max(depths)),
                }
                if depths.size
                else None
            ),
        }

    return summary


def destroy_actors_safely(
    client: carla.Client,
    actors: list[Optional[carla.Actor]],
) -> list[str]:
    """Destroy each live actor once, with a fallback only for failed commands.

    CARLA actor proxies can temporarily report ``is_alive=True`` after a
    successful server-side batch destroy.  Rechecking that stale property and
    destroying again produced harmless but alarming ``actor: not found``
    messages.  Batch responses are therefore the source of truth here.
    """
    unique: dict[
        int,
        carla.Actor,
    ] = {}
    for actor in actors:
        if actor is None:
            continue
        try:
            if actor.is_alive:
                unique[int(actor.id)] = actor
        except RuntimeError:
            continue

    if not unique:
        return []

    actor_list = list(
        unique.values()
    )
    errors: list[str] = []

    fallback_actors: list[
        carla.Actor
    ] = []
    batch_error: Optional[str] = None

    try:
        responses = client.apply_batch_sync(
            [
                carla.command.DestroyActor(
                    actor.id
                )
                for actor in actor_list
            ],
            True,
        )
    except RuntimeError as exc:
        responses = []
        batch_error = str(exc)
        fallback_actors.extend(
            actor_list
        )

    if batch_error is None:
        for actor, response in zip(
            actor_list,
            responses,
        ):
            response_error = str(
                response.error
                or ""
            ).strip()
            if not response_error:
                continue
            if "not found" in response_error.lower():
                # Already absent is the intended final state.
                continue
            fallback_actors.append(
                actor
            )

        # Be defensive if a CARLA build returns fewer responses than commands.
        if len(responses) < len(actor_list):
            fallback_actors.extend(
                actor_list[
                    len(responses):
                ]
            )

    # Retry only commands that did not receive a successful batch response.
    for actor in fallback_actors:
        try:
            actor.destroy()
        except RuntimeError as exc:
            message = str(exc)
            if "not found" in message.lower():
                continue
            errors.append(
                f"actor_{actor.id}: {message}"
            )

    if (
        batch_error is not None
        and errors
    ):
        errors.insert(
            0,
            f"batch_destroy: {batch_error}",
        )

    return errors


def main() -> None:
    args = parse_args()

    if (
        args.duration
        <= 1.0
    ):
        raise ValueError(
            "--duration must be > 1 second."
        )

    if not (
        0.60
        <= args.night_strength
        <= 1.35
    ):
        raise ValueError(
            "--night-strength must be between 0.60 and 1.35."
        )

    if not (
        0.0
        <= args.sensor_noise
        <= 1.5
    ):
        raise ValueError(
            "--sensor-noise must be between 0.0 and 1.5."
        )

    route_json = (
        args.route_json.resolve()
        if args.route_json
        else fogbase.latest_route_json()
    )

    (
        route_payload,
        driving_route,
        locked_transform,
    ) = fogbase.load_locked_transform(
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

    world = (
        client.get_world()
    )

    active_map = (
        world.get_map().name
    )

    if not active_map.endswith(
        f"/{args.town}"
    ):
        raise RuntimeError(
            f"Active map is {active_map}; expected {args.town}."
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

    vehicles: dict[
        str,
        carla.Vehicle,
    ] = {}

    vehicle_light_status: dict[
        str,
        str,
    ] = {}

    people: dict[
        str,
        carla.Walker,
    ] = {}

    source_writer: Optional[
        cv2.VideoWriter
    ] = None

    night_writer: Optional[
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

    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir = (
        output_root
        / (
            "run_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_video = (
        run_dir
        / "night_source_raw.mp4"
    )

    final_video = (
        run_dir
        / "town10_night_fullscene_FINAL.mp4"
    )

    metrics_path = (
        run_dir
        / "ground_truth_metrics.csv"
    )

    report_path = (
        run_dir
        / "capture_report.json"
    )

    print(
        "="
        * 100
    )
    print(
        "STEP 92.4 - TOWN10 CALIBRATED-ACTOR-LIGHT SYNTHETIC NIGHT"
    )
    print(
        "="
        * 100
    )
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
        "Architecture:   accepted Step 90 capture engine"
    )
    print(
        "Renderer:       positive sun +38 deg (stable CARLA RGB source)"
    )
    print(
        f"Night grade:    {NIGHT_GRADE_VERSION} | "
        f"strength={args.night_strength:.2f} | noise={args.sensor_noise:.2f}"
    )
    print(
        "Scene:          2 persons | ego -> motorcycle -> car | bus -> truck"
    )
    print(
        f"Camera:         {WIDTH}x{HEIGHT}@{FPS:.0f} | world={WORLD_FPS:.0f} FPS"
    )
    print(
        f"Duration:       {args.duration:.1f}s | "
        f"{int(round(args.duration * FPS))} output frames"
    )
    print(
        "Vehicle lights: genuine CARLA Position + LowBeam "
        "+ rigid 3D actor-anchored lamp highlights"
    )
    print(
        "="
        * 100
    )

    try:
        # ------------------------------------------------------------------
        # EXACT proven Fog world architecture.
        # ------------------------------------------------------------------
        settings = (
            world.get_settings()
        )
        settings.synchronous_mode = (
            True
        )
        settings.fixed_delta_seconds = (
            1.0
            / WORLD_FPS
        )
        settings.no_rendering_mode = (
            False
        )

        world.apply_settings(
            settings
        )

        traffic_manager.set_synchronous_mode(
            True
        )
        traffic_manager.set_random_device_seed(
            args.seed
        )

        # IMPORTANT: positive-sun stable renderer.
        world.set_weather(
            stable_source_weather()
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
                "Locked route could not be projected to Town10 driving lane."
            )

        # Same geometry preflight as accepted Fog.
        _ = fogbase.advance_straight(
            start_waypoint,
            92.0,
        )

        all_waypoints = (
            carla_map.generate_waypoints(
                2.0
            )
        )

        forward_motorcycle_wp = (
            fogbase.advance_straight(
                start_waypoint,
                FORWARD_MOTORCYCLE_DISTANCE_M,
            )
        )

        forward_car_wp = (
            fogbase.advance_straight(
                start_waypoint,
                FORWARD_CAR_DISTANCE_M,
            )
        )

        bus_route_wp = (
            fogbase.advance_straight(
                start_waypoint,
                ONCOMING_BUS_ROUTE_DISTANCE_M,
            )
        )

        truck_route_wp = (
            fogbase.advance_straight(
                start_waypoint,
                ONCOMING_TRUCK_ROUTE_DISTANCE_M,
            )
        )

        oncoming_bus_wp = (
            fogbase.find_oncoming_waypoint(
                all_waypoints,
                bus_route_wp,
            )
        )

        oncoming_truck_wp = (
            fogbase.find_oncoming_waypoint(
                all_waypoints,
                truck_route_wp,
            )
        )

        spawn_waypoints = {
            "forward_motorcycle": (
                forward_motorcycle_wp
            ),
            "forward_car": (
                forward_car_wp
            ),
            "oncoming_bus": (
                oncoming_bus_wp
            ),
            "oncoming_truck": (
                oncoming_truck_wp
            ),
        }

        library = (
            world.get_blueprint_library()
        )

        # ------------------------------------------------------------------
        # Ego + exact proven target blueprints.
        # ------------------------------------------------------------------
        ego_blueprint = (
            library.find(
                "vehicle.tesla.model3"
            )
        )

        fogbase.configure_blueprint(
            ego_blueprint,
            f"{ROLE_PREFIX}ego",
            color_index=0,
        )

        ego = world.try_spawn_actor(
            ego_blueprint,
            fogbase.raised_transform(
                start_waypoint.transform,
                0.45,
            ),
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn ego."
            )

        for index, key in enumerate(
            VEHICLE_KEYS,
            start=1,
        ):
            blueprint = (
                library.find(
                    TARGET_BLUEPRINTS[
                        key
                    ]
                )
            )

            fogbase.configure_blueprint(
                blueprint,
                f"{ROLE_PREFIX}{key}",
                color_index=index,
            )

            actor = (
                world.try_spawn_actor(
                    blueprint,
                    fogbase.raised_transform(
                        spawn_waypoints[
                            key
                        ].transform,
                        0.45,
                    ),
                )
            )

            if actor is None:
                raise RuntimeError(
                    f"Could not spawn {key}."
                )

            vehicles[
                key
            ] = actor

        # ------------------------------------------------------------------
        # Exact Fog camera architecture.
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
            str(
                WIDTH
            ),
        )
        camera_blueprint.set_attribute(
            "image_size_y",
            str(
                HEIGHT
            ),
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
                fogbase.image_to_bgr(
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

        # ------------------------------------------------------------------
        # EXACT accepted Fog startup:
        # 6 settle ticks BEFORE Traffic Manager.
        # ------------------------------------------------------------------
        for _ in range(
            6
        ):
            world.tick()

        fogbase.configure_autopilot(
            traffic_manager,
            ego,
            EGO_SPEED_KMH,
            args.tm_port,
        )

        for key in VEHICLE_KEYS:
            fogbase.configure_autopilot(
                traffic_manager,
                vehicles[
                    key
                ],
                TARGET_SPEED_KMH[
                    key
                ],
                args.tm_port,
            )

        vehicle_light_status = enable_genuine_vehicle_lights(
            {
                "ego": ego,
                **vehicles,
            }
        )
        print(
            "Genuine lights: "
            + ", ".join(
                f"{key}={value}"
                for key, value in vehicle_light_status.items()
            ),
            flush=True,
        )

        hidden_ticks = int(
            WORLD_FPS
            * 1.0
        )

        for _ in range(
            hidden_ticks
        ):
            world.tick()

        # Flush hidden camera frames.
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        # ------------------------------------------------------------------
        # Persons are added AFTER the exact proven vehicle startup.
        # This cannot disturb the startup path that already worked in Fog.
        # ------------------------------------------------------------------
        walker_pool = sorted(
            library.filter(
                "walker.pedestrian.*"
            ),
            key=lambda blueprint: (
                blueprint.id
            ),
        )

        if len(
            walker_pool
        ) < 2:
            raise RuntimeError(
                "Two walker blueprints are required."
            )

        person_anchor_1 = (
            fogbase.advance_straight(
                start_waypoint,
                PERSON_1_ROUTE_DISTANCE_M,
            )
        )

        person_anchor_2 = (
            fogbase.advance_straight(
                start_waypoint,
                PERSON_2_ROUTE_DISTANCE_M,
            )
        )

        people[
            "person_1"
        ] = spawn_static_person(
            world,
            walker_pool[
                min(
                    1,
                    len(
                        walker_pool
                    )
                    - 1,
                )
            ],
            sidewalk_transform(
                carla_map,
                person_anchor_1,
                -1.0,
            ),
            f"{ROLE_PREFIX}person_1",
        )

        people[
            "person_2"
        ] = spawn_static_person(
            world,
            walker_pool[
                min(
                    13,
                    len(
                        walker_pool
                    )
                    - 1,
                )
            ],
            sidewalk_transform(
                carla_map,
                person_anchor_2,
                1.0,
            ),
            f"{ROLE_PREFIX}person_2",
        )

        print(
            "Standing persons spawned: "
            f"{people['person_1'].id}, "
            f"{people['person_2'].id}",
            flush=True,
        )

        source_writer = (
            fogbase.create_writer(
                source_video,
                FPS,
                WIDTH,
                HEIGHT,
            )
        )

        night_writer = (
            fogbase.create_writer(
                final_video,
                FPS,
                WIDTH,
                HEIGHT,
            )
        )

        intrinsic = (
            fogbase.camera_intrinsic(
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
            in ALL_TARGET_KEYS
        }

        encounter_counts = {
            key: 0
            for key
            in ALL_TARGET_KEYS
        }

        ignored_counts = {
            key: 0
            for key
            in ALL_TARGET_KEYS
        }

        edge_touching_counts = {
            key: 0
            for key
            in ALL_TARGET_KEYS
        }

        first_visible = {
            key: None
            for key
            in ALL_TARGET_KEYS
        }

        first_encounter = {
            key: None
            for key
            in ALL_TARGET_KEYS
        }

        recorded = 0
        capture_start_timestamp: Optional[
            float
        ] = None

        review_frames = {
            1,
            max(
                1,
                required_frames
                // 4,
            ),
            max(
                1,
                required_frames
                // 2,
            ),
            max(
                1,
                3
                * required_frames
                // 4,
            ),
            required_frames,
        }

        print(
            "CAPTURE STARTED - stable CARLA source + CPU night grading",
            flush=True,
        )

        while (
            recorded
            < required_frames
        ):
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

            for (
                carla_frame,
                image_timestamp,
                source_frame,
            ) in available:
                if (
                    recorded
                    >= required_frames
                ):
                    break

                if (
                    capture_start_timestamp
                    is None
                ):
                    capture_start_timestamp = (
                        image_timestamp
                    )

                recorded += 1

                assert source_writer is not None
                source_writer.write(
                    source_frame
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
                    "source_sun_altitude_deg": (
                        SOURCE_SUN_ALTITUDE_DEG
                    ),
                    "night_strength": (
                        args.night_strength
                    ),
                    "sensor_noise_strength": (
                        args.sensor_noise
                    ),
                    "grading_seed": (
                        args.seed
                    ),
                    "ego_speed_mps": (
                        fogbase.speed_mps(
                            ego
                        )
                    ),
                    "ego_yaw": float(
                        ego.get_transform().rotation.yaw
                    ),
                }

                visible_now: list[str] = []
                eligible_now: list[str] = []
                ignored_now: list[str] = []

                for key in VEHICLE_KEYS:
                    actor = (
                        vehicles[
                            key
                        ]
                    )

                    encounter = (
                        write_actor_gt(
                            row,
                            key,
                            actor,
                            camera,
                            intrinsic,
                        )
                    )

                    row[
                        f"{key}_speed_mps"
                    ] = fogbase.speed_mps(
                        actor
                    )

                    row[
                        f"{key}_yaw"
                    ] = float(
                        actor.get_transform().rotation.yaw
                    )

                    if bool(
                        row[
                            f"{key}_visible"
                        ]
                    ):
                        visible_counts[
                            key
                        ] += 1

                        visible_now.append(
                            key
                        )

                        if (
                            first_visible[
                                key
                            ]
                            is None
                        ):
                            first_visible[
                                key
                            ] = (
                                recorded
                            )

                    if encounter:
                        encounter_counts[
                            key
                        ] += 1

                        eligible_now.append(
                            key
                        )

                        if (
                            first_encounter[
                                key
                            ]
                            is None
                        ):
                            first_encounter[
                                key
                            ] = (
                                recorded
                            )

                    if bool(
                        row[
                            f"{key}_ignore_for_detection"
                        ]
                    ):
                        ignored_counts[
                            key
                        ] += 1
                        ignored_now.append(
                            key
                        )

                    if bool(
                        row[
                            f"{key}_edge_touching"
                        ]
                    ):
                        edge_touching_counts[
                            key
                        ] += 1

                for key in PERSON_KEYS:
                    actor = (
                        people[
                            key
                        ]
                    )

                    encounter = (
                        write_actor_gt(
                            row,
                            key,
                            actor,
                            camera,
                            intrinsic,
                        )
                    )

                    row[
                        f"{key}_speed_mps"
                    ] = 0.0

                    row[
                        f"{key}_yaw"
                    ] = float(
                        actor.get_transform().rotation.yaw
                    )

                    if bool(
                        row[
                            f"{key}_visible"
                        ]
                    ):
                        visible_counts[
                            key
                        ] += 1

                        visible_now.append(
                            key
                        )

                        if (
                            first_visible[
                                key
                            ]
                            is None
                        ):
                            first_visible[
                                key
                            ] = (
                                recorded
                            )

                    if encounter:
                        encounter_counts[
                            key
                        ] += 1

                        eligible_now.append(
                            key
                        )

                        if (
                            first_encounter[
                                key
                            ]
                            is None
                        ):
                            first_encounter[
                                key
                            ] = (
                                recorded
                            )

                    if bool(
                        row[
                            f"{key}_ignore_for_detection"
                        ]
                    ):
                        ignored_counts[
                            key
                        ] += 1
                        ignored_now.append(
                            key
                        )

                    if bool(
                        row[
                            f"{key}_edge_touching"
                        ]
                    ):
                        edge_touching_counts[
                            key
                        ] += 1

                row[
                    "controlled_visible_count"
                ] = len(
                    visible_now
                )
                row[
                    "controlled_label_eligible_count"
                ] = len(
                    eligible_now
                )
                row[
                    "controlled_ignored_count"
                ] = len(
                    ignored_now
                )

                night_frame = (
                    night_grade_frame(
                        source_frame,
                        args.night_strength,
                        recorded,
                        args.sensor_noise,
                        args.seed,
                    )
                )

                if args.actor_anchored_light_cues:
                    night_frame = (
                        add_actor_anchored_light_cues(
                            night_frame,
                            row,
                            vehicles,
                            camera,
                            intrinsic,
                        )
                    )

                row.update(
                    frame_visual_metrics(
                        source_frame,
                        "source",
                    )
                )
                row.update(
                    frame_visual_metrics(
                        night_frame,
                        "night",
                    )
                )
                rows.append(
                    row
                )

                assert night_writer is not None
                night_writer.write(
                    night_frame
                )

                if (
                    recorded
                    in review_frames
                ):
                    cv2.imwrite(
                        str(
                            run_dir
                            / (
                                "night_review_"
                                f"{recorded:03d}.png"
                            )
                        ),
                        night_frame,
                    )
                    cv2.imwrite(
                        str(
                            run_dir
                            / (
                                "night_review_"
                                f"{recorded:03d}_ANNOTATED.png"
                            )
                        ),
                        annotate_review_frame(
                            night_frame,
                            row,
                        ),
                    )

                if (
                    recorded
                    % 20
                    == 0
                ):
                    print(
                        f"Captured {recorded}/{required_frames} | "
                        f"projected={visible_now} | "
                        f"labels={eligible_now}",
                        flush=True,
                    )

        if not rows:
            raise RuntimeError(
                "RGB camera produced no frames."
            )

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

        minimum_encounter_frames = max(
            3,
            int(
                FPS
                * 0.3
            ),
        )

        image_quality_summary = (
            summarize_image_quality(
                rows
            )
        )
        target_quality_summary = (
            summarize_target_quality(
                rows
            )
        )
        source_mean_luma = float(
            image_quality_summary[
                "source"
            ][
                "mean_luma_mean"
            ]
        )
        night_mean_luma = float(
            image_quality_summary[
                "night"
            ][
                "mean_luma_mean"
            ]
        )
        night_median_luma = float(
            image_quality_summary[
                "night"
            ][
                "median_luma_p50"
            ]
        )

        checks = {
            "full_frame_count": (
                recorded
                == required_frames
            ),
            "motorcycle_seen": (
                encounter_counts[
                    "forward_motorcycle"
                ]
                >= minimum_encounter_frames
            ),
            "car_seen": (
                encounter_counts[
                    "forward_car"
                ]
                >= minimum_encounter_frames
            ),
            "bus_seen": (
                encounter_counts[
                    "oncoming_bus"
                ]
                >= minimum_encounter_frames
            ),
            "truck_seen": (
                encounter_counts[
                    "oncoming_truck"
                ]
                >= minimum_encounter_frames
            ),
            "person_1_seen": (
                encounter_counts[
                    "person_1"
                ]
                >= minimum_encounter_frames
            ),
            "person_2_seen": (
                encounter_counts[
                    "person_2"
                ]
                >= minimum_encounter_frames
            ),
            "night_darker_than_source": (
                night_mean_luma
                < source_mean_luma * 0.78
            ),
            "night_not_fully_crushed": (
                night_median_luma
                >= 5.0
            ),
        }

        failure_reasons = [
            key
            for key, passed
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
            "script_version": (
                SCRIPT_VERSION
            ),
            "map": active_map,
            "route_json": str(
                route_json
            ),
            "route_rank": (
                args.route_rank
            ),
            "scene": (
                "Town10 straight full night presentation: "
                "2 standing persons; ego -> motorcycle -> car; "
                "oncoming bus -> truck"
            ),
            "capture_architecture": (
                "Accepted Step 90 synchronous Town10 engine"
            ),
            "renderer_method": {
                "native_negative_sun": (
                    False
                ),
                "source_sun_altitude_angle": (
                    SOURCE_SUN_ALTITUDE_DEG
                ),
                "final_visual": (
                    "Post-processed synthetic night from genuine CARLA RGB"
                ),
                "grading_version": (
                    NIGHT_GRADE_VERSION
                ),
                "night_strength": (
                    args.night_strength
                ),
                "sensor_noise_strength": (
                    args.sensor_noise
                ),
                "genuine_vehicle_lights": (
                    vehicle_light_status
                ),
                "actor_anchored_light_cues_enabled": (
                    args.actor_anchored_light_cues
                ),
                "lamp_lateral_fraction": (
                    LAMP_LATERAL_FRACTION
                ),
                "lamp_center_y_bias": (
                    LAMP_CENTER_Y_BIAS
                ),
                "legacy_bbox_light_cues_enabled": (
                    False
                ),
                "deterministic_seed": (
                    args.seed
                ),
                "reason": (
                    "Native negative-sun Town10 caused repeatable "
                    "UE4/D3D device-loss on this machine."
                ),
                "scientific_use_note": (
                    "This output is a controlled CPU day-for-night domain, "
                    "not native physically rendered CARLA night illumination."
                ),
            },
            "ground_truth_policy": {
                "raw_projection_retained": (
                    True
                ),
                "label_eligible_requires": {
                    "minimum_depth_m": (
                        MIN_LABEL_DEPTH_M
                    ),
                    "maximum_depth_m": (
                        MAX_LABEL_DEPTH_M
                    ),
                    "minimum_bbox_width_px": (
                        MIN_LABEL_WIDTH_PX
                    ),
                    "minimum_bbox_height_px": (
                        MIN_LABEL_HEIGHT_PX
                    ),
                    "minimum_bbox_area_px2": (
                        MIN_LABEL_AREA_PX2
                    ),
                    "centre_inside_frame": (
                        True
                    ),
                    "edge_touching_or_truncated": (
                        False
                    ),
                },
                "encounter_field_semantics": (
                    "Backward-compatible alias for label_eligible."
                ),
            },
            "resolution": [
                WIDTH,
                HEIGHT,
            ],
            "fps": FPS,
            "world_fps": (
                WORLD_FPS
            ),
            "frame_count": (
                recorded
            ),
            "first_visible_frame": (
                first_visible
            ),
            "visible_frame_counts": (
                visible_counts
            ),
            "first_encounter_frame": (
                first_encounter
            ),
            "encounter_frame_counts": (
                encounter_counts
            ),
            "first_label_eligible_frame": (
                first_encounter
            ),
            "label_eligible_frame_counts": (
                encounter_counts
            ),
            "ignored_projected_frame_counts": (
                ignored_counts
            ),
            "edge_touching_frame_counts": (
                edge_touching_counts
            ),
            "target_quality_summary": (
                target_quality_summary
            ),
            "image_quality_summary": (
                image_quality_summary
            ),
            "checks": checks,
            "failure_reasons": (
                failure_reasons
            ),
            "source_video": str(
                source_video
            ),
            "final_night_video": str(
                final_video
            ),
            "ground_truth_metrics": str(
                metrics_path
            ),
            "clean_review_frames": [
                str(
                    run_dir
                    / (
                        "night_review_"
                        f"{frame_number:03d}.png"
                    )
                )
                for frame_number
                in sorted(
                    review_frames
                )
            ],
            "annotated_review_frames": [
                str(
                    run_dir
                    / (
                        "night_review_"
                        f"{frame_number:03d}_ANNOTATED.png"
                    )
                )
                for frame_number
                in sorted(
                    review_frames
                )
            ],
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "="
            * 100
        )
        print(
            f"FINAL STATUS: {status}"
        )
        print(
            f"First label-eligible frames: {first_encounter}"
        )
        print(
            f"Label-eligible counts: {encounter_counts}"
        )
        print(
            f"Ignored projected counts: {ignored_counts}"
        )
        print(
            "Mean luma source -> night: "
            f"{source_mean_luma:.1f} -> {night_mean_luma:.1f}"
        )
        print(
            f"SOURCE VIDEO: {source_video}"
        )
        print(
            f"FINAL NIGHT VIDEO: {final_video}"
        )
        print(
            f"REPORT: {report_path}"
        )
        print(
            "="
            * 100
        )

    finally:
        cleanup_errors: list[str] = []

        # Stop sensor callbacks before releasing writers or destroying actors.
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError as exc:
                cleanup_errors.append(
                    f"camera_stop: {exc}"
                )

        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break

        if source_writer is not None:
            try:
                source_writer.release()
            except Exception as exc:
                cleanup_errors.append(
                    f"source_writer_release: {exc}"
                )

        if night_writer is not None:
            try:
                night_writer.release()
            except Exception as exc:
                cleanup_errors.append(
                    f"night_writer_release: {exc}"
                )

        cleanup_errors.extend(
            destroy_actors_safely(
                client,
                [
                    camera,
                    *people.values(),
                    *vehicles.values(),
                    ego,
                ],
            )
        )

        try:
            traffic_manager.set_synchronous_mode(
                False
            )
        except RuntimeError as exc:
            cleanup_errors.append(
                f"traffic_manager_restore: {exc}"
            )

        try:
            world.apply_settings(
                original_settings
            )
        except RuntimeError as exc:
            cleanup_errors.append(
                f"world_settings_restore: {exc}"
            )

        try:
            world.set_weather(
                original_weather
            )
        except RuntimeError as exc:
            cleanup_errors.append(
                f"weather_restore: {exc}"
            )

        # Drop Python-side libcarla actor references before interpreter exit.
        people.clear()
        vehicles.clear()
        camera = None
        ego = None
        source_writer = None
        night_writer = None
        actor = None
        traffic_manager = None
        world = None
        client = None
        original_settings = None
        original_weather = None
        gc.collect()

        if cleanup_errors:
            print(
                "CLEANUP WARNING: "
                + " | ".join(
                    cleanup_errors
                ),
                flush=True,
            )
        else:
            print(
                "CLEANUP STATUS: PASS",
                flush=True,
            )


if __name__ == "__main__":
    main()


