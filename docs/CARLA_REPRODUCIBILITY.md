# CARLA Reproducibility and Evidence Boundaries

This document describes the approved controlled-simulation evidence and the practical requirements for inspecting or rerunning the CARLA scripts.

## Verified platform

- CARLA simulator and Python API: 0.9.16
- Map: `Town10HD_Opt`
- RGB output: 640×360
- Camera sensor rate: 10 frames per second
- Synchronous world rate: 20 ticks per second
- Verified client platform: Python 3.12.10 on Microsoft Windows

The pip `carla` package is only the Python client. The full CARLA simulator application must be installed separately and must be compatible with the client API.

## Environment variables

The scripts avoid a fixed workstation root by supporting:

- `CARLA_RESEARCH_ROOT`: parent directory containing local CARLA outputs, models, and generated runs;
- `DMSC_GENERATED_DATASET_ROOT`: generated weather-image root used when rebuilding the locked reference route.

If `CARLA_RESEARCH_ROOT` is not set, several scripts resolve it from the current working directory. Explicitly setting it is safer for a rerun.

Example Windows session:

```cmd
set CARLA_RESEARCH_ROOT=D:\path\to\CARLA
set DMSC_GENERATED_DATASET_ROOT=D:\path\to\Generated
```

## Starting CARLA

A resource-conservative Windows launch example is:

```cmd
CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=320 -ResY=180 -nosound -d3d11
```

Confirm the required map is available before running a capture:

```cmd
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(300); w=c.load_world('Town10HD_Opt'); print(w.get_map().name)"
```

## Locked route

The active route configuration is `configs/CARLA/locked_town10_reference_route.json`.

Its portable path metadata is relative to `DMSC_GENERATED_DATASET_ROOT`. The route-defining fields remain locked:

- selected spawn index: 36;
- road ID: 11;
- lane ID: 1;
- location: x 67.65974426269531, y 69.8227767944336, z 0.5999999642372131;
- rotation: pitch 0.0, yaw 0.07327299565076828, roll 0.0;
- camera blueprint: `sensor.camera.rgb`;
- camera transform: x 1.5, y 0.0, z 2.4;
- field of view: 90 degrees.

The copy under `results/CARLA/locked_town10_reference_route.json` is historical provenance and retains the original workstation paths. It should not be edited merely to make the active configuration portable.

Pass the active route explicitly when a script provides `--route-json`, rather than relying on discovery of an old local `run_*` directory.

## Approved scripts

| Purpose | Script |
|---|---|
| Lock/reference route construction | `src/carla/CLEAR/lock_town10_reference_route.py` |
| Clear raw scene | `src/carla/CLEAR/generate_town10_clear_raw_scene.py` |
| Clear detected scene | `src/carla/CLEAR/run_town10_clear_detection.py` |
| Clear stable output | `src/carla/CLEAR/generate_town10_clear_stable_output.py` |
| Rain scene construction | `src/carla/RAIN/generate_town10_rain_scene.py` |
| Rain stable detection | `src/carla/RAIN/run_town10_rain_stable_detection.py` |
| Rain accepted presentation | `src/carla/RAIN/generate_town10_rain_presentation_output.py` |
| Fog scene construction | `src/carla/FOG/generate_town10_fog_scene.py` |
| Fog accepted presentation | `src/carla/FOG/generate_town10_fog_presentation_output.py` |
| Synthetic-night capture | `src/carla/NIGHT/generate_town10_synthetic_night_scene.py` |
| Synthetic-night actor-association diagnostic | `src/carla/NIGHT/run_town10_synthetic_night_diagnostics.py` |

Use `python <script> --help` to inspect each stage's explicit model, input, output, route, host, and port options. Some later stages require artefacts created by earlier stages, and trained weights are intentionally excluded from Git.

Repository-facing scripts and selected presentation artefacts use descriptive lowercase filenames. Historical JSON fields may retain the exact filenames produced on the original workstation; they are provenance values rather than current repository paths. See [File-naming convention](FILE_NAMING_CONVENTION.md).

The Night capture imports helper functions from the accepted Fog capture script. Both files must remain in the committed `src/carla` hierarchy.

## Interpretation boundaries

- Clear, Rain, Fog, and Night-looking artefacts support controlled qualitative evaluation.
- They are not directly interchangeable with the labelled real-world Ultralytics validation benchmark.
- Stable or ground-truth-aided presentation outputs must not be described as raw detector recall.
- Actor-association diagnostics must not be described as standard Ultralytics Precision/Recall/F1.
- The Night-looking appearance is a calibrated CPU day-for-night transformation applied to genuine CARLA RGB frames. It is not native negative-sun physical night rendering.
- Absolute paths inside historical JSON reports document the original run environment; they are provenance, not portable commands.

## Validation

Syntax and repository-level metadata can be checked without starting CARLA:

```cmd
python -m compileall -q src
python src\evaluation\verify_repository.py
```

A complete runtime rerun additionally requires the CARLA server, excluded model weights, sufficient GPU resources, and the correct stage inputs.
