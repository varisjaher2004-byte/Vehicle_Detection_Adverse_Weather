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
| Lock/reference route construction | `src/carla/CLEAR/step_67_lock_town10_route_from_generated.py` |
| Clear raw scene | `src/carla/CLEAR/step_68_v3_generate_town10_final_raw_demo.py` |
| Clear detected scene | `src/carla/CLEAR/step_69_generate_town10_detected_demo.py` |
| Clear stable output | `src/carla/CLEAR/step_74_generate_town10_clear_stable_final.py` |
| Rain scene construction | `src/carla/RAIN/step_80_generate_town10_rain_signal_continuation_v8.py` |
| Rain stable detection | `src/carla/RAIN/step_86_generate_town10_rain_FINAL_STABLE.py` |
| Rain accepted presentation | `src/carla/RAIN/step_87_generate_town10_rain_FINAL_PERFECT_STABLE.py` |
| Fog scene construction | `src/carla/FOG/step_90_generate_town10_fog_multivehicle.py` |
| Fog accepted presentation | `src/carla/FOG/step_91_generate_town10_fog_FINAL_PERFECT_STABLE.py` |
| Night-looking capture | `src/carla/NIGHT/step_92_4_generate_town10_night_CALIBRATED_LIGHTS_FINAL.py` |
| Night actor-association diagnostic | `src/carla/NIGHT/step_93_1_detect_town10_night_ACTOR_LIGHTS_FINAL.py` |

Use `python <script> --help` to inspect each stage's explicit model, input, output, route, host, and port options. Some later stages require artefacts created by earlier stages, and trained weights are intentionally excluded from Git.

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
