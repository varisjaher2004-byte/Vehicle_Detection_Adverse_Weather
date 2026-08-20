# YOLO-Based Vehicle Detection Under Adverse Environmental Conditions

This repository contains the source code, experiment notebooks, configuration files, and selected evidence produced for an MSc Artificial Intelligence research project at Sheffield Hallam University.

The project evaluates the robustness and cross-domain generalisation of an Ultralytics YOLO detector under adverse environmental conditions. It combines real-world experiments using ACDC and DAWN with controlled CARLA simulation evidence for clear, rain, fog, and night-looking scenes.

## Research scope

The investigation addresses three related questions:

- how detection performance changes across adverse environmental conditions;
- how strongly performance depends on the source dataset and its annotation domain;
- whether combining datasets automatically improves cross-domain robustness.

YOLO is the implemented detector in this repository. Faster R-CNN and other architectures are discussed only as literature context in the associated dissertation; they were not implemented as experimental baselines here.

## Experimental design

### Real-world evidence

The approved real-world experiments are:

- ACDC: Fog, Rain, Snow, Night, and Entire;
- DAWN: Fog, Rain, and Entire;
- Combined: prepared ACDC and DAWN data under a shared eight-class schema.

The common class mapping is:

| ID | Class |
|---:|---|
| 0 | person |
| 1 | rider |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | train |
| 6 | motorcycle |
| 7 | bicycle |

Aggregate YOLO metrics therefore cover the annotated eight-class road-user schema, not vehicles alone, unless a result file explicitly states otherwise.

### Simulation evidence

The approved simulation evidence uses CARLA 0.9.16 and `Town10HD_Opt` under four conditions:

- Clear;
- Rain;
- Fog;
- Night-looking output.

The main capture configuration is 640×360 RGB at 10 sensor frames per second, with the synchronous world running at 20 ticks per second.

The night-looking sequence requires a specific integrity qualification: the scene geometry, actors, route, and RGB source frames are genuine CARLA outputs, but the final night appearance is a calibrated CPU day-for-night transformation. It must not be described as native physical night rendering.

## Evidence-integrity rules

The repository follows these reporting rules:

- only completed and verified experiment outputs are treated as dissertation evidence;
- missing-label, wrong-path, incomplete, and diagnostic runs are excluded from final findings;
- Precision, Recall, and mAP values must come from the same selected checkpoint;
- F1 is calculated from the Precision and Recall reported for that checkpoint;
- `mAP50-95` is reported only where it is available and verified;
- CARLA visual evidence is not treated as an identical quantitative benchmark to labelled real-world validation data;
- ground-truth-aided or stable presentation outputs are not described as raw YOLO recall or detection performance;
- the relaxed night actor-association diagnostic is not presented as a standard Ultralytics Precision/Recall/F1 evaluation;
- historical result files under `results/` are retained as provenance and should not be silently altered.

## Main findings represented by the evidence

The retained results support the following cautious conclusions:

- ACDC produced the strongest overall real-world performance in this investigation;
- DAWN performance was substantially weaker and more variable;
- the Combined experiment was intermediate rather than consistently superior;
- increasing dataset volume did not automatically improve generalisation;
- robustness was domain-dependent and sensitive to weather, annotation quality, class balance, and dataset composition.

These are within-study findings. They are not claims that one dataset or detector is universally superior.

## Repository structure

```text
Vehicle_Detection_Adverse_Weather/
|-- configs/
|   |-- ACDC/
|   |-- DAWN/
|   |-- COMBINED/
|   `-- CARLA/
|-- docs/
|   |-- ENVIRONMENT.md
|   |-- CARLA_REPRODUCIBILITY.md
|   |-- EVIDENCE_INTEGRITY.md
|   |-- EVIDENCE_MANIFEST.csv
|   `-- NOTEBOOK_PROVENANCE.md
|-- notebooks/
|   |-- ACDC_ENTIRE.ipynb
|   |-- ACDC_FOG.ipynb
|   |-- ACDC_NIGHT.ipynb
|   |-- ACDC_RAIN.ipynb
|   |-- DAWN_ENTIRE.ipynb
|   |-- DAWN_FOG.ipynb
|   |-- DAWN_RAIN.ipynb
|   |-- inference_pipeline.ipynb
|   `-- YOLO_COMBINED.ipynb
|-- results/
|   |-- ACDC/
|   |-- DAWN/
|   |-- COMBINED/
|   `-- CARLA/
|-- src/
|   |-- carla/
|   |   |-- CLEAR/
|   |   |-- RAIN/
|   |   |-- FOG/
|   |   `-- NIGHT/
|   |-- evaluation/
|   |   `-- utils/
|   `-- training/
|       `-- train_experiment.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Installation

The verified environment is documented in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

On Windows, create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned environment uses Python 3.12.10, Ultralytics 8.4.104, PyTorch 2.12.1 with CUDA 12.6, and CARLA Python API 0.9.16.

The pip `carla` package is only the Python client API. The full CARLA 0.9.16 simulator application must be installed and started separately for simulation scripts.

Configure Ultralytics to use the local parent dataset directory:

```cmd
yolo settings datasets_dir=D:/path/to/datasets
```

## Dataset acquisition

Raw datasets are intentionally excluded from Git because of size and third-party distribution conditions.

- ACDC must be obtained from the [official ACDC website](https://acdc.vision.ee.ethz.ch/). The object-detection annotations and anonymised RGB data are distributed separately.
- DAWN should be obtained through the source identified by the [DAWN dataset publication](https://arxiv.org/abs/2008.05402) and used under the provider's terms.
- CARLA must be obtained from the [official CARLA project](https://github.com/carla-simulator/carla).

A prepared local dataset root should follow this structure:

```text
datasets/
|-- ACDC/
|   |-- YOLO_ENTIRE/
|   |-- YOLO_FOG/
|   |-- YOLO_NIGHT/
|   |-- YOLO_RAIN/
|   `-- YOLO_SNOW/
|-- DAWN/
|   |-- YOLO_ENTIRE/
|   |-- YOLO_FOG/
|   `-- YOLO_RAIN/
`-- YOLO_COMBINED/
```

Each prepared YOLO dataset directory is expected to contain:

```text
images/train/
images/val/
labels/train/
labels/val/
```

Active YAML files under `configs/` use roots relative to the Ultralytics `datasets_dir` setting. Configure that setting for the local machine rather than inserting a user-specific absolute path into every active YAML. YAML copies under `results/` are historical records and should not be edited merely to suit another machine.

## Running real-world experiments

The notebooks contain dataset conversion, integrity checking, training, validation, and result-processing cells from the completed investigation.

Before executing a notebook:

1. obtain the relevant dataset under its original licence;
2. reproduce the expected image/label directory structure;
3. configure the Ultralytics dataset root and any notebook-specific model/output paths;
4. verify image-label stem matching and class IDs;
5. confirm the selected YAML file and model checkpoint;
6. execute cells in order in a compatible notebook environment.

The notebooks retain research provenance, including some original Windows paths. They are not guaranteed to be turn-key on a different computer without path configuration. Do not interpret historical cell output as evidence that a newly edited cell has been rerun.

For a portable training entry point, list and inspect the approved presets before starting a run:

```cmd
python src\training\train_experiment.py --list
python src\training\train_experiment.py --experiment ACDC_FOG --dry-run
python src\training\train_experiment.py --experiment ACDC_FOG
```

Preset provenance is recorded in `configs/TRAINING_PRESETS.json`. The archival notebook path audit and safe adaptation rules are documented in [`docs/NOTEBOOK_PROVENANCE.md`](docs/NOTEBOOK_PROVENANCE.md).

## Running CARLA scripts

Start the CARLA 0.9.16 server before running client scripts. A resource-conservative Windows launch example is:

```cmd
CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=320 -ResY=180 -nosound -d3d11
```

The principal approved scripts are:

| Condition | Script |
|---|---|
| Clear | `src/carla/CLEAR/step_74_generate_town10_clear_stable_final.py` |
| Rain | `src/carla/RAIN/step_87_generate_town10_rain_FINAL_PERFECT_STABLE.py` |
| Fog | `src/carla/FOG/step_91_generate_town10_fog_FINAL_PERFECT_STABLE.py` |
| Night capture | `src/carla/NIGHT/step_92_4_generate_town10_night_CALIBRATED_LIGHTS_FINAL.py` |
| Night diagnostic | `src/carla/NIGHT/step_93_1_detect_town10_night_ACTOR_LIGHTS_FINAL.py` |

Several scripts depend on neighbouring helper scripts and local model/output paths. Review their command-line help and path arguments before execution. Running CARLA can be GPU-intensive; simulator stability depends on compatible graphics drivers and system resources.

The locked route, environment variables, script-stage inventory, and interpretation boundaries are documented in [`docs/CARLA_REPRODUCIBILITY.md`](docs/CARLA_REPRODUCIBILITY.md).

## Results and artefacts

The `results/` directory contains selected compact evidence, including:

- training histories and validation summaries;
- performance curves and confusion matrices;
- real-world experiment CSV files;
- CARLA review frames, reports, and frame-level diagnostic files.

Raw datasets, model weights, videos, archives, caches, and failed or superseded runs are not included. Their absence is intentional and must not be mistaken for evidence that those artefacts were never used locally.

The approved real-world summary is recorded in [`docs/EVIDENCE_MANIFEST.csv`](docs/EVIDENCE_MANIFEST.csv), and its selection and interpretation rules are documented in [`docs/EVIDENCE_INTEGRITY.md`](docs/EVIDENCE_INTEGRITY.md). Legacy and independently maximised summaries are qualified in [`results/README.md`](results/README.md). Verify the portable configs, presets, and manifest directly from the committed sources with:

```cmd
python src\evaluation\verify_repository.py
```

## Reproducibility limitations

Exact numerical reproduction can be affected by:

- dataset version, split, and annotation conversion;
- random initialisation and data-loader behaviour;
- GPU, CUDA, driver, and library differences;
- augmentation and checkpoint-selection behaviour;
- excluded trained weights and raw datasets;
- CARLA rendering and Traffic Manager timing.

The repository supports transparent inspection and reconstruction of the method; it does not claim bit-for-bit determinism on unrelated systems.

## Data protection and ethics

No attempt is made to identify individuals or vehicles. Raw third-party driving data is not redistributed here. Users are responsible for complying with the original dataset licences, data-protection conditions, institutional ethics requirements, and CARLA/Ultralytics software licences.

## Project licence and citation

Third-party datasets and software retain their original licences. No separate licence for this repository's original source code is granted unless a `LICENSE` file is added. Contact the repository owner before reuse beyond inspection or academic assessment.

When citing the datasets, use their original publications and provider guidance. When citing this repository, identify the repository version or commit hash used so that the referenced evidence can be traced.
