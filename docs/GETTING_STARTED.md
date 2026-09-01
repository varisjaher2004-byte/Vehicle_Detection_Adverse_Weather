# Getting Started

This guide separates simple research inspection from full experiment reproduction. You do not need the raw datasets or a GPU to inspect the committed evidence.

## 1. Choose the route you need

### Assessor or research reader

1. Read the [final dissertation](submission/Varis_Kureshi_Dissertation_SUBMISSION_READY_FINAL_2026-08-31.docx).
2. Open the [final defence presentation](submission/Varis_Kureshi_Dissertation_Defence_MSC_SUBMISSION_READY_FINAL_2026-09-01.pptx).
3. Inspect the [seven-cell matrix](../results/CORRECTED_2026-08-27/final_cross_domain_validation_matrix.csv).
4. Read [Evidence integrity](EVIDENCE_INTEGRITY.md) and the [DAWN correction record](LABEL_INTEGRITY_CORRECTION_2026-08-27.md) before interpreting older results.

### Evidence verifier

Only Python 3.12 and PyYAML are required:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install PyYAML==6.0.3
python src/evaluation/verify_repository.py
python src/evaluation/verify_submission_package.py
python -m compileall -q src
```

These checks are read-only. They verify the committed configuration and evidence; they do not start training.

### Experiment reproducer

Install the complete recorded dependency set:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file targets the recorded CUDA 12.6 environment. A CPU-only or different-CUDA reproduction may require a compatible PyTorch installation and must be documented as a deviation.

## 2. Obtain the external data and software

Raw data and trained weights are not redistributed.

- Obtain ACDC from the [official ACDC project](https://acdc.vision.ee.ethz.ch/).
- Obtain DAWN through the source identified by the [DAWN publication](https://doi.org/10.48550/arXiv.2008.05402).
- Obtain the CARLA 0.9.16 simulator from the [official CARLA project](https://github.com/carla-simulator/carla).
- Obtain the YOLOv8l pretrained weights through the official Ultralytics mechanism or an independently verified local source.

Respect each provider's terms. Do not add raw data or weights to this repository.

## 3. Prepare the dataset layout

The active YAMLs expect the following structure below the Ultralytics dataset root:

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

Every prepared YOLO directory must contain:

```text
images/train/
images/val/
labels/train/
labels/val/
```

Configure the local root once rather than inserting an absolute personal path into every YAML:

```cmd
yolo settings datasets_dir=D:\path\to\datasets
```

The active class order is fixed:

```text
0 person
1 rider
2 car
3 truck
4 bus
5 train
6 motorcycle
7 bicycle
```

## 4. Inspect or run the historical presets

List all approved experiment IDs:

```bash
python src/training/train_experiment.py --list
```

Resolve a run without starting it:

```bash
python src/training/train_experiment.py --experiment ACDC_FOG --dry-run
```

Start a new run only after checking the resolved data, model and output paths:

```bash
python src/training/train_experiment.py --experiment ACDC_FOG
```

New outputs default to `runs/reproduction/`, which is ignored by Git. Do not overwrite committed historical evidence.

## 5. Build the corrected DAWN view

The source directories are read-only inputs. The command below creates a separate corrected copy:

```cmd
python src\evaluation\prepare_corrected_dawn_labels.py ^
  --dawn-entire-root D:\data\DAWN\YOLO_ENTIRE ^
  --dawn-fog-root D:\data\DAWN\YOLO_FOG ^
  --dawn-rain-root D:\data\DAWN\YOLO_RAIN ^
  --fog-voc-root D:\data\DAWN\FOG_XML ^
  --output-root D:\data\DAWN_CORRECTED
```

Expected final-protocol counts are 422 training images, 120 validation images, 2,829 training instances and 1,029 validation instances. The retained Fog/Rain validation subsets contain 78 and 42 images respectively. A reproduction should stop if pairing, class range, normalised bounds or subset checks fail.

## 6. Inspect or run corrected training

Resolve the four-run plan first:

```cmd
python src\training\run_corrected_training.py ^
  --prepared-dawn-root D:\data\DAWN_CORRECTED ^
  --acdc-root D:\data\ACDC\YOLO_ENTIRE ^
  --model D:\models\yolov8l.pt ^
  --output-root D:\runs\corrected ^
  --dry-run
```

Remove `--dry-run` only when the plan is correct. The locked route uses 100 epochs, 640-pixel images, seed 0, deterministic mode, AMP and CUDA device 0. Fog/Rain use batch 4; Entire/Combined use batch 2. The runner resumes an incomplete checkpoint where possible and records the accepted `best.pt` SHA-256 digest.

## 7. Run the seven-cell validation matrix

```cmd
python src\evaluation\run_cross_domain_validation.py ^
  --acdc-checkpoint D:\models\acdc_best.pt ^
  --dawn-checkpoint D:\models\dawn_best.pt ^
  --combined-checkpoint D:\models\combined_best.pt ^
  --acdc-root D:\data\ACDC\YOLO_ENTIRE ^
  --prepared-dawn-root D:\data\DAWN_CORRECTED ^
  --output-root D:\runs\final_matrix ^
  --dry-run
```

The runner validates:

1. ACDC on ACDC;
2. ACDC on DAWN;
3. DAWN on ACDC;
4. DAWN on DAWN;
5. Combined on ACDC;
6. Combined on DAWN;
7. Combined on the union.

Remove `--dry-run` after verifying the checkpoint hashes and roots. Do not compare results from independently selected epochs or different evaluation settings.

## 8. Run CARLA diagnostics

Install and start the complete CARLA 0.9.16 simulator separately. The pip package is only the client API. A resource-conservative Windows launch example is:

```cmd
CarlaUE4.exe -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=320 -ResY=180 -nosound -d3d11
```

Use the approved condition scripts listed in [CARLA reproducibility](CARLA_REPRODUCIBILITY.md). Configure model, output and dataset paths through the documented arguments/environment variables before running them.

CARLA outputs are qualitative diagnostics. Clear, Rain, Fog and synthetic Night were not produced under one common labelled real-data evaluation protocol and must not be ranked numerically against ACDC or DAWN.

## 9. Know which files are authoritative

- Cross-domain conclusion: `results/CORRECTED_2026-08-27/final_cross_domain_validation_matrix.csv`
- Corrected run outcomes: `results/CORRECTED_2026-08-27/corrected_training_metrics.csv`
- Checkpoint and row lineage: `docs/CORRECTED_EVIDENCE_MANIFEST.csv`
- Correction audit: `results/CORRECTED_2026-08-27/label_remap_audit.json`
- Final document hashes: `docs/FINAL_SUBMISSION_MANIFEST.csv`
- Historical `results.csv`: provenance only when DAWN/Combined evidence was later superseded
- CARLA review frames/reports: qualitative or diagnostic only

## 10. Common mistakes to avoid

- Do not call validation scores test accuracy.
- Do not state that Faster R-CNN or RT-DETR was implemented.
- Do not treat the Combined model as universally best.
- Do not infer universal rain robustness from 42 DAWN Rain validation images.
- Do not mix the training-history result with a later locked validation result.
- Do not calculate F1 from independently rounded or independently selected values when the unrounded same-row values exist.
- Do not present synthetic Night as native negative-sun CARLA night.
- Do not edit archival notebook paths while retaining stale outputs and call the notebook reproduced.
