# YOLO-Based Vehicle Detection Under Adverse Environmental Conditions

[![Repository integrity](https://github.com/varisjaher2004-byte/Vehicle_Detection_Adverse_Weather/actions/workflows/repository-integrity.yml/badge.svg)](https://github.com/varisjaher2004-byte/Vehicle_Detection_Adverse_Weather/actions/workflows/repository-integrity.yml)

MSc Artificial Intelligence research project by **Varis Jahirbhai Kureshi** at Sheffield Hallam University, supervised by **Yasir Javed**.

This repository evaluates how an Ultralytics **YOLOv8l** detector behaves when it is trained and validated across two prepared adverse-weather data domains: **ACDC** and a **corrected DAWN** view. The central result is that strong within-domain performance does not reliably transfer to another dataset domain. Joint ACDC+DAWN training improves the descriptive balance across both recorded domains, but it is not universally best on every metric.

## Start here

| Goal | Open or run |
|---|---|
| Read the complete research | [Final dissertation](docs/submission/Varis_Kureshi_Dissertation_SUBMISSION_READY_FINAL_2026-08-31.docx) |
| View the defence | [Final 20-slide presentation](docs/submission/Varis_Kureshi_Dissertation_Defence_MSC_SUBMISSION_READY_FINAL_2026-09-01.pptx) |
| Understand the project quickly | [Getting started](docs/GETTING_STARTED.md) |
| Inspect the canonical numbers | [Seven-cell validation matrix](results/CORRECTED_2026-08-27/final_cross_domain_validation_matrix.csv) |
| Understand the DAWN correction | [Label-integrity record](docs/LABEL_INTEGRITY_CORRECTION_2026-08-27.md) |
| Check evidence rules | [Evidence-integrity guide](docs/EVIDENCE_INTEGRITY.md) |
| Verify the repository | Run the two verification commands below |
| Audit the final package | [Final repository audit](docs/FINAL_REPOSITORY_AUDIT_2026-09-01.md) |
| Understand formal filenames | [File-naming convention and rename provenance](docs/FILE_NAMING_CONVENTION.md) |

### Fast integrity check

The verification route does not require raw datasets, trained weights, CARLA or Ultralytics:

```bash
python -m pip install PyYAML==6.0.3
python src/evaluation/verify_repository.py
python src/evaluation/verify_submission_package.py
python -m compileall -q src
```

The first verifier checks active YAMLs, historical same-row evidence, training presets, the locked CARLA route and the corrected evidence set. The second checks the final DOCX/PPTX hashes, package integrity, required research content, 20-slide/20-note structure and the embedded CARLA video.

## Research at a glance

| Item | Final study definition |
|---|---|
| Problem | Apparent robustness within one dataset may collapse after transfer to another camera, scene and annotation domain. |
| Implemented detector | Ultralytics YOLOv8l, fine-tuned through transfer learning. |
| Real-data domains | Prepared ACDC, corrected DAWN and their Combined union. |
| Label space | Eight harmonised road-user classes: person, rider, car, truck, bus, train, motorcycle and bicycle. |
| Main evaluation | Seven protocol-matched model-to-validation cells using Precision, Recall, F1, mAP50 and mAP50-95. |
| Simulation | CARLA 0.9.16/Town10HD_Opt qualitative engineering diagnostics for Clear, Rain, Fog and synthetic Night. |
| Main finding | Direct transfer mean mAP50 was 0.1242; Combined training achieved a two-domain mean of 0.5226. |
| Evidence boundary | Validation-only, single-seed, descriptive results; no deployment-safety or population-level guarantee. |
| Contribution | An auditable evaluation protocol: non-destructive label correction, locked checkpoints, same-invocation metrics, bidirectional transfer and explicit construct boundaries. |

## Canonical seven-cell result

All values below are validation-bound. F1 is computed from unrounded Precision and Recall from the same validation invocation.

| Train -> validate | Images | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| ACDC -> ACDC | 406 | 0.6226 | 0.4014 | 0.4881 | 0.4221 | 0.2634 |
| ACDC -> DAWN | 120 | 0.5955 | 0.1190 | 0.1983 | 0.1362 | 0.0797 |
| DAWN -> ACDC | 406 | 0.2055 | 0.1330 | 0.1615 | 0.1122 | 0.0666 |
| DAWN -> DAWN | 120 | 0.8465 | 0.5794 | 0.6879 | 0.6277 | 0.4369 |
| Combined -> ACDC | 406 | 0.5225 | 0.4129 | 0.4613 | 0.4069 | 0.2452 |
| Combined -> DAWN | 120 | 0.8142 | 0.6093 | 0.6970 | 0.6382 | 0.4228 |
| Combined -> Union | 526 | 0.5808 | 0.4376 | 0.4991 | 0.4472 | 0.2720 |

The defensible interpretation is:

- direct single-domain transfer was weak in both directions: ACDC-to-DAWN mAP50 0.1362 and DAWN-to-ACDC 0.1122;
- low transfer Recall (0.1190 and 0.1330) shows that missed annotated objects were the clearest aggregate failure symptom;
- Combined training retained ACDC mAP50 0.4069 and reached DAWN mAP50 0.6382, producing the strongest two-domain balance;
- Combined was not universally dominant: it was 0.0152 below the ACDC specialist on ACDC mAP50, and its DAWN mAP50-95 was 0.0141 below the DAWN specialist;
- DAWN Rain mAP50 0.9916 is a small-split, car-dominated in-domain estimate from only 42 validation images, not evidence of universal rain robustness.

The CSV is authoritative for machine-readable values. The dissertation and presentation use the same rounded display values.

## Data used by the final protocol

| Prepared source | Training | Validation | Role |
|---|---:|---:|---|
| ACDC | 1,600 images / 9,831 instances | 406 images / 2,731 instances | Real adverse-condition domain |
| Corrected DAWN | 422 images / 2,829 instances | 120 images / 1,029 instances | Independent real-traffic adverse-weather domain |
| Combined | 2,022 images | 526-image union | Joint exposure and constituent-domain evaluation |

The upstream releases and the prepared local artefacts are different objects. ACDC is officially released for semantic/panoptic scene understanding; this study uses a prepared YOLO bounding-box artefact. DAWN is described upstream as a 1,000-image collection; the locked local protocol uses the verified 422/120 split shown above. The repository does not invent missing conversion or inclusion history.

Combined training is not domain-balanced: ACDC contributes 79.1% of its training images and 77.2% of union validation images. For this reason, the Combined-to-ACDC and Combined-to-DAWN rows are reported separately instead of relying only on the union score.

## What was implemented

- active repository-relative YAML configurations for nine historical real-data experiments;
- one portable preset-driven YOLOv8l training entry point;
- a non-destructive DAWN validation-label correction pipeline;
- a locked four-run corrected training pipeline for DAWN Fog, Rain, Entire and Combined;
- a common seven-cell cross-domain validation runner;
- same-invocation Precision, Recall, F1, mAP50 and mAP50-95 recording;
- SHA-256 checkpoint lineage, manifests and automated evidence verification;
- CARLA Clear, Rain, Fog and synthetic-Night capture/diagnostic scripts;
- compact historical and corrected result evidence;
- final dissertation and 20-slide defence presentation with an embedded one-minute CARLA showreel.

The following were **not** implemented and must not be claimed as experimental results: Faster R-CNN or RT-DETR baselines, a new YOLO architecture, multiple training seeds, an untouched scene-grouped test set, confidence intervals, a complete class-by-weather error ledger, calibrated sim-to-real evaluation or autonomous-vehicle deployment validation.

## The DAWN label-integrity correction

The archived DAWN training labels already used the unified zero-to-seven class IDs, but validation labels retained an older numeric scheme. The audited mapping was:

| Archived ID | Unified ID | Class |
|---:|---:|---|
| 1 | 0 | person |
| 2 | 7 | bicycle |
| 3 | 2 | car |
| 4 | 6 | motorcycle |
| 6 | 4 | bus |
| 8 | 3 | truck |

ID 8 was outside the configured 0-7 range, affecting 46 of 120 validation images, while other retained IDs could silently receive the wrong class meaning. The correction pipeline copies the source data, remaps only the copied validation labels, checks five-field YOLO rows, class range, normalised box bounds and image/label pairing, and leaves the original source untouched. All 78 aligned Fog validation images and 710 objects were also checked against retained Pascal VOC class names.

Earlier DAWN and Combined outputs are preserved as historical provenance but are superseded for dissertation claims by `results/CORRECTED_2026-08-27/`.

## Repository structure

```text
Vehicle_Detection_Adverse_Weather/
|-- configs/                  # Active relative dataset YAMLs, presets and locked CARLA route
|-- docs/
|   |-- submission/           # Final dissertation and final 20-slide defence deck
|   |-- GETTING_STARTED.md
|   |-- EVIDENCE_INTEGRITY.md
|   |-- LABEL_INTEGRITY_CORRECTION_2026-08-27.md
|   |-- CARLA_REPRODUCIBILITY.md
|   `-- evidence/audit manifests
|-- notebooks/                # Archival development notebooks; see NOTEBOOK_PROVENANCE.md
|-- results/
|   |-- CORRECTED_2026-08-27/ # Canonical corrected training and seven-cell evidence
|   |-- ACDC, DAWN, COMBINED/ # Historical compact experiment records
|   `-- CARLA/                # Qualitative/diagnostic reports and review frames
|-- src/
|   |-- training/             # Portable historical and corrected training runners
|   |-- evaluation/           # Correction, validation and verification tools
|   `-- carla/                # Approved simulator scripts by condition
|-- .github/workflows/        # Automated repository verification
|-- CITATION.cff
|-- requirements.txt
`-- README.md
```

## Reproduce or inspect the workflow

Detailed installation, dataset preparation, dry-run and execution instructions are in [Getting started](docs/GETTING_STARTED.md). The shortest portable inspection route is:

```bash
python src/training/train_experiment.py --list
python src/training/train_experiment.py --experiment ACDC_FOG --dry-run
```

Corrected data preparation, training and cross-domain validation use explicit source, model and output paths. They never assume that datasets or weights are stored in Git.

## Evidence hierarchy

Use repository artefacts in this order:

1. `results/CORRECTED_2026-08-27/final_cross_domain_validation_matrix.csv` for cross-domain comparison;
2. `results/CORRECTED_2026-08-27/corrected_training_metrics.csv` for the four corrected run outcomes;
3. `docs/CORRECTED_EVIDENCE_MANIFEST.csv` and checkpoint hashes for lineage;
4. historical `results.csv` files only for preserved same-row provenance;
5. real-world annotated predictions and CARLA images only as
   qualitative/diagnostic evidence.

Do not combine independent maxima from different epochs, replace the canonical matrix with older convenience summaries, or compare CARLA numbers directly with ACDC/DAWN validation metrics.

## CARLA boundary

The approved simulator evidence uses CARLA 0.9.16, `Town10HD_Opt`, spawn 36 on road 11/lane 1, a 640x360 RGB camera at 10 sensor FPS and a synchronous world at 20 ticks per second.

Clear is raw tracking evidence. Rain, Fog and Night use stable or actor-aided presentation procedures. The Night appearance is a calibrated CPU day-for-night transformation of genuine CARLA RGB frames, not native physical night rendering. These outputs demonstrate that the pipeline can be exercised under controlled visual conditions; they do not prove real-world robustness or provide a numerical weather ranking.

See [CARLA reproducibility](docs/CARLA_REPRODUCIBILITY.md) for the locked route, environment variables, scripts and failure-recovery boundary.

## Reproducibility and exclusions

The verified environment is Windows, Python 3.12.10, Ultralytics 8.4.104, PyTorch 2.12.1+cu126, Torchvision 0.27.1+cu126, CUDA 12.6 and an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM.

Raw datasets, ACDC-background annotated predictions, model weights, generated
videos outside the embedded presentation asset, local caches and
failed/superseded runs are intentionally excluded because of licensing, size,
privacy and evidence-governance concerns. Licence-compatible DAWN annotated
predictions, all seven saved inference logs and a SHA-256 inventory of the full
saved output collection are documented in [Real-world inference evidence](results/REAL_WORLD_INFERENCE_EVIDENCE.md).
Exact reruns require separately obtained datasets and weights plus compatible
software and hardware. The repository supports method reconstruction and evidence
audit; it does not promise bit-for-bit numerical identity on another system.

The notebooks preserve original development history, including 118 machine-specific Windows-path occurrences. They are archival records, not the supported portable route. See [Notebook provenance](docs/NOTEBOOK_PROVENANCE.md).

## Ethics and public-release boundary

The implemented study uses public secondary datasets and CARLA and recruited no human participants. The dissertation's Blackboard submission remains subject to the university's full ethics-approval requirement. Signed UREC1 and publication forms may contain signatures or institutional metadata and should not be committed publicly without explicit approval and a privacy review. The public repository therefore focuses on research content, code, evidence and reproducibility.

## Citation and reuse

Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff). Cite the exact commit or release tag used, and cite ACDC, DAWN, CARLA, YOLO/Ultralytics and any other third-party resources through their original publications or provider guidance.

Third-party datasets and software retain their own licences. No separate licence for the repository's original code is granted unless a `LICENSE` file is added; contact the repository owner before reuse beyond academic inspection.
