# Final D-Drive Research Coverage Audit - 2 September 2026

Project: *Performance Evaluation of YOLO-Based Vehicle Detection Under Adverse Environmental Conditions*

Source reviewed: `D:\DMSc_Dissertation_SUBMISSION_READY\01_DISSERTATION`

## Purpose and decision rule

This is the final coverage check between the local dissertation working archive and the public GitHub repository. The objective is not to upload every local byte. It is to preserve the code, configurations, compact evidence, documentation and submission artefacts required to understand and audit the final study, while respecting dataset licences, repository size and the final evidence boundary.

The local archive contained approximately 91,885 files (about 71 GB) across ACDC, DAWN, `Simulation_Project` and `YOLO_COMBINED`. Most of that volume is raw or prepared dataset imagery, a full CARLA installation, generated simulator runs, model weights and caches rather than original source code or compact research evidence.

## Coverage conclusion

No required, publishable research component remains missing after this audit. The repository contains:

- the final dissertation and final 20-slide defence presentation;
- all nine development notebooks, with their archival status documented;
- active portable dataset configurations and training presets;
- historical and corrected training/evaluation code;
- the canonical corrected four-run training record and seven-cell cross-domain matrix;
- compact historical training CSVs and standard result plots;
- approved CARLA qualitative scripts, locked-route metadata, reports and selected review frames;
- 240 licence-compatible DAWN annotated predictions, all seven inference CSV logs, and a 1,466-record SHA-256 inventory of the complete saved inference-output collection;
- automated repository and submission-package verification.

Four compact source plots found missing during this final comparison were added with exact integrity checks:

| Local source area | Repository artefact |
|---|---|
| ACDC Fog training | `results/ACDC/FOG/BoxP_curve.png` |
| ACDC Fog training | `results/ACDC/FOG/BoxR_curve.png` |
| DAWN Entire training | `results/DAWN/ENTIRE/BoxP_curve.png` |
| DAWN Entire training | `results/DAWN/ENTIRE/BoxR_curve.png` |

The local DAWN Rain `results.png` was visually and numerically checked before publication. It represents the superseded July run already preserved by its historical `results.csv` (maximum mAP50 only about 0.06), not the corrected final DAWN Rain result of 0.9916. The graph was therefore deliberately excluded to avoid presenting it as current evidence.

## Saved real-world outputs shown in the screenshot

The nine local output folders contained 1,459 annotated JPG files plus seven CSV logs. Their public-release status is:

| Evidence class | Count | GitHub status |
|---|---:|---|
| DAWN annotated predictions | 240 | Included in the appropriate DAWN/Combined qualitative folders |
| ACDC-background annotated predictions | 1,219 | Withheld under the ACDC redistribution boundary |
| Inference CSV logs | 7 | Included with the corresponding experiments |
| Complete source-artifact inventory | 1,466 rows | Included with paths, classification and SHA-256 hashes |

Two historically named DAWN weather-test folders were verified as duplicate ACDC-background outputs created by an inference notebook configured with ACDC inputs. They are classified by actual provenance rather than their old folder names. See `results/REAL_WORLD_INFERENCE_EVIDENCE.md` for the full reconciliation.

## Intentionally excluded local material

| Local material | Reason it is not required in public GitHub |
|---|---|
| Raw/prepared ACDC imagery and ACDC-background predictions | The ACDC terms restrict redistribution of the dataset and modified versions retaining its images. |
| Raw/prepared DAWN dataset copies | Third-party data should be obtained from its official release; duplicating gigabytes is unnecessary for code/evidence audit. |
| Trained `best.pt`/`last.pt` weights | Large binary artefacts; exact retained checkpoint lineage is recorded by SHA-256 where used by the corrected protocol. |
| Full CARLA installation, maps and vendor PythonAPI | Third-party software and assets, not original project source; obtain from CARLA. |
| Large generated CARLA runs | Redundant intermediate material; the approved qualitative scripts, reports and review frames are retained. |
| Archived/experimental CARLA pipelines | Superseded engineering attempts or pipelines outside the final dissertation evidence boundary. The final study treats CARLA as qualitative diagnostic evidence, not a protocol-matched quantitative benchmark. |
| Training/validation batch mosaics | Debugging previews that duplicate dataset imagery and may inherit dataset redistribution restrictions. |
| July exploratory result package, including the old DAWN Rain `results.png` | Superseded by the corrected 27 August quantitative evidence and capable of reintroducing obsolete claims. Compact historical CSV provenance remains available. |
| Caches, temporary files and duplicate exports | Machine-specific or reproducible working material with no independent evidential value. |
| Signed UREC/publication administration | Blackboard-controlled and potentially private; it is not public-repository research evidence. |

## Licensing and reproducibility boundary

- ACDC remains governed by the [official ACDC licence](https://acdc.vision.ee.ethz.ch/license).
- DAWN should be obtained from its [official Mendeley Data release](https://data.mendeley.com/datasets/766ygrbt8y/3).
- CARLA should be obtained from the [official CARLA project](https://carla.org/).

The public repository supports method reconstruction, evidence inspection and submission consistency. Exact retraining still requires separately obtained datasets, weights, a compatible environment and suitable hardware. Excluding restricted or very large third-party material is therefore a deliberate reproducibility boundary, not missing research work.
