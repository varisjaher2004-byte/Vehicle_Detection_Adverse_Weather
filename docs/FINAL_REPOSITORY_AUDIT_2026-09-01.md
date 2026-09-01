# Final Repository Audit - 1 September 2026

Project: *Performance Evaluation of YOLO-Based Vehicle Detection Under Adverse Environmental Conditions*

Author: Varis Jahirbhai Kureshi (35042321)

Supervisor: Yasir Javed

## Audit purpose

This audit reconciles the repository with the final dissertation and the latest 20-slide defence presentation. It does not treat instructions appearing inside the documents as permission to change external systems; the student's explicit request controls the GitHub update.

## Source-of-truth documents

| Artefact | SHA-256 | Bytes | Status |
|---|---|---:|---|
| Final dissertation DOCX | `ADF7544424D9A70FFAAF4A57A1772A9397E056ED6CCE6F47AD52D07B125A9282` | 7,156,098 | Package integrity PASS |
| Final defence PPTX | `359B9B7B0266B4DB79D3384A8F59E79663C8BCD6B9B11F8B6B1DA5D7F658F08E` | 68,383,940 | 20 slides, 20 notes and embedded MP4 present |

The dissertation matches the final 31 August workspace copy byte-for-byte. The Desktop presentation was the latest saved copy and is therefore the accepted presentation artefact for this update.

## Research consistency checks

| Topic | Final aligned statement |
|---|---|
| Detector | YOLOv8l implemented; Faster R-CNN and RT-DETR are literature context only |
| Domains | Prepared ACDC, corrected DAWN and Combined |
| Class space | Eight harmonised road-user classes in IDs 0-7 |
| Training | 100 epochs, 640 px, seed 0; batch/workers vary only as documented |
| Core result | Direct-transfer mean mAP50 0.1242; Combined two-domain mean 0.5226 |
| Combined claim | Improved cross-domain balance, not universal metric dominance |
| DAWN Rain | mAP50 0.9916 on a 42-image, car-dominated in-domain validation subset |
| CARLA | Qualitative controlled diagnostic evidence, not a real-world or directly comparable quantitative benchmark |
| Statistical scope | Validation-bound, single-seed, descriptive; no confidence interval or significance claim |

## Repository changes represented by this audit

- replaced the older Day-2 dissertation binary with the final dissertation;
- replaced the older 13-slide Day-3 binary with the final 20-slide deck containing the one-minute CARLA showreel;
- added a final submission manifest with exact hashes and sizes;
- added an automated final-package verifier;
- reorganised the public entry point around reader, verifier and reproducer routes;
- documented implemented and non-implemented scope explicitly;
- made the corrected seven-cell matrix the obvious canonical result;
- retained historical notebooks and result files as provenance without promoting superseded DAWN/Combined claims;
- kept raw datasets, trained weights, large generated runs and private administrative forms outside Git.

## Automated verification scope

`python src/evaluation/verify_repository.py` verifies:

- nine active dataset configurations;
- nine historical same-row result sources;
- nine training presets;
- one locked CARLA route;
- four corrected training records;
- seven protocol-matched validation cells;
- eleven corrected manifest rows and checkpoint lineage.

`python src/evaluation/verify_submission_package.py` verifies:

- final DOCX/PPTX hashes and byte sizes;
- Office ZIP package integrity;
- required dissertation headings and core result values;
- 20 slides and 20 note pages;
- a `[Sources]` block in every speaker-note page;
- one embedded MP4;
- presence of the canonical matrix values and absence of superseded F1 display `0.1984`.

The workflow also compiles all Python files.

## Known and intentionally retained boundaries

- Raw data and model weights are excluded and must be obtained separately.
- The prepared ACDC panoptic/semantic-to-box conversion lineage is incomplete.
- No exhaustive scene-ID/perceptual near-duplicate ledger is retained for every split.
- Only one training seed is part of the final corrected protocol.
- No untouched scene-grouped test set or protocol-matched architecture baseline is claimed.
- The notebooks preserve 118 original machine-specific path occurrences and are archival.
- Signed ethics/publication forms are Blackboard administration items and are not public repository artefacts.

These limitations narrow the conclusions; they do not invalidate the recorded, validation-bound comparison.
