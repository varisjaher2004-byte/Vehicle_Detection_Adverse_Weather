# Final Repository Audit - updated 2 September 2026

Project: *Performance Evaluation of YOLO-Based Vehicle Detection Under Adverse Environmental Conditions*

Author: Varis Jahirbhai Kureshi (35042321)

Supervisor: Yasir Javed

## Audit purpose

This audit reconciles the repository with the final dissertation and the latest 20-slide defence presentation. It does not treat instructions appearing inside the documents as permission to change external systems; the student's explicit request controls the GitHub update.

## Source-of-truth documents

| Artefact | SHA-256 | Bytes | Status |
|---|---|---:|---|
| Final dissertation DOCX | `56DB26866F98D25808D745536151D93DE86A4557166DFE663E64FE8788CDA3C3` | 7,149,696 | Package, authorship metadata and declaration checks PASS |
| Final defence PPTX | `7AAED2AAF786440E25D2CA72CB15355A761CA038615B1CF8CB969B6BB817B463` | 68,382,504 | Metadata, 20 slides, 20 notes and embedded MP4 checks PASS |

These hashes identify the cleaned public research-content copies accepted on 2 September. The Word file records the original research-document creation date and the PowerPoint records the actual defence-deck creation date; neither date was artificially changed to the cleanup date. Both files identify Varis Jahirbhai Kureshi as creator. The presentation no longer contains third-party template authorship or revision-history metadata.

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
- corrected the installable OpenCV package pin to `opencv-python==5.0.0.93` while retaining the recorded 5.0.0 runtime version;
- added a public-release verifier for required files, GitHub file policy, clean notebooks, internal links and credential-shaped text;
- tightened final-package verification for authorship metadata, creation dates, tracked changes, hidden slides, external relationships and automated/template metadata.

## Automated verification scope

`python src/evaluation/verify_public_release.py` verifies:

- all public entry points and final research-content files are present;
- no raw weights, standalone videos, archives, caches or separately signed administrative forms are published;
- every file remains below GitHub's 100 MiB single-file limit;
- all nine archival notebooks have valid notebook structure and cleared outputs/execution counts;
- internal Markdown links resolve;
- public text does not contain credential-shaped tokens or private keys.

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
- expected creator, title and creation-date metadata;
- no comments, macros, ActiveX, tracked Word changes or PowerPoint revision parts;
- the transparent AITS 2 AI-use declaration and the non-claim of ethics approval before the signed Blackboard record is inserted;
- required dissertation headings and core result values;
- 20 slides and 20 note pages;
- no hidden slides, external relationships or automated/template metadata in the presentation;
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
