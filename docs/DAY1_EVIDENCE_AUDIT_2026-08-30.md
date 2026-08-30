# Day-1 Evidence Audit and Lock

Project: *Performance Evaluation of YOLO-based Vehicle Detection under Adverse Environmental Conditions*
Audit date: 30 August 2026
Evidence base commit: `1eb81b595c6b7f8b38851eab5ca1978a5d12c8a5`

## Decision

Day-1 established a hash-locked, dissertation-facing real-world evidence layer.
The corrected result tables are internally consistent and tied to five accepted
YOLOv8l checkpoints by SHA-256. Checkpoint binaries and raw datasets remain
outside Git; this repository stores their identities, protocol, metrics and
audit trail.

The final quantitative claims remain validation-bound estimates. They are not
described as untouched hold-out test performance, test accuracy or generalized
real-world performance bounds.

## Locked protocol

- Model family: Ultralytics YOLOv8l.
- Training: 100 epochs, image size 640, seed 0, deterministic mode enabled.
- Training batches: 4 for DAWN Fog and DAWN Rain; 2 for DAWN Entire and Combined.
- Final seven-cell evaluation batch: 4.
- Checkpoint selection: maximum validation mAP@0.5, recorded as `best.pt`.
- F1: calculated from the unrounded Precision and Recall in the same row.
- Analysis boundary: descriptive, single-seed and non-causal; no confidence
  intervals are estimated.

## Dataset support

| Validation domain | Images | Instances |
|---|---:|---:|
| ACDC | 406 | 2,731 |
| Corrected DAWN | 120 | 1,029 |
| ACDC + corrected DAWN union | 526 | 3,760 |

DAWN contains 78 Fog validation images (710 instances) and 42 Rain validation
images (319 instances). The corrected Rain support is concentrated in car (266),
truck (46), bus (5) and person (2), with no validation instances for several
other project classes. Its mAP@0.5 of 0.991635 is therefore reported as a
near-saturated estimate on a small, low-diversity validation subset, not as
universal rain robustness.

## Metric reconciliation

All four corrected-training rows and all seven cells in the final cross-domain
matrix passed independent same-row F1 recomputation. The authoritative ACDC to
DAWN transfer row uses Precision 0.595542 and Recall 0.118977:

```text
F1 = 2 × 0.595542 × 0.118977 / (0.595542 + 0.118977)
   = 0.198331466...
```

The correct four-decimal presentation is therefore **0.1983**. Computing from
already rounded Precision and Recall and reporting 0.1984 would violate the
locked unrounded-row rule.

Cross-domain degradation was primarily recall-led for ACDC to DAWN: recall fell
from 0.401423 to 0.118977 (70.4% relative reduction), while precision decreased
by 4.3%. DAWN to ACDC transfer was weaker in both precision and recall, with
mAP@0.5 falling from 0.627660 to 0.112241. These results describe dataset shift;
they do not establish causal superiority of an architecture or training domain.

Combined training produced a balance rather than uniform dominance. Relative to
the ACDC specialist, Combined-on-ACDC mAP@0.5 was 0.015150 lower. Relative to the
DAWN specialist, Combined-on-DAWN mAP@0.5 was 0.010559 higher while
mAP@0.5:0.95 was 0.014061 lower. Comparisons must use the seven-cell matrix and
must not mix its rows with separately evaluated training-summary rows.

## Label-integrity correction

Archived DAWN validation labels retained old numeric IDs. The audited,
non-destructive corrected view remapped those IDs to the shared eight-class
schema. The original source data was not modified. Fog and Rain validation
subsets were verified as disjoint and complete with respect to DAWN Entire.

Earlier DAWN and Combined outputs are retained only as historical provenance.
They are superseded for dissertation reporting by
`results/CORRECTED_2026-08-27/` and `docs/CORRECTED_EVIDENCE_MANIFEST.csv`.

## Visual-evidence boundary

Three matched ACDC Fog validation batches were located in a historical,
condition-specific run. One traceable failure scene shows a ground-truth bus
predicted as train at confidence 0.6 while nearby car and truck detections remain
partly successful. This supports qualitative error discussion and rare-class
caution, but it is not matched evidence for a cell in the corrected seven-cell
matrix. The source batches remain outside Git and are registered by hash in
`DAY1_VISUAL_EVIDENCE_REGISTER.csv`.

Selected qualitative illustrations are deliberately informative examples.
They are not representative error-rate estimates and do not provide matched
visual evidence for every quantitative cell.

## Simulation boundary

CARLA Clear, Rain, Fog and synthetic Night outputs remain qualitative diagnostic
and presentation assets. They are not inserted into the seven-cell real-world
Ultralytics matrix and are not numerically compared as if they shared the same
labelled validation protocol.

## Submission-state warning

The pre-existing `SUBMISSION_READY` tree contained superseded DAWN/Combined
artefacts and was not modified during Day-1. It must not be submitted as-is.
Final submission assembly will occur only after dissertation editing, evidence
selection and an independent manifest audit.

Signed UREC1 evidence and the institutional publication form remain external
student/administrative responsibilities and are not represented as complete by
this technical audit.
