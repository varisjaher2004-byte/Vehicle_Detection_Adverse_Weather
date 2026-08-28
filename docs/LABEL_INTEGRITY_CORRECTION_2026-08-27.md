# DAWN Label-Integrity Correction (27 August 2026)

## Why a correction was required

The archived DAWN training labels already used the project's unified eight-class
mapping, but the archived validation labels retained older DAWN numeric IDs.
Under the eight-class YAML, class ID `8` was out of range. Ultralytics therefore
ignored 46 of the 120 validation images that contained that ID, while the other
old IDs were interpreted under incorrect class names. Earlier DAWN and Combined
validation values derived from that inconsistent label view are retained only as
historical diagnostic provenance and are not dissertation-facing evidence.

## Audited remapping

| Raw DAWN validation ID | Unified project ID | Unified class |
|---:|---:|---|
| 1 | 0 | person |
| 2 | 7 | bicycle |
| 3 | 2 | car |
| 4 | 6 | motorcycle |
| 6 | 4 | bus |
| 8 | 3 | truck |

The corrected labels use the common project order:
`person, rider, car, truck, bus, train, motorcycle, bicycle`.

## Non-destructive procedure

- The source DAWN dataset and archived experiment folders were not edited.
- A corrected local data view was created by copying the images and writing
  remapped validation labels to a new directory.
- Image and label stems were checked for exact one-to-one correspondence.
- Every row was checked for five YOLO fields, a class ID from 0 to 7 and
  normalised box coordinates from 0 to 1.
- Fog validation labels were checked against the available Pascal VOC object
  names for all 78 aligned Fog validation images (710 objects).
- Fog and Rain subsets were verified to be disjoint and to reconstruct the
  120-image DAWN Entire validation split.

## Verified input counts

| Split | Images | Instances |
|---|---:|---:|
| DAWN train | 422 | 2,829 |
| DAWN validation | 120 | 1,029 |
| ACDC train | 1,600 | 9,831 |
| ACDC validation | 406 | 2,731 |

DAWN condition validation sizes were 78 Fog images and 42 Rain images. The high
DAWN Rain in-domain score must therefore be interpreted with its small split and
must not be presented as universal rain robustness.

## Evidence consequence

Four models were trained after the audit: DAWN Fog, DAWN Rain, DAWN Entire and
ACDC+DAWN Combined. A final seven-cell matrix then evaluated the locked ACDC,
corrected DAWN and corrected Combined checkpoints under one validation protocol.
The final values and checkpoint hashes are in
`results/CORRECTED_2026-08-27/`. These are validation-split measurements, not a
newly claimed untouched test set.

