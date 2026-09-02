# Results Evidence Guide

## Dissertation-facing result set

The final quantitative evidence is `CORRECTED_2026-08-27/`. It was produced
after a non-destructive audit found that archived DAWN validation labels retained
older numeric class IDs. Use the corrected training CSV and the final seven-cell
cross-domain matrix for dissertation claims. The remap and verification boundary
are documented in `docs/LABEL_INTEGRITY_CORRECTION_2026-08-27.md`.

The older DAWN and Combined outputs in this directory are preserved as historical
provenance. They must not be reported as final performance and must not be used to
support the superseded claims that DAWN was substantially weaker, Combined was
merely intermediate, or additional heterogeneous data failed to aid
generalisation.

The original files below this directory remain compact artefacts from completed experiments. Their historical same-checkpoint verification route is:

1. the relevant committed `results.csv`;
2. the row with maximum `metrics/mAP50(B)`;
3. Precision, Recall, mAP50, and mAP50-95 from that same row;
4. F1 derived from the same row's Precision and Recall;
5. the historical summary in `docs/EVIDENCE_MANIFEST.csv`.

Run `python src/evaluation/verify_repository.py` from the repository root to verify both this provenance chain and the corrected final evidence.

## Training visual diagnostics

Compact standard training curves and result plots are retained alongside their corresponding experiment CSVs. The final D-drive coverage audit added the previously omitted ACDC Fog and DAWN Entire Precision/Recall curves, with exact SHA-256 checks in the repository verifier. The old DAWN Rain `results.png` was not added because it depicts the superseded low-performing July run rather than the corrected final result. Training/validation batch mosaics remain excluded because they are redundant debugging previews and may reproduce restricted source-dataset imagery.

## Real-world qualitative inference evidence

The repository now includes 240 licence-compatible annotated DAWN predictions:

- 120 from the DAWN Entire checkpoint in `DAWN/ENTIRE/qualitative_detections/`;
- 120 from the Combined checkpoint evaluated on the DAWN portion of the mixed
  validation union in `COMBINED/qualitative_detections/DAWN/`.

Seven row-level inference logs are filed with their corresponding experiment.
The complete 1,466-record saved-output inventory, SHA-256 integrity data,
historical folder-name correction and dataset redistribution boundary are in
[`REAL_WORLD_INFERENCE_EVIDENCE.md`](REAL_WORLD_INFERENCE_EVIDENCE.md). These
outputs are qualitative execution evidence; they do not replace the corrected
quantitative evidence above.

## Legacy and diagnostic summaries

Some older summary artefacts were created for exploratory analysis and must not override the canonical manifest.

- `ACDC/RAIN/training_summary.txt` describes the mAP values around raw epoch index 73, but `results.csv` reaches a higher mAP50 at raw epoch index 63. Its Precision/Recall lines also do not define the canonical same-row result. Treat this file as a legacy summary.
- Several `performance_summary.csv` files contain the independent maximum of each metric column. Independent maxima can come from different epochs and must not be combined into a same-checkpoint result.
- Training-information text files document run provenance but are not a substitute for row-level metric verification.

Examples of independent maxima occurring at different raw epoch indices are:

| Experiment | Maximum-mAP50 epoch index | Same-row mAP50-95 | Independent maximum mAP50-95 epoch index | Independent maximum mAP50-95 |
|---|---:|---:|---:|---:|
| ACDC Entire | 87 | 0.25579 | 97 | 0.26138 |
| ACDC Night | 82 | 0.15061 | 89 | 0.15885 |
| ACDC Snow | 57 | 0.25762 | 44 | 0.27001 |
| Combined | 95 | 0.15988 | 79 | 0.17115 |

These files remain in Git to preserve provenance. Their continued presence is not permission to mix independently maximised metrics in dissertation tables.

## CARLA boundary

CARLA files are separate controlled-simulation artefacts. They do not share an identical labelled evaluation protocol with the real-world training CSVs. Stable, ground-truth-aided, or actor-association outputs must not be represented as standard Ultralytics validation metrics unless the corresponding file explicitly establishes that protocol.
