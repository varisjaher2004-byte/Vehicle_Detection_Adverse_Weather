# Evidence Integrity and Metric Selection

## Final dissertation-facing evidence

The final quantitative result set is under `results/CORRECTED_2026-08-27/` and
is indexed by `docs/CORRECTED_EVIDENCE_MANIFEST.csv`. It contains four corrected
training outcomes and a seven-cell protocol-matched validation matrix. Each row
uses Precision and Recall from one Ultralytics validation result and derives F1
only from that pair. Checkpoint SHA-256 hashes preserve model identity.

The DAWN correction was non-destructive. Archived validation labels retained old
numeric IDs, including class 8, which caused 46 of 120 images to be excluded by
the eight-class YAML and caused other IDs to be interpreted under incorrect class
names. The audited mapping and split counts are documented in
`LABEL_INTEGRITY_CORRECTION_2026-08-27.md`.

Final findings are limited to validation splits. The cross-domain matrix directly
tests transfer, but it is not relabelled as an untouched test set. DAWN Rain's
near-saturated in-domain value uses only 42 validation images and is not evidence
of universal rain robustness.

This document defines how the compact real-world evidence manifest is derived and how the simulation artefacts may be interpreted.

## Historical real-world result set

The original manifest contains nine completed experiments:

- ACDC Entire, Fog, Night, Rain, and Snow;
- DAWN Entire, Fog, and Rain;
- Combined Entire.

Runs affected by missing labels, incorrect paths, incomplete execution, or diagnostic-only objectives were excluded at that stage. After the DAWN validation-ID issue was discovered, the original DAWN and Combined metrics were superseded for dissertation reporting while remaining available for provenance.

## Selection rule

For each approved `results.csv`, the selected row is the row with the maximum value of the Ultralytics column `metrics/mAP50(B)`.

The following values are copied from that same row:

- `metrics/precision(B)`;
- `metrics/recall(B)`;
- `metrics/mAP50(B)`;
- `metrics/mAP50-95(B)`.

F1 is derived from the selected row's Precision and Recall:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

The `epoch_index` field in the manifest preserves the raw `epoch` value stored by Ultralytics. It is an index from the source CSV and should not be silently relabelled as a human-counted epoch number.

The manifest does not claim that the omitted model weights can be reconstructed from metrics alone. It also does not independently prove the dataset split, annotation quality, or training seed.

## Superseded and diagnostic summaries

Within the historical evidence layer, the original manifest takes precedence over older convenience summaries:

- `results/ACDC/RAIN/training_summary.txt` reports values associated with raw epoch index 73, whereas `results.csv` reaches a higher mAP50 at raw epoch index 63. The text summary is retained as legacy provenance, not as the final same-checkpoint result.
- `performance_summary.csv` files may contain the independent maximum of each metric column. For ACDC Entire, ACDC Night, ACDC Snow, and Combined, the maximum mAP50 and maximum mAP50-95 occur at different raw epoch indices.
- independently maximised Precision, Recall, mAP50, or mAP50-95 values must not be combined into one synthetic checkpoint row.

See `results/README.md` for the verified examples and interpretation guidance.

## Simulation boundary

CARLA artefacts are retained separately because they are not an identical labelled validation benchmark to the real-world experiments. They must not be inserted into the real-world metric table as if they shared the same evaluation protocol.

In particular:

- stable or ground-truth-aided presentation outputs are not raw YOLO recall measurements;
- frame-level diagnostic associations are not standard Ultralytics Precision/Recall/F1 values;
- the Night appearance is a calibrated CPU day-for-night transformation of genuine CARLA RGB frames, not native negative-sun physical night rendering;
- Clear, Rain, Fog, and Night visuals support controlled qualitative evaluation, not a direct numerical weather ranking unless a common labelled protocol is established.

## Automated verification

From the repository root, run:

```cmd
python src\evaluation\verify_repository.py
```

The verifier checks the nine active historical configurations, presets and source CSV rows, then independently checks the corrected four-run summary, seven-cell matrix, same-row F1 calculations, checkpoint-hash lineage and label-remap audit. It does not require excluded raw datasets or model weights.
