# Evidence Integrity and Metric Selection

This document defines how the compact real-world evidence manifest is derived and how the simulation artefacts may be interpreted.

## Approved real-world result set

The dissertation-facing real-world result set contains nine completed experiments:

- ACDC Entire, Fog, Night, Rain, and Snow;
- DAWN Entire, Fog, and Rain;
- Combined Entire.

Runs affected by missing labels, incorrect paths, incomplete execution, or diagnostic-only objectives are excluded.

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

The canonical manifest takes precedence over older convenience summaries:

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

The verifier checks the nine active dataset configurations, approved training presets, manifest membership, maximum-mAP50 selection rule, same-row metrics, and F1 calculation. It does not require the excluded raw datasets or model weights.
