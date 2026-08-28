# Corrected Training and Cross-Domain Evidence

This directory is the dissertation-facing quantitative result set completed on
27 August 2026 after the DAWN validation-label integrity correction.

Files:

- `corrected_training_metrics.csv`: final validation metrics and SHA-256 hashes
  for the four corrected training runs;
- `final_cross_domain_validation_matrix.csv`: seven protocol-matched within-
  domain, cross-domain and Combined evaluations;
- `evidence_metadata.json`: locked software, split, checkpoint and interpretation
  metadata;
- `label_remap_audit.json`: sanitised counts and the non-destructive DAWN mapping.

The values use Ultralytics validation splits. They are not results from a newly
claimed untouched test set. Precision and Recall in each row come from the same
validation result and F1 is calculated only from that pair. Checkpoint binaries,
raw datasets and generated plots are intentionally excluded from Git.

Earlier DAWN and Combined artefacts elsewhere under `results/` are retained as
historical provenance but are superseded for dissertation reporting because they
used the inconsistent DAWN validation-label view.

