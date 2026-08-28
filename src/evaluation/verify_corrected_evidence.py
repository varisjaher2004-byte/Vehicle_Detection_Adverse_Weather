"""Verify the committed corrected metrics, F1 calculations and hash lineage."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "results" / "CORRECTED_2026-08-27"
MANIFEST = REPOSITORY_ROOT / "docs" / "CORRECTED_EVIDENCE_MANIFEST.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def check_f1(row: dict[str, str], label: str) -> None:
    precision, recall, actual = float(row["precision"]), float(row["recall"]), float(row["f1"])
    expected = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    assert math.isclose(actual, expected, abs_tol=1.5e-6), f"F1 mismatch: {label}"


def main() -> None:
    training = rows(RESULTS / "corrected_training_metrics.csv")
    matrix = rows(RESULTS / "final_cross_domain_validation_matrix.csv")
    manifest = rows(MANIFEST)
    metadata = json.loads((RESULTS / "evidence_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "label_remap_audit.json").read_text(encoding="utf-8"))
    assert len(training) == 4 and len(matrix) == 7 and len(manifest) == 11
    assert audit["source_files_modified"] is False
    assert audit["condition_splits_disjoint_and_complete"] is True
    assert metadata["status"] == "PASS" and metadata["protocol"]["split"] == "val"
    for row in training:
        check_f1(row, row["experiment"])
        assert len(row["best_checkpoint_sha256"]) == 64
    for row in matrix:
        check_f1(row, f"{row['model_training_domain']}->{row['evaluation_domain']}")
        expected = metadata["checkpoint_sha256"]["COMBINED" if row["model_training_domain"] == "ACDC+DAWN" else row["model_training_domain"]]
        assert row["checkpoint_sha256"] == expected
    for row in manifest:
        check_f1(row, row["evidence_id"])
        assert len(row["checkpoint_sha256"]) == 64
    print("CORRECTED EVIDENCE PASS: 4 training runs, 7 validation cells, 11 manifest rows")


if __name__ == "__main__":
    main()
