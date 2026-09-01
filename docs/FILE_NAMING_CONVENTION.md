# File-Naming Convention and Rename Provenance

## Purpose

Repository-facing filenames are descriptive, concise and readable without relying on an undocumented experiment sequence. This naming pass changed paths only; it did not remove evidence, alter research metrics or rewrite the final dissertation and defence packages.

## Convention

- Python scripts and notebooks use lowercase `snake_case`.
- Training notebooks end in `_training.ipynb`.
- Capture scripts start with `generate_`; detection and diagnostic scripts start with `run_`.
- CARLA evidence filenames identify the condition, evidence role and frame number where applicable.
- Words such as `perfect`, repeated `final` markers and opaque development step numbers are not used in current repository-facing filenames.
- Established experiment IDs such as `ACDC_FOG` and condition folders such as `ACDC/FOG` remain unchanged because they are stable keys in manifests and training presets.

## Protected names

The following were intentionally not renamed:

- final DOCX/PPTX submission filenames, because their exact names, hashes and sizes are locked in `FINAL_SUBMISSION_MANIFEST.csv`;
- canonical corrected result files, because their names already state their role and are cited by the report, presentation and README;
- historical `results.csv`, `args.yaml` and run-snapshot filenames inside condition-specific directories, because their directory context is unambiguous and they preserve experiment provenance;
- experiment and evidence IDs inside CSV/JSON content, because they are research identifiers rather than presentation filenames.

## Historical report fields

Some committed CARLA JSON reports contain absolute source-workstation paths and the filenames originally emitted during the historical run. Those values remain unchanged as provenance. The current GitHub filename is the descriptive path in the repository tree; an old path inside a report is not an instruction or an active repository path.

The complete old-to-new mapping is recorded in [`FILE_RENAME_MANIFEST_2026-09-01.csv`](FILE_RENAME_MANIFEST_2026-09-01.csv). Git history preserves the byte-level ancestry of every renamed artefact.

## Automated check

`python src/evaluation/verify_repository.py` confirms that all formally named notebooks, CARLA scripts and selected CARLA evidence files exist, and that informal legacy markers do not reappear in those active path scopes.
