# Day 3 Defence Presentation Audit

> Historical checkpoint: this 13-slide audit is retained for provenance. It is superseded for current access by `docs/FINAL_REPOSITORY_AUDIT_2026-09-01.md` and the final 20-slide deck under `docs/submission/`.

Project: *Performance Evaluation of YOLO-based Vehicle Detection under Adverse Environmental Conditions*

Author: Varis Jahirbhai Kureshi (35042321)

Audit date: 31 August 2026

## Accepted presentation

`docs/presentation/Varis_Kureshi_Dissertation_Defence_Day3_Final_2026-08-30.pptx`

The accepted deck contains 13 slides and 13 speaker-note pages. It is structured for an approximately 10–12 minute MSc defence presentation and uses the established dissertation visual style.

## Evidence and validity controls

- All ACDC, corrected DAWN and Combined numerical results are framed as validation-bound estimates.
- The seven protocol-matched quantitative cells are kept separate from CARLA qualitative diagnostic assets.
- CARLA is not numerically ranked against ACDC or DAWN.
- The study is described as single-seed, descriptive and non-causal, without confidence intervals or significance claims.
- No untouched independent hold-out is claimed.
- Unequal class and domain support is made explicit.
- The rare ACDC train example is not presented as robust class-level performance.
- The DAWN Rain result is qualified by its 42-image validation support and limited framing/background diversity.

## Metric reconciliation

The ACDC-to-DAWN F1 display is `.1983`. It is derived from the authoritative unrounded same-row precision and recall values. The superseded `.1984` display is absent.

## Visual evidence correction

The earlier ACDC Fog placeholder was replaced by a matched ground-truth/prediction crop from the same validation frame. Source pixels, boxes, labels and confidences were not edited. The slide states that this example illustrates errors but does not estimate a cell-level error rate.

## Traceability

- Accepted checkpoints remain SHA-256 locked.
- Repository lineage retains commit `1eb81b5` for corrected evidence.
- The presentation is based on the Day 2 repository state `8e4be2b`.
- Every slide has a speaker-note `[Sources]` block.

## Quality assurance

- Template fidelity check: PASS
- Slide-canvas overflow test: PASS
- Full visual inspection: 13/13 slides PASS
- Unresolved ACDC Fog placeholder: absent
- Superseded F1 `.1984`: absent
- Administrative UREC1 and publication documentation: external student responsibility and not stored in this Git update

## Git scope

This Day 3 update contains only the final presentation and its audit/hash records. It does not add model checkpoint binaries, datasets, raw experiments, CONTROL evidence or submission-ready administrative documents.
