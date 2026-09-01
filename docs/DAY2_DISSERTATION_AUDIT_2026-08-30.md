# Day 2 Dissertation Finalisation Audit

> Historical checkpoint: this audit is retained for provenance. It is superseded for current access by `docs/FINAL_REPOSITORY_AUDIT_2026-09-01.md` and the final dissertation under `docs/submission/`.

Project: *Performance Evaluation of YOLO-based Vehicle Detection under Adverse Environmental Conditions*

Author: Varis Jahirbhai Kureshi (Student ID 35042321)

Audit date: 30 August 2026

## Outcome

Day 2 dissertation editing is complete. The corrected report is stored at:

`docs/dissertation/Varis_Kureshi_Dissertation_Day2_Corrected_2026-08-30.docx`

The student confirmed in Microsoft Word that the exact Introduction-to-Conclusion selection contains 2,996 words. This is the counting boundary declared on the report cover page. Presentation material and appendices are outside that declared core-text range.

## Locked academic controls

- Quantitative performance is framed as validation-bound evidence, not independent-test accuracy or a generalized real-world performance bound.
- Training is explicitly described as single-seed, descriptive and non-causal; no confidence intervals or architectural-superiority claims are inferred.
- CARLA simulation remains qualitative diagnostic evidence and is excluded from the seven-cell real-world quantitative validation matrix.
- DAWN label remapping, checkpoint SHA-256 provenance and Git lineage remain explicit.
- Class and domain imbalance are disclosed through domain-level image counts and instance support.
- Selected qualitative examples are identified as deliberately successful, high-confidence illustrations; they are not representative error analysis and are not matched evidence for every quantitative cell.
- The ACDC Fog prediction/ground-truth panel and low-level architecture/test material are placed in Appendix J.
- Detailed dataset provenance, integrity auditing and weighting notes are placed in Appendix G.
- The cross-domain ACDC-to-DAWN row retains F1 = 0.1983 because the authoritative unrounded values give F1 = 0.198331 under F1 = 2PR/(P+R).

## Document verification

- Microsoft Word core word count: 2,996 (student-confirmed)
- Word-count scope: exact Introduction-to-Conclusion range
- Page count in the final render: 77
- Contents, List of Tables and List of Figures pagination: verified against the final render
- Tracked changes, comments and unresolved placeholders: none
- Appendix tables, captions and notes: 12 pt
- Repeating table headers: enabled
- Sections 3.8-3.11 body paragraphs: left aligned
- Final DOCX SHA-256: `0715179BE4ACAC68C09BDFABF8080506F24D629DF888097EFB0876CDD0FC46B7`

## Administrative boundary

Signed UREC1 evidence and the signed publication form are external student-controlled prerequisites. They are intentionally excluded from Git because they may contain personal or administrative information. Their absence from this repository is not evidence that the forms have been completed.

## Remaining work after Day 2

- Obtain and retain the signed UREC1 and publication-form evidence before institutional submission.
- Complete the separate PowerPoint refinement and viva preparation work.
- Perform the final submission-package audit and freeze only after all later roadmap tasks pass.
