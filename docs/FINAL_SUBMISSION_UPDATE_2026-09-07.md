# Final Submission Update Audit - refreshed 8 September 2026

Project: *Performance evaluation of YOLO-based vehicle detection under adverse conditions: Validation-bound evidence across ACDC, corrected DAWN and Combined training*

Author: Varis Jahirbhai Kureshi (35042321)

Supervisor: Yasir Javed

## Release decision

The repository now reflects the final dissertation narrative and the final 24-slide defence deck. The private Blackboard dissertation remains the authoritative submission copy because it contains completed administrative forms. The public repository copy preserves the research content and pagination but withholds the seven UREC1 and Publication Procedure Form images.

## Published artefacts

| Artefact | SHA-256 | Bytes | Verification |
|---|---|---:|---|
| Public dissertation DOCX | `EDBC237A0B1F31E1C008D5803C4766C44D8F04A1DD2A851A49A19C84E83ADD62` | 7,343,700 | 84 pages; 2,970 Microsoft Word-counted words from Introduction to References; no comments or tracked changes |
| Final defence PPTX | `749426F04EE5655F2182FE8B730747B4BF21B6020DA12BD462E74EC4922495A8` | 68,436,567 | 24 slides; 24 sourced note pages; no hidden slides or comments; one embedded 68,068,847-byte MP4 |

The public dissertation was derived from the verified private submission file identified by SHA-256 `3B91ABCA0CBB3D895F35992948E7B135C3625DBBEA307FAB3CFC0DB25456DA99`. Only the ethics/publication visibility statement, public-copy metadata and seven administrative form images differ.

## Content alignment

- The dissertation and presentation use the same exact research title.
- Both cover pages identify Sheffield Hallam University and the module `COMPUTING RESEARCH PROJECT (TRI3 BF-2025/6)`, code `55-710244-BF-20256`.
- Both retain the validation-bound interpretation that direct transfer was weak and Combined training offered the strongest overall balance without dominating every metric.
- CARLA remains qualitative diagnostic evidence only.
- The dissertation reports approved UREC1 evidence and the completed Publication Procedure Form as private Blackboard records.
- The presentation states that Blackboard approval was recorded and that the signed copy appears in the private Appendix B.
- The AITS 2 declaration remains present and supervisor-confirmed.
- This working-branch package supersedes the 2 September submission-package snapshot; `main` and the final release tag remain unchanged until the final approval step.

## Privacy boundary

The public report does not publish student or supervisor signatures, institutional email addresses from the UREC1 form, or the completed Publication Procedure Form. Each withheld page contains a visible notice explaining that the completed administrative record remains in the private Blackboard submission. Research findings, references, code links, evidence tables and appendices unrelated to the forms remain available.

## Verification commands

```bash
python src/evaluation/verify_public_release.py
python src/evaluation/verify_repository.py
python src/evaluation/verify_submission_package.py
python -m compileall -q src
```

All four checks must pass before the release tag is applied.
