# Final Submission Update Audit - 7 September 2026

Project: *Performance evaluation of YOLO-based vehicle detection under adverse conditions: Validation-bound evidence across ACDC, corrected DAWN and Combined training*

Author: Varis Jahirbhai Kureshi (35042321)

Supervisor: Yasir Javed

## Release decision

The repository now reflects the final dissertation narrative and the final 24-slide defence deck. The private Blackboard dissertation remains the authoritative submission copy because it contains completed administrative forms. The public repository copy preserves the research content and pagination but withholds the seven UREC1 and Publication Procedure Form images.

## Published artefacts

| Artefact | SHA-256 | Bytes | Verification |
|---|---|---:|---|
| Public dissertation DOCX | `4051B6B5E77D1CD4CCDBE70264DBBE7539551685B208F4EB713EBB24C89BC48A` | 7,307,008 | 84 pages; 2,993 Microsoft Word-counted words from Introduction to References; no comments or tracked changes |
| Final defence PPTX | `B6DA4FB5580AE5220C7D844095589404080401BA6B50C8BAB5083963C54C6DB7` | 68,400,016 | 24 slides; 24 sourced note pages; no hidden slides or comments; one embedded 68,068,847-byte MP4 |

The public dissertation was derived from the verified private submission file identified by SHA-256 `7A0861C064EE7B493B7F73E54CF2CFB067CC7FFD6850425F2B40D5AD2528DF51`. Only the ethics/publication visibility statement, public-copy metadata and seven administrative form images differ.

## Content alignment

- The dissertation and presentation use the same exact research title.
- Both retain the validation-bound interpretation that direct transfer was weak and Combined training offered the strongest overall balance without dominating every metric.
- CARLA remains qualitative diagnostic evidence only.
- The dissertation reports approved UREC1 evidence and the completed Publication Procedure Form as private Blackboard records.
- The presentation states that Blackboard approval was recorded and that the signed copy appears in the private Appendix B.
- The AITS 2 declaration remains present and supervisor-confirmed.
- The final repository release commit and tag supersede the 2 September submission-package snapshot.

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
