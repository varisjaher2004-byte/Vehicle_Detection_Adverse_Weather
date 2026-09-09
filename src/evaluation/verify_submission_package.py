"""Verify final dissertation and defence artefacts without Office dependencies."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "docs" / "FINAL_SUBMISSION_MANIFEST.csv"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
CORE_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
EXPECTED_AUTHOR = "Varis Jahirbhai Kureshi"
EXPECTED_REPOSITORY_URL = (
    "https://github.com/varisjaher2004-byte/Vehicle_Detection_Adverse_Weather"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def package_text(archive: ZipFile, member: str, namespace: str) -> str:
    root = ET.fromstring(archive.read(member))
    return " ".join((node.text or "") for node in root.iter(f"{{{namespace}}}t"))


def numbered_members(archive: ZipFile, pattern: str) -> list[str]:
    expression = re.compile(pattern)
    members = [name for name in archive.namelist() if expression.fullmatch(name)]
    return sorted(members, key=lambda name: int(re.search(r"(\d+)(?=\.xml$)", name).group(1)))


def core_properties(archive: ZipFile) -> dict[str, str]:
    root = ET.fromstring(archive.read("docProps/core.xml"))
    names = {
        "title": (DC_NS, "title"),
        "subject": (DC_NS, "subject"),
        "description": (DC_NS, "description"),
        "creator": (DC_NS, "creator"),
        "last_modified_by": (CORE_NS, "lastModifiedBy"),
        "created": (DCTERMS_NS, "created"),
        "modified": (DCTERMS_NS, "modified"),
    }
    return {
        name: (root.findtext(f"{{{namespace}}}{tag}") or "").strip()
        for name, (namespace, tag) in names.items()
    }


def forbidden_office_members(archive: ZipFile) -> list[str]:
    patterns = (
        "vbaproject.bin",
        "activex",
        "/comments",
        "commentauthors",
        "people.xml",
        "revisioninfo",
    )
    return [name for name in archive.namelist() if any(token in name.casefold() for token in patterns)]


def verify_manifest() -> dict[str, Path]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["artifact_id"] for row in rows} == {"FINAL_DISSERTATION", "FINAL_DEFENCE"}
    paths: dict[str, Path] = {}
    for row in rows:
        path = REPOSITORY_ROOT / row["path"]
        assert path.is_file(), f"Missing final artefact: {row['path']}"
        assert path.stat().st_size == int(row["bytes"]), f"Byte-size mismatch: {row['path']}"
        assert sha256(path) == row["sha256"].upper(), f"SHA-256 mismatch: {row['path']}"
        assert row["verification_status"] == "PASS"
        paths[row["artifact_id"]] = path
        print(f"ARTEFACT PASS: {row['artifact_id']} {path.stat().st_size:,} bytes")
    return paths


def verify_dissertation(path: Path) -> None:
    with ZipFile(path) as archive:
        assert archive.testzip() is None, "Corrupt DOCX member"
        required_members = {"word/document.xml", "word/styles.xml", "word/settings.xml"}
        assert required_members <= set(archive.namelist()), "Incomplete DOCX package"
        props = core_properties(archive)
        assert props["creator"] == EXPECTED_AUTHOR, f"Unexpected DOCX author: {props['creator']}"
        assert props["last_modified_by"] == EXPECTED_AUTHOR, (
            f"Unexpected DOCX last-modified author: {props['last_modified_by']}"
        )
        assert props["created"] == "2026-09-06T08:39:00Z", f"Unexpected DOCX creation date: {props['created']}"
        assert props["title"] == (
            "Performance evaluation of YOLO-based vehicle detection under adverse conditions: "
            "Validation-bound evidence across ACDC, corrected DAWN and Combined training"
        ), f"Unexpected DOCX title: {props['title']}"
        assert "public repository copy" in props["description"].casefold(), (
            "DOCX metadata does not identify the public repository copy"
        )
        forbidden = forbidden_office_members(archive)
        assert not forbidden, f"DOCX contains comments, macro, ActiveX or revision parts: {forbidden}"
        tracked_changes: list[str] = []
        for member in archive.namelist():
            if not member.startswith("word/") or not member.endswith(".xml"):
                continue
            root = ET.fromstring(archive.read(member))
            if root.find(f".//{{{WORD_NS}}}ins") is not None or root.find(f".//{{{WORD_NS}}}del") is not None:
                tracked_changes.append(member)
        assert not tracked_changes, f"DOCX contains tracked insertions/deletions: {tracked_changes}"
        expected_public_admin_images = {
            **{
                f"word/media/image{index}.png": (
                    "908D43A809CC2D249D0DB8262472F458C706F2468BB32F550F79638464B9C2B2"
                )
                for index in range(1, 7)
            },
            "word/media/image7.png": (
                "B9B2217D2523EF845FE53562AE07031D16D3F3CAD1514AB896F1C2E7DFCC0963"
            ),
        }
        for member, expected_hash in expected_public_admin_images.items():
            assert member in archive.namelist(), f"Missing public administrative placeholder: {member}"
            actual_hash = hashlib.sha256(archive.read(member)).hexdigest().upper()
            assert actual_hash == expected_hash, f"Administrative placeholder mismatch: {member}"
        text = package_text(archive, "word/document.xml", WORD_NS)
    normalised = " ".join(text.split()).casefold()

    required_text = (
        "Performance evaluation of YOLO-based vehicle detection under adverse conditions",
        "COMPUTING RESEARCH PROJECT (TRI3 BF-2025/6)",
        "55-710244-BF-20256",
        "1. Introduction",
        "References",
        "Appendix A - AI Declaration",
        "Appendix B - Ethics Form and Approval Evidence",
        "Appendix C - Official Publication Procedure Form",
        "Appendix G - Data, Code and Evidence",
        "Appendix J - Supporting Evidence",
        "Generative-AI tools were used within the permitted AITS 2 scope",
        "No passwords, authentication credentials, human-participant data",
        "public repository omits signed pages",
        "0.1362",
        "0.1122",
        "0.4069",
        "0.6382",
        "0.5226",
    )
    for token in required_text:
        assert token.casefold() in normalised, f"Missing dissertation content: {token}"
    for unwanted in ("chatgpt", "codex", "openai", "walnut exporter"):
        assert unwanted not in normalised, f"Unwanted tool/template name found in dissertation: {unwanted}"
    assert "0.1984" not in normalised, "Superseded rounded F1 display found in dissertation"
    print(
        "DISSERTATION PASS: metadata, clean revision state, transparent AI/ethics declarations, "
        "appendices and canonical result values present"
    )


def verify_defence(path: Path) -> None:
    with ZipFile(path) as archive:
        assert archive.testzip() is None, "Corrupt PPTX member"
        props = core_properties(archive)
        assert props["creator"] == EXPECTED_AUTHOR, f"Unexpected PPTX author: {props['creator']}"
        assert props["last_modified_by"] == EXPECTED_AUTHOR, (
            f"Unexpected PPTX last-modified author: {props['last_modified_by']}"
        )
        assert props["created"] == "2026-08-31T23:10:35Z", (
            f"Unexpected PPTX creation date: {props['created']}"
        )
        assert props["title"] == (
            "Performance evaluation of YOLO-based vehicle detection under adverse conditions: "
            "Validation-bound evidence across ACDC, corrected DAWN and Combined training"
        ), f"Unexpected PPTX title: {props['title']}"
        assert props["subject"] == "MSc Artificial Intelligence dissertation defence", (
            f"Unexpected PPTX subject: {props['subject']}"
        )
        forbidden = forbidden_office_members(archive)
        assert not forbidden, f"PPTX contains comments, macro, ActiveX or revision parts: {forbidden}"
        slides = numbered_members(archive, r"ppt/slides/slide\d+\.xml")
        notes = numbered_members(archive, r"ppt/notesSlides/notesSlide\d+\.xml")
        media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
        videos = [name for name in media if name.lower().endswith(".mp4")]
        assert len(slides) == 24, f"Expected 24 slides, found {len(slides)}"
        assert len(notes) == 24, f"Expected 24 note pages, found {len(notes)}"
        assert len(videos) == 1, f"Expected one embedded MP4, found {len(videos)}"
        assert archive.getinfo(videos[0]).file_size > 1024 * 1024, "Embedded MP4 is unexpectedly small"

        hidden_slides = []
        for index, member in enumerate(slides, start=1):
            root = ET.fromstring(archive.read(member))
            if root.attrib.get("show", "1") in {"0", "false", "False"}:
                hidden_slides.append(index)
        assert not hidden_slides, f"Hidden slides found: {hidden_slides}"

        external_relationships: list[str] = []
        approved_repository_links = 0
        for member in archive.namelist():
            if not member.endswith(".rels"):
                continue
            root = ET.fromstring(archive.read(member))
            for relationship in root.findall(f"{{{REL_NS}}}Relationship"):
                if relationship.attrib.get("TargetMode") == "External":
                    target = relationship.attrib.get("Target", "")
                    relation_type = relationship.attrib.get("Type", "")
                    if (
                        member == "ppt/slides/_rels/slide1.xml.rels"
                        and relation_type.endswith("/hyperlink")
                        and target == EXPECTED_REPOSITORY_URL
                    ):
                        approved_repository_links += 1
                    else:
                        external_relationships.append(f"{member}: {target}")
        assert not external_relationships, (
            f"Unapproved external PPTX relationships found: {external_relationships}"
        )
        assert approved_repository_links == 1, (
            f"Expected one approved cover repository link, found {approved_repository_links}"
        )

        metadata_members = [
            name
            for name in archive.namelist()
            if name in {"docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"}
            or name.startswith(("ppt/theme/", "ppt/slideMasters/"))
        ]
        metadata_text = " ".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in metadata_members
        ).casefold()
        for unwanted in ("walnut exporter", "chatgpt", "codex", "openai"):
            assert unwanted not in metadata_text, f"Automated/template metadata found: {unwanted}"

        slide_text = " ".join(package_text(archive, name, DRAWING_NS) for name in slides)
        note_texts = [package_text(archive, name, DRAWING_NS) for name in notes]
        missing_sources = [index + 1 for index, text in enumerate(note_texts) if "[Sources]" not in text]
        assert not missing_sources, f"Speaker notes without [Sources]: {missing_sources}"
    normalised = " ".join(slide_text.split()).casefold()

    required_text = (
        "Context, problem and proposed solution",
        "COMPUTING RESEARCH PROJECT (TRI3 BF-2025/6)",
        "55-710244-BF-20256",
        "github.com/varisjaher2004-byte/Vehicle_Detection_Adverse_Weather",
        "Research question, aim, objectives and contribution",
        "Research methodology and justification",
        "Ethics and data governance",
        "Implementation and testing evidence",
        "One chart captures the result: performance is domain dependent",
        "Direct transfer collapses because recall remains low",
        "Combined training improves balance",
        "CARLA demonstrates conditions—not real-world robustness",
        "AI use declaration — AITS 2",
        "Backup: complete seven-cell validation matrix",
        "0.1362",
        "0.1122",
        "0.4069",
        "0.6382",
        "0.1242",
        "0.5226",
        ".1983",
    )
    for token in required_text:
        assert token.casefold() in normalised, f"Missing presentation content: {token}"
    for unwanted in ("chatgpt", "codex", "openai", "walnut exporter"):
        assert unwanted not in normalised, f"Unwanted tool/template name found in presentation: {unwanted}"
    assert ".1984" not in normalised, "Superseded rounded F1 display found in presentation"
    print(
        "DEFENCE PASS: clean metadata, no hidden slides/unapproved external links/revision parts, "
        "one approved cover repository link, 24 slides, 24 sourced note pages and one embedded MP4"
    )


def main() -> None:
    paths = verify_manifest()
    verify_dissertation(paths["FINAL_DISSERTATION"])
    verify_defence(paths["FINAL_DEFENCE"])
    print("FINAL SUBMISSION PACKAGE PASS")


if __name__ == "__main__":
    main()
