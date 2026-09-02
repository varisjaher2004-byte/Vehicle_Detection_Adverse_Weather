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
        assert props["created"] == "2026-06-06T14:45:00Z", f"Unexpected DOCX creation date: {props['created']}"
        assert props["title"] == (
            "Performance Evaluation of YOLO-based Vehicle Detection under Adverse Environmental Conditions "
            "Using Simulation and Real-World Datasets"
        ), f"Unexpected DOCX title: {props['title']}"
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
        text = package_text(archive, "word/document.xml", WORD_NS)
    normalised = " ".join(text.split()).casefold()

    required_text = (
        "Performance evaluation of YOLO-based vehicle detection under adverse environmental conditions",
        "1. Introduction",
        "References",
        "Appendix A - AI Declaration",
        "Appendix B - Ethics Form and Approval Evidence",
        "Appendix G - Data, Code and Evidence",
        "Appendix J - Supporting Evidence",
        "Generative-AI tools were used within the permitted AITS 2 scope",
        "No passwords, authentication credentials, human-participant data",
        "no ethics approval is claimed",
        "0.1362",
        "0.1122",
        "0.4069",
        "0.6382",
        "0.5226",
    )
    for token in required_text:
        assert token.casefold() in normalised, f"Missing dissertation content: {token}"
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
        assert props["created"] == "2026-08-30T20:08:45.6080000Z", (
            f"Unexpected PPTX creation date: {props['created']}"
        )
        assert props["title"] == (
            "Performance Evaluation of YOLO-based Vehicle Detection under Adverse Environmental Conditions"
        ), f"Unexpected PPTX title: {props['title']}"
        assert props["subject"] == "MSc Artificial Intelligence Dissertation", (
            f"Unexpected PPTX subject: {props['subject']}"
        )
        forbidden = forbidden_office_members(archive)
        assert not forbidden, f"PPTX contains comments, macro, ActiveX or revision parts: {forbidden}"
        slides = numbered_members(archive, r"ppt/slides/slide\d+\.xml")
        notes = numbered_members(archive, r"ppt/notesSlides/notesSlide\d+\.xml")
        media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
        videos = [name for name in media if name.lower().endswith(".mp4")]
        assert len(slides) == 20, f"Expected 20 slides, found {len(slides)}"
        assert len(notes) == 20, f"Expected 20 note pages, found {len(notes)}"
        assert len(videos) == 1, f"Expected one embedded MP4, found {len(videos)}"
        assert archive.getinfo(videos[0]).file_size > 1024 * 1024, "Embedded MP4 is unexpectedly small"

        hidden_slides = []
        for index, member in enumerate(slides, start=1):
            root = ET.fromstring(archive.read(member))
            if root.attrib.get("show", "1") in {"0", "false", "False"}:
                hidden_slides.append(index)
        assert not hidden_slides, f"Hidden slides found: {hidden_slides}"

        external_relationships: list[str] = []
        for member in archive.namelist():
            if not member.endswith(".rels"):
                continue
            root = ET.fromstring(archive.read(member))
            for relationship in root.findall(f"{{{REL_NS}}}Relationship"):
                if relationship.attrib.get("TargetMode") == "External":
                    external_relationships.append(f"{member}: {relationship.attrib.get('Target', '')}")
        assert not external_relationships, f"External PPTX relationships found: {external_relationships}"

        metadata_members = [
            name
            for name in archive.namelist()
            if name in {"docProps/core.xml", "docProps/app.xml", "docProps/custom.xml"}
            or name.startswith(("ppt/theme/", "ppt/slideMasters/"))
        ]
        metadata_text = " ".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in metadata_members
        ).casefold()
        for unwanted in ("walnut exporter", "chatgpt"):
            assert unwanted not in metadata_text, f"Automated/template metadata found: {unwanted}"

        slide_text = " ".join(package_text(archive, name, DRAWING_NS) for name in slides)
        note_texts = [package_text(archive, name, DRAWING_NS) for name in notes]
        missing_sources = [index + 1 for index, text in enumerate(note_texts) if "[Sources]" not in text]
        assert not missing_sources, f"Speaker notes without [Sources]: {missing_sources}"
    normalised = " ".join(slide_text.split()).casefold()

    required_text = (
        "Context, problem and proposed solution",
        "Research question, aim, objectives and contribution",
        "One chart captures the result: performance is domain dependent",
        "Direct transfer collapses because recall remains low",
        "Combined training improves balance",
        "CARLA demonstrates conditions—not real-world robustness",
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
    assert ".1984" not in normalised, "Superseded rounded F1 display found in presentation"
    print(
        "DEFENCE PASS: clean metadata, no hidden slides/external links/revision parts, "
        "20 slides, 20 sourced note pages and one embedded MP4"
    )


def main() -> None:
    paths = verify_manifest()
    verify_dissertation(paths["FINAL_DISSERTATION"])
    verify_defence(paths["FINAL_DEFENCE"])
    print("FINAL SUBMISSION PACKAGE PASS")


if __name__ == "__main__":
    main()
