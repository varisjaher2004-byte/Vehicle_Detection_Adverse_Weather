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
        "0.1362",
        "0.1122",
        "0.4069",
        "0.6382",
        "0.5226",
    )
    for token in required_text:
        assert token.casefold() in normalised, f"Missing dissertation content: {token}"
    assert "0.1984" not in normalised, "Superseded rounded F1 display found in dissertation"
    print("DISSERTATION PASS: headings, appendices and canonical result values present")


def verify_defence(path: Path) -> None:
    with ZipFile(path) as archive:
        assert archive.testzip() is None, "Corrupt PPTX member"
        slides = numbered_members(archive, r"ppt/slides/slide\d+\.xml")
        notes = numbered_members(archive, r"ppt/notesSlides/notesSlide\d+\.xml")
        media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
        videos = [name for name in media if name.lower().endswith(".mp4")]
        assert len(slides) == 20, f"Expected 20 slides, found {len(slides)}"
        assert len(notes) == 20, f"Expected 20 note pages, found {len(notes)}"
        assert len(videos) == 1, f"Expected one embedded MP4, found {len(videos)}"

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
    print("DEFENCE PASS: 20 slides, 20 sourced note pages and one embedded MP4")


def main() -> None:
    paths = verify_manifest()
    verify_dissertation(paths["FINAL_DISSERTATION"])
    verify_defence(paths["FINAL_DEFENCE"])
    print("FINAL SUBMISSION PACKAGE PASS")


if __name__ == "__main__":
    main()
