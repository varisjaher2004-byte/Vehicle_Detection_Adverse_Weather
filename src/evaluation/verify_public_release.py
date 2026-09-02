"""Audit the committed repository for a safe, portable public release."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
EXPECTED_NOTEBOOKS = {
    "acdc_entire_training.ipynb",
    "acdc_fog_training.ipynb",
    "acdc_night_training.ipynb",
    "acdc_rain_training.ipynb",
    "combined_yolov8_training.ipynb",
    "dawn_entire_training.ipynb",
    "dawn_fog_training.ipynb",
    "dawn_rain_training.ipynb",
    "yolov8_inference_pipeline.ipynb",
}
REQUIRED_PATHS = {
    "README.md",
    "CITATION.cff",
    "requirements.txt",
    "docs/GETTING_STARTED.md",
    "docs/EVIDENCE_INTEGRITY.md",
    "docs/FINAL_SUBMISSION_MANIFEST.csv",
    "docs/submission/Varis_Kureshi_Dissertation_SUBMISSION_READY_FINAL_2026-08-31.docx",
    "docs/submission/Varis_Kureshi_Dissertation_Defence_MSC_SUBMISSION_READY_FINAL_2026-09-01.pptx",
    "results/CORRECTED_2026-08-27/final_cross_domain_validation_matrix.csv",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".7z",
    ".pyc",
    ".pyo",
}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "OpenAI-style secret key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def repository_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPOSITORY_ROOT,
        )
        return sorted(
            REPOSITORY_ROOT / Path(raw.decode("utf-8"))
            for raw in output.split(b"\0")
            if raw
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in REPOSITORY_ROOT.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(REPOSITORY_ROOT).parts
        )


def verify_required_paths() -> None:
    missing = sorted(path for path in REQUIRED_PATHS if not (REPOSITORY_ROOT / path).is_file())
    assert not missing, f"Missing required public-release files: {missing}"
    print(f"STRUCTURE PASS: {len(REQUIRED_PATHS)} required entry points are present")


def verify_file_policy(files: list[Path]) -> None:
    violations: list[str] = []
    signed_form_pattern = re.compile(
        r"(?:signed.*(?:urec|ethics|publication)|(?:urec|ethics|publication).*signed)", re.IGNORECASE
    )
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT)
        if path.stat().st_size >= MAX_GITHUB_FILE_BYTES:
            violations.append(f"GitHub 100 MiB limit: {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            violations.append(f"excluded binary type: {relative}")
        if path.name.casefold() == ".env" or "__pycache__" in relative.parts:
            violations.append(f"local/private runtime artefact: {relative}")
        if signed_form_pattern.search(path.stem):
            violations.append(f"signed administrative form must remain private: {relative}")
    assert not violations, "Public-release policy violations:\n- " + "\n- ".join(violations)
    largest = max(files, key=lambda item: item.stat().st_size)
    print(
        "FILE POLICY PASS: no raw weights, standalone videos, archives, caches or signed forms; "
        f"largest file is {largest.relative_to(REPOSITORY_ROOT)} ({largest.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


def verify_notebooks() -> None:
    notebooks = sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
    assert {path.name for path in notebooks} == EXPECTED_NOTEBOOKS, "Unexpected notebook inventory"
    executed: list[str] = []
    outputs: list[str] = []
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4, f"Unsupported notebook format: {path.name}"
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                executed.append(path.name)
            if cell.get("outputs"):
                outputs.append(path.name)
    assert not executed, f"Notebook execution counts were not cleared: {sorted(set(executed))}"
    assert not outputs, f"Notebook outputs were not cleared: {sorted(set(outputs))}"
    print("NOTEBOOK PASS: nine archival notebooks are valid and contain no outputs or execution counts")


def local_link_target(markdown: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    target = unquote(target).split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    return (markdown.parent / target).resolve()


def verify_markdown_links() -> None:
    broken: list[str] = []
    checked = 0
    for markdown in REPOSITORY_ROOT.rglob("*.md"):
        if ".git" in markdown.relative_to(REPOSITORY_ROOT).parts:
            continue
        text = markdown.read_text(encoding="utf-8", errors="replace")
        for match in MARKDOWN_LINK.finditer(text):
            target = local_link_target(markdown, match.group(1))
            if target is None:
                continue
            checked += 1
            if not target.exists():
                broken.append(f"{markdown.relative_to(REPOSITORY_ROOT)} -> {match.group(1)}")
    assert not broken, "Broken internal Markdown links:\n- " + "\n- ".join(broken)
    print(f"LINK PASS: {checked} internal Markdown links resolve")


def verify_no_secrets(files: list[Path]) -> None:
    findings: list[str] = []
    for path in files:
        if path.suffix.casefold() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(REPOSITORY_ROOT)}")
    assert not findings, "Possible secrets detected:\n- " + "\n- ".join(findings)
    print("SECRET PASS: no credential-shaped tokens or private keys detected in public text files")


def main() -> None:
    files = repository_files()
    verify_required_paths()
    verify_file_policy(files)
    verify_notebooks()
    verify_markdown_links()
    verify_no_secrets(files)
    print(f"PUBLIC RELEASE PASS: {len(files)} repository files inspected")


if __name__ == "__main__":
    main()
