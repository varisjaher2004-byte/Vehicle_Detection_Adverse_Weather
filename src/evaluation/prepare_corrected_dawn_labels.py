"""Build a non-destructive DAWN data view with unified validation class IDs.

The source folders are read-only inputs. Images and labels are copied to a new
output directory, and only the copied validation labels are remapped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


RAW_TO_UNIFIED = {1: 0, 2: 7, 3: 2, 4: 6, 6: 4, 8: 3}
NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def copy_images(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    files = image_files(source)
    for item in files:
        shutil.copy2(item, target / item.name)
    return len(files)


def copy_training_labels(source: Path, target: Path) -> tuple[int, int]:
    target.mkdir(parents=True, exist_ok=True)
    count = instances = 0
    for label in sorted(source.glob("*.txt")):
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        for number, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) != 5 or int(fields[0]) not in range(8):
                raise ValueError(f"Invalid unified row: {label}:{number}")
        (target / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1
        instances += len(lines)
    return count, instances


def remap_validation_labels(source: Path, target: Path) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    before: Counter[int] = Counter()
    after: Counter[int] = Counter()
    files = 0
    for label in sorted(source.glob("*.txt")):
        converted: list[str] = []
        for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed YOLO row: {label}:{number}")
            raw_id = int(fields[0])
            if raw_id not in RAW_TO_UNIFIED:
                raise ValueError(f"Unexpected raw class {raw_id}: {label}:{number}")
            values = [float(value) for value in fields[1:]]
            if any(value < 0 or value > 1 for value in values):
                raise ValueError(f"Out-of-range box: {label}:{number}")
            class_id = RAW_TO_UNIFIED[raw_id]
            before[raw_id] += 1
            after[class_id] += 1
            converted.append(" ".join([str(class_id), *fields[1:]]))
        (target / label.name).write_text(
            "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
        )
        files += 1
    return {
        "validation_labels_written": files,
        "class_instances_before": dict(sorted(before.items())),
        "class_instances_after": dict(sorted(after.items())),
    }


def write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(item.resolve().as_posix() for item in images) + "\n", encoding="utf-8")


def write_yaml(path: Path, root: Path, train: str, val: str) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(NAMES))
    text = (
        f"path: {root.resolve().as_posix()}\n"
        f"train: {train}\nval: {val}\n\nnc: 8\nnames:\n{names}\n"
    )
    path.write_text(text, encoding="utf-8")


def verify_xml(xml_root: Path | None, corrected_labels: Path) -> dict[str, int]:
    if xml_root is None:
        return {"fog_xml_files_verified": 0, "fog_objects_verified": 0}
    checked_files = checked_objects = 0
    for xml_path in sorted(xml_root.glob("*.xml")):
        label = corrected_labels / f"{xml_path.stem}.txt"
        if not label.is_file():
            continue
        xml_names = [node.findtext("name", "").strip() for node in ET.parse(xml_path).getroot().findall("object")]
        ids = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(xml_names) != len(ids):
            continue
        if [NAMES[class_id] for class_id in ids] != xml_names:
            raise AssertionError(f"XML/YOLO class mismatch: {xml_path.name}")
        checked_files += 1
        checked_objects += len(ids)
    return {"fog_xml_files_verified": checked_files, "fog_objects_verified": checked_objects}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dawn-entire-root", type=Path, required=True)
    parser.add_argument("--dawn-fog-root", type=Path, required=True)
    parser.add_argument("--dawn-rain-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fog-voc-root", type=Path)
    args = parser.parse_args()

    source = args.dawn_entire_root.resolve()
    output = args.output_root.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from the source dataset")

    train_images = copy_images(source / "images" / "train", output / "images" / "train")
    val_images = copy_images(source / "images" / "val", output / "images" / "val")
    train_labels, train_instances = copy_training_labels(
        source / "labels" / "train", output / "labels" / "train"
    )
    remap = remap_validation_labels(source / "labels" / "val", output / "labels" / "val")

    for split in ("train", "val"):
        expected = {p.stem for p in image_files(output / "images" / split)}
        labels = {p.stem for p in (output / "labels" / split).glob("*.txt")}
        if expected != labels:
            raise AssertionError(f"Image/label mismatch in {split}")

    condition_counts: dict[str, int] = {}
    condition_sets: dict[tuple[str, str], set[str]] = {}
    for condition, root in (("fog", args.dawn_fog_root), ("rain", args.dawn_rain_root)):
        for split in ("train", "val"):
            candidates = image_files(root / "images" / split)
            target = output / "images" / split
            names = {p.stem for p in candidates}
            if not names <= {p.stem for p in image_files(target)}:
                raise AssertionError(f"{condition} {split} is not a subset of DAWN Entire")
            condition_sets[(condition, split)] = names
            write_list(output / "lists" / f"dawn_{condition}_{split}.txt", [target / p.name for p in candidates])
            condition_counts[f"{condition}_{split}_images"] = len(candidates)
    for split in ("train", "val"):
        fog = condition_sets[("fog", split)]
        rain = condition_sets[("rain", split)]
        entire = {p.stem for p in image_files(output / "images" / split)}
        if fog & rain or fog | rain != entire:
            raise AssertionError(f"Fog/Rain {split} sets are not disjoint and complete")

    write_yaml(output / "dawn_entire.yaml", output, "images/train", "images/val")
    write_yaml(output / "dawn_fog.yaml", output, "lists/dawn_fog_train.txt", "lists/dawn_fog_val.txt")
    write_yaml(output / "dawn_rain.yaml", output, "lists/dawn_rain_train.txt", "lists/dawn_rain_val.txt")

    audit = {
        "source_files_modified": False,
        "mapping": RAW_TO_UNIFIED,
        "unified_names": dict(enumerate(NAMES)),
        "training_images": train_images,
        "training_labels": train_labels,
        "training_instances": train_instances,
        "validation_images": val_images,
        **remap,
        **condition_counts,
        **verify_xml(args.fog_voc_root, output / "labels" / "val"),
    }
    (output / "label_remap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print("CORRECTED DAWN DATA PREPARATION PASS")


if __name__ == "__main__":
    main()

