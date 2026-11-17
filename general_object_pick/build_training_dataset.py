#!/usr/bin/env python3
"""Validate reviewed annotations and build a grouped YOLO train/val/test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "printed_apple"
ORGANIZED_ROOT = DATASET_ROOT / "organized"
LABEL_ROOT = DATASET_ROOT / "labels"
REVIEW_PATH = DATASET_ROOT / "annotation_review.json"
EXCLUDE_PATH = DATASET_ROOT / "training_exclude.txt"
OUTPUT_ROOT = DATASET_ROOT / "yolo_dataset"
CLASS_NAMES = ("red_apple", "green_apple")
CATEGORIES = ("red_apple", "green_apple", "mixed_apples", "negative")
FRAME_PATTERN = re.compile(r"printed_apple_(\d{8}_\d{6})__frame_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Maximum consecutive frames kept in one split group.",
    )
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    return LABEL_ROOT / image_path.parent.name / f"{image_path.stem}.txt"


def load_exclusions() -> set[Path]:
    if not EXCLUDE_PATH.is_file():
        return set()
    result = set()
    for line in EXCLUDE_PATH.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            result.add((PROJECT_ROOT / value).resolve())
    return result


def validate_label(path: Path) -> Counter:
    counts: Counter = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from error
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_number}: unsupported class {class_id}")
        x, y, width, height = values
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path}:{line_number}: center is outside the image")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: invalid box size")
        if x - width / 2 < -1e-6 or x + width / 2 > 1.0 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box exceeds image width")
        if y - height / 2 < -1e-6 or y + height / 2 > 1.0 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box exceeds image height")
        counts[class_id] += 1
    return counts


def source_position(path: Path) -> tuple[str, int]:
    match = FRAME_PATTERN.search(path.name)
    if not match:
        return path.stem, 0
    return match.group(1), int(match.group(2))


def make_groups(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    """Keep short consecutive bursts together to reduce near-duplicate leakage."""
    runs: list[list[Path]] = []
    for category in CATEGORIES:
        by_session: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for path in paths:
            if path.parent.name == category:
                session, frame = source_position(path)
                by_session[session].append((frame, path))
        for entries in by_session.values():
            entries.sort()
            current: list[Path] = []
            previous_frame: int | None = None
            for frame, path in entries:
                if previous_frame is not None and frame - previous_frame > 2:
                    if current:
                        runs.append(current)
                    current = []
                current.append(path)
                previous_frame = frame
            if current:
                runs.append(current)

    groups = []
    for run in runs:
        groups.extend(run[index : index + chunk_size] for index in range(0, len(run), chunk_size))
    return groups


def assign_splits(groups: list[list[Path]], seed: int) -> dict[Path, str]:
    """Stratify grouped bursts approximately 70/20/10 by folder category."""
    grouped: dict[str, list[list[Path]]] = defaultdict(list)
    for group in groups:
        grouped[group[0].parent.name].append(group)

    assignments: dict[Path, str] = {}
    for category, category_groups in grouped.items():
        category_groups.sort(
            key=lambda group: hashlib.sha256(
                f"{seed}:{category}:{group[0].name}".encode()
            ).hexdigest()
        )
        count = len(category_groups)
        test_count = 1 if count >= 3 else 0
        val_count = max(1, round(count * 0.2)) if count >= 2 else 0
        if test_count + val_count >= count:
            val_count = max(0, count - test_count - 1)
        split_order = (
            ["test"] * test_count
            + ["val"] * val_count
            + ["train"] * (count - test_count - val_count)
        )
        for group, split in zip(category_groups, split_order):
            for path in group:
                assignments[path] = split
    return assignments


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    images = [
        path.resolve()
        for category in CATEGORIES
        for path in sorted((ORGANIZED_ROOT / category).glob("*.jpg"))
    ]
    if not images:
        raise SystemExit(f"No organized images found under {ORGANIZED_ROOT}")

    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    reviewed = review.get("images", {})
    missing_review = [
        path
        for path in images
        if not reviewed.get(str(path.relative_to(PROJECT_ROOT)), {}).get("reviewed")
    ]
    if missing_review:
        raise SystemExit(f"Refusing to build: {len(missing_review)} image(s) are not reviewed")

    exclusions = load_exclusions()
    unknown_exclusions = sorted(exclusions - set(images))
    if unknown_exclusions:
        raise SystemExit(f"Exclusion file contains {len(unknown_exclusions)} missing image(s)")

    included = [path for path in images if path not in exclusions]
    class_totals: Counter = Counter()
    image_class_counts: dict[Path, Counter] = {}
    for image_path in included:
        annotation = label_path(image_path)
        if not annotation.is_file():
            raise SystemExit(f"Missing annotation: {annotation}")
        try:
            counts = validate_label(annotation)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        image_class_counts[image_path] = counts
        class_totals.update(counts)

    groups = make_groups(included, args.chunk_size)
    assignments = assign_splits(groups, args.seed)
    if set(assignments) != set(included):
        raise SystemExit("Internal error: not every image received a split")

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    split_images: Counter = Counter()
    split_boxes: dict[str, Counter] = defaultdict(Counter)
    for image_path in sorted(included):
        split = assignments[image_path]
        annotation = label_path(image_path)
        shutil.copy2(image_path, args.output / "images" / split / image_path.name)
        shutil.copy2(annotation, args.output / "labels" / split / annotation.name)
        counts = image_class_counts[image_path]
        split_images[split] += 1
        split_boxes[split].update(counts)
        manifest_rows.append(
            {
                "split": split,
                "category": image_path.parent.name,
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "red_boxes": counts[0],
                "green_boxes": counts[1],
            }
        )

    data_yaml = (
        f"path: /workspace/{args.output.relative_to(PROJECT_ROOT)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: red_apple\n"
        "  1: green_apple\n"
    )
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Reviewed source images: {len(images)}")
    print(f"Excluded ambiguous images: {len(exclusions)}")
    print(f"Training dataset images: {len(included)}")
    for split in ("train", "val", "test"):
        boxes = split_boxes[split]
        print(
            f"  {split}: {split_images[split]} images, "
            f"red boxes={boxes[0]}, green boxes={boxes[1]}"
        )
    print(f"Total boxes: red={class_totals[0]}, green={class_totals[1]}")
    print(f"Dataset YAML: {args.output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
