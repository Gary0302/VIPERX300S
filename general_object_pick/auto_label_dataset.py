#!/usr/bin/env python3
"""Create review-required YOLO labels for the printed-apple dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORGANIZED_ROOT = PROJECT_ROOT / "datasets" / "printed_apple" / "organized"
LABEL_ROOT = PROJECT_ROOT / "datasets" / "printed_apple" / "labels"
REPORT_PATH = PROJECT_ROOT / "datasets" / "printed_apple" / "auto_label_report.csv"
CLASS_NAMES = ("red_apple", "green_apple")


def color_class(image, xyxy: list[float]) -> tuple[int, float, float]:
    """Classify an apple crop as red or green using its central color pixels."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0, 0.0, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    crop_height, crop_width = crop.shape[:2]
    yy, xx = np.ogrid[:crop_height, :crop_width]
    center_x = (crop_width - 1) / 2.0
    center_y = (crop_height - 1) / 2.0
    radius_x = max(crop_width * 0.46, 1.0)
    radius_y = max(crop_height * 0.46, 1.0)
    center_mask = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1.0
    )
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    visible = center_mask & (value >= 35)
    red_mask = visible & ((hue <= 15) | (hue >= 170)) & (saturation >= 55)
    green_mask = visible & (hue >= 25) & (hue <= 100) & (saturation >= 20)
    red_score = float(red_mask.sum()) / max(float(visible.sum()), 1.0)
    green_score = float(green_mask.sum()) / max(float(visible.sum()), 1.0)
    return (0 if red_score >= green_score else 1), red_score, green_score


def yolo_line(class_id: int, xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    values = [
        max(0.0, min(1.0, value))
        for value in (center_x, center_y, box_width, box_height)
    ]
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "yolo11s.pt"
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing label files. Do not use after manual review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    categories = ("red_apple", "green_apple", "mixed_apples", "negative")
    image_paths = [
        path
        for category in categories
        for path in sorted((ORGANIZED_ROOT / category).glob("*.jpg"))
    ]
    if not image_paths:
        raise SystemExit(f"No organized images found in {ORGANIZED_ROOT}")

    model = YOLO(str(args.model))
    prediction_paths = [
        path for path in image_paths if path.parent.name != "negative"
    ]
    predictions = iter(
        model.predict(
            source=[str(path) for path in prediction_paths],
            classes=[47],
            conf=args.confidence,
            imgsz=640,
            device=args.device,
            verbose=False,
            stream=True,
        )
    )
    rows: list[dict[str, str | int | float]] = []
    created = 0
    skipped = 0

    for image_path in image_paths:
        category = image_path.parent.name
        label_dir = LABEL_ROOT / category
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            rows.append(
                {
                    "image": str(image_path.relative_to(PROJECT_ROOT)),
                    "category": category,
                    "detections": "existing",
                    "max_confidence": "",
                    "status": "preserved_existing_label",
                }
            )
            if category != "negative":
                next(predictions)
            continue

        lines: list[str] = []
        confidences: list[float] = []
        if category != "negative":
            result = next(predictions)
            image = result.orig_img
            height, width = image.shape[:2]
            for box in result.boxes:
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                if category == "red_apple":
                    class_id = 0
                elif category == "green_apple":
                    class_id = 1
                else:
                    class_id, _red_score, _green_score = color_class(image, xyxy)
                lines.append(yolo_line(class_id, xyxy, width, height))
                confidences.append(float(box.conf[0]))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        created += 1
        rows.append(
            {
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "category": category,
                "detections": len(lines),
                "max_confidence": f"{max(confidences, default=0.0):.3f}",
                "status": "needs_manual_review",
            }
        )

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image",
                "category",
                "detections",
                "max_confidence",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"images={len(image_paths)} labels_created={created} labels_preserved={skipped}")
    print(f"classes: 0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]}")
    print(f"labels={LABEL_ROOT}")
    print(f"report={REPORT_PATH}")
    print("All generated labels still require manual review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
