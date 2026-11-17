#!/usr/bin/env python3
"""Save rate-limited raw ROS 2 color frames for object-detection datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def write_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


class DatasetCaptureNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("object_dataset_capture")
        self.args = args
        self.output_dir = args.output.resolve()
        self.images_dir = self.output_dir / "images"
        self.metadata_path = self.output_dir / "session.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.saved_this_run = 0
        self.existing_count = len(list(self.images_dir.glob("*.jpg")))
        self.next_index = self.existing_count + 1
        self.last_saved_monotonic = 0.0
        self.start_monotonic = time.monotonic()
        self.finished = False
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.metadata = self._load_metadata()
        self.subscription = self.create_subscription(Image, args.topic, self.on_image, 10)
        self._write_metadata("recording")
        emit(
            "ready",
            output=str(self.output_dir),
            topic=args.topic,
            existing_images=self.existing_count,
            max_images=args.max_images,
        )

    def _load_metadata(self) -> dict:
        if self.metadata_path.is_file():
            try:
                value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "schema_version": 1,
            "session_name": self.output_dir.name,
            "created_at": self.started_at,
            "topic": self.args.topic,
            "scene_label": self.args.scene_label,
            "interval_seconds": self.args.interval,
            "jpeg_quality": self.args.jpeg_quality,
            "runs": [],
        }

    def _write_metadata(self, status: str, resolution: list[int] | None = None) -> None:
        self.metadata.update(
            {
                "status": status,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "topic": self.args.topic,
                "scene_label": self.args.scene_label,
                "interval_seconds": self.args.interval,
                "jpeg_quality": self.args.jpeg_quality,
                "images_saved": self.existing_count + self.saved_this_run,
            }
        )
        if resolution is not None:
            self.metadata["resolution"] = resolution
        write_json_atomic(self.metadata_path, self.metadata)

    def on_image(self, message: Image) -> None:
        if self.finished:
            return
        now = time.monotonic()
        if self.saved_this_run and now - self.last_saved_monotonic < self.args.interval:
            return

        free_gb = shutil.disk_usage(self.output_dir).free / (1024**3)
        if free_gb < self.args.min_free_gb:
            self.finish(
                "disk_limit",
                error=f"Free disk space {free_gb:.2f} GB is below the safety limit.",
            )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"frame_{self.next_index:06d}_{timestamp}.jpg"
            image_path = self.images_dir / filename
            written = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
            )
            if not written:
                raise RuntimeError("cv2.imwrite returned false")
        except Exception as error:
            emit("error", message=f"Could not save image: {error}")
            return

        self.saved_this_run += 1
        self.next_index += 1
        self.last_saved_monotonic = now
        height, width = frame.shape[:2]
        self._write_metadata("recording", [width, height])
        emit(
            "saved",
            count=self.saved_this_run,
            total=self.existing_count + self.saved_this_run,
            file=filename,
            free_gb=round(free_gb, 2),
        )
        if self.saved_this_run >= self.args.max_images:
            self.finish("completed")

    def finish(self, status: str, error: str | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        run = {
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "scene_label": self.args.scene_label,
            "images_saved": self.saved_this_run,
            "status": status,
        }
        self.metadata.setdefault("runs", []).append(run)
        self._write_metadata(status)
        if error:
            emit("error", message=error)
        emit(
            "complete",
            status=status,
            count=self.saved_this_run,
            total=self.existing_count + self.saved_this_run,
            output=str(self.output_dir),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--scene-label", default="apple")
    args = parser.parse_args()
    if args.interval <= 0 or args.max_images <= 0:
        parser.error("--interval and --max-images must be greater than zero")
    if (
        not 1 <= args.jpeg_quality <= 100
        or args.min_free_gb < 0
        or args.startup_timeout <= 0
    ):
        parser.error("invalid JPEG quality or free-space limit")
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = DatasetCaptureNode(args)

    def stop_capture(_signum, _frame) -> None:
        node.finish("stopped")

    signal.signal(signal.SIGTERM, stop_capture)
    signal.signal(signal.SIGINT, stop_capture)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
            if (
                not node.saved_this_run
                and time.monotonic() - node.start_monotonic > args.startup_timeout
            ):
                node.finish(
                    "no_image",
                    error=f"No image was received from {args.topic} within "
                    f"{args.startup_timeout:.0f} seconds.",
                )
    except KeyboardInterrupt:
        node.finish("stopped")
    finally:
        if not node.finished:
            node.finish("stopped")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
