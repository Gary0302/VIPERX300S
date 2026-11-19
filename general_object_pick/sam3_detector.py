#!/usr/bin/env python3
"""ROS 2 SAM 3 detector for the general object picking UI.

SAM 3 replaces YOLO here. You give it a word such as "apple" and it finds
every matching object and draws its exact outline (a mask). Everything else
(depth, TF, status messages, the UI video stream) is reused from
generic_yolo_detector.py, so the UI and the robot code do not change.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from transformers import Sam3Model, Sam3Processor

from generic_yolo_detector import GenericYoloDetector


LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sam3"
DEFAULT_MODEL = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else "facebook/sam3"


class Sam3Detector(GenericYoloDetector):
    NODE_NAME = "general_object_sam3_detector"

    def load_model(self, model_path: Path | str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Sam3Processor.from_pretrained(str(model_path))
        self.model = Sam3Model.from_pretrained(str(model_path), dtype=self.dtype)
        self.model.to(self.device).eval()

        # "red_apple" -> text prompt "red apple". The class name stays
        # "red_apple" so the UI colour filter keeps working.
        self.class_name = self.target
        self.class_ids = [0]
        self.prompt = self.target.replace("_", " ")
        self.text_inputs = self.processor(text=self.prompt, return_tensors="pt").to(self.device)
        self.get_logger().info(f"SAM 3 prompt: '{self.prompt}' on {self.device}")

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(
            pixel_values=image_inputs["pixel_values"].to(self.dtype),
            input_ids=self.text_inputs["input_ids"],
            attention_mask=self.text_inputs["attention_mask"],
        )
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence,
            mask_threshold=0.5,
            target_sizes=[frame.shape[:2]],
        )[0]

        detections = []
        annotated = frame.copy()
        for score, mask in zip(result["scores"].tolist(), result["masks"].cpu().numpy()):
            mask = mask.astype(bool)
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            # Aim at the middle of the outline; if that point is not on the
            # object (e.g. scissors), use the object pixel closest to it.
            u, v = int(xs.mean()), int(ys.mean())
            if not mask[v, u]:
                nearest = np.argmin((xs - u) ** 2 + (ys - v) ** 2)
                u, v = int(xs[nearest]), int(ys[nearest])
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            detections.append(
                {
                    "class_id": 0,
                    "class_name": self.class_name,
                    "confidence": float(score),
                    "bbox": (x1, y1, x2, y2),
                    "pixel": (u, v),
                }
            )

            annotated[mask] = (0.5 * annotated[mask] + (0, 110, 0)).astype(np.uint8)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(
                annotated,
                f"{self.class_name} {score:.2f}",
                (int(x1), max(16, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
        return detections, annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3 ROS 2 detector")
    parser.add_argument("--model", default=os.environ.get("SAM3_MODEL", DEFAULT_MODEL))
    parser.add_argument("--target", required=True, help='Words to look for, e.g. "apple"')
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("Confidence must be greater than 0 and no more than 1")

    rclpy.init()
    node: Sam3Detector | None = None
    try:
        node = Sam3Detector(args.model, args.target, args.confidence)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
