#!/usr/bin/env python3
"""ROS 2 YOLO detector for the general object picking UI.

This node only reads camera images and publishes annotated images and status.
It does not publish TF transforms or robot commands.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
ANNOTATED_TOPIC = "/object_detector/annotated_image"
STATUS_TOPIC = "/object_detector/status"
UI_STREAM_ENDPOINT = "tcp://127.0.0.1:5560"
BASE_FRAME = "vx300s/base_link"
# Keep only the newest color frame so a slow model never works on old images.
LATEST_FRAME_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class GenericYoloDetector(Node):
    NODE_NAME = "general_object_yolo_detector"

    def __init__(self, model_path: Path | str, target: str, confidence: float) -> None:
        super().__init__(self.NODE_NAME)
        self.bridge = CvBridge()
        self.target = target.strip()
        self.confidence = confidence
        self.last_frame_time: float | None = None
        self.depth_image: np.ndarray | None = None
        self.depth_encoding = ""
        self.camera_intrinsics: tuple[float, float, float, float] | None = None
        self.coordinate_frame = "camera_color_optical_frame"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.zmq_context = zmq.Context()
        self.ui_socket = self.zmq_context.socket(zmq.PUB)
        self.ui_socket.setsockopt(zmq.LINGER, 0)
        self.ui_socket.bind(UI_STREAM_ENDPOINT)

        self.get_logger().info(f"Loading model: {model_path}")
        self.load_model(model_path)
        self.get_logger().info(
            f"Detector ready: target={self.class_name}, confidence={self.confidence:.2f}"
        )

        self.image_publisher = self.create_publisher(Image, ANNOTATED_TOPIC, 10)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(
            Image,
            COLOR_TOPIC,
            self.image_callback,
            LATEST_FRAME_QOS,
        )
        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

    def load_model(self, model_path: Path | str) -> None:
        self.model = YOLO(str(model_path))
        self.class_ids, self.class_name = self._resolve_target(self.target)

    def detect(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Return (detections, annotated image). Subclasses swap the model here."""
        result = self.model.predict(
            frame,
            classes=self.class_ids,
            conf=self.confidence,
            iou=0.45,
            verbose=False,
        )[0]
        detections = []
        count = len(result.boxes) if result.boxes is not None else 0
        for box_index in range(count):
            class_id = int(result.boxes.cls[box_index].cpu().item())
            x1, y1, x2, y2 = result.boxes.xyxy[box_index].cpu().tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(self.model.names[class_id]),
                    "confidence": float(result.boxes.conf[box_index].cpu().item()),
                    "bbox": (x1, y1, x2, y2),
                    "pixel": (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                }
            )
        return detections, result.plot()

    def _resolve_target(self, target: str) -> tuple[list[int], str]:
        names = self.model.names
        items = list(names.items()) if isinstance(names, dict) else list(enumerate(names))
        normalized = target.casefold()
        for class_id, class_name in items:
            if str(class_name).casefold() == normalized:
                return [int(class_id)], str(class_name)

        # A fine-tuned printed-apple model has separate red_apple and
        # green_apple classes.  The existing pick workflow deliberately keeps
        # using the generic "apple" target, so combine both colour classes.
        if normalized == "apple":
            apple_classes = [
                int(class_id)
                for class_id, class_name in items
                if str(class_name).casefold().endswith("_apple")
            ]
            if apple_classes:
                return apple_classes, "apple"

        available = ", ".join(str(name) for _, name in items)
        raise ValueError(
            f"Target '{target}' is not available in this model. Available classes: {available}"
        )

    def depth_callback(self, message: Image) -> None:
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(message, "passthrough")
            self.depth_encoding = message.encoding
        except Exception as error:
            self.get_logger().error(f"Depth conversion failed: {error}")

    def camera_info_callback(self, message: CameraInfo) -> None:
        fx, fy = float(message.k[0]), float(message.k[4])
        cx, cy = float(message.k[2]), float(message.k[5])
        if fx > 0.0 and fy > 0.0:
            self.camera_intrinsics = (fx, fy, cx, cy)
            if message.header.frame_id:
                self.coordinate_frame = message.header.frame_id

    def position_from_pixel(self, u: int, v: int) -> dict[str, float] | None:
        if self.depth_image is None or self.camera_intrinsics is None:
            return None

        height, width = self.depth_image.shape[:2]
        if not 0 <= u < width or not 0 <= v < height:
            return None

        radius = 7
        x1, x2 = max(0, u - radius), min(width, u + radius + 1)
        y1, y2 = max(0, v - radius), min(height, v + radius + 1)
        region = self.depth_image[y1:y2, x1:x2].astype(np.float64)
        valid = region[np.isfinite(region) & (region > 0)]
        if valid.size == 0:
            return None

        depth_value = float(np.median(valid))
        depth_m = depth_value / 1000.0 if "16U" in self.depth_encoding else depth_value
        if not 0.05 <= depth_m <= 10.0:
            return None

        fx, fy, cx, cy = self.camera_intrinsics
        return {
            "x": (u - cx) * depth_m / fx,
            "y": (v - cy) * depth_m / fy,
            "z": depth_m,
        }

    def lookup_base_transform(self):
        try:
            return self.tf_buffer.lookup_transform(
                BASE_FRAME,
                self.coordinate_frame,
                rclpy.time.Time(),
            )
        except Exception:
            return None

    @staticmethod
    def transform_position(position: dict[str, float], transform) -> dict[str, float]:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
        norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if norm == 0.0:
            raise ValueError("TF quaternion has zero length")
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        x, y, z = position["x"], position["y"], position["z"]
        rotated_x = (
            (1.0 - 2.0 * (qy * qy + qz * qz)) * x
            + 2.0 * (qx * qy - qz * qw) * y
            + 2.0 * (qx * qz + qy * qw) * z
        )
        rotated_y = (
            2.0 * (qx * qy + qz * qw) * x
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * y
            + 2.0 * (qy * qz - qx * qw) * z
        )
        rotated_z = (
            2.0 * (qx * qz - qy * qw) * x
            + 2.0 * (qy * qz + qx * qw) * y
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * z
        )
        return {
            "x": rotated_x + translation.x,
            "y": rotated_y + translation.y,
            "z": rotated_z + translation.z,
        }

    def image_callback(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
            detections, annotated = self.detect(frame)
            count = len(detections)
            max_confidence = 0.0
            targets = []
            target_pixel = None
            position_camera_m = None
            position_base_m = None
            base_transform = self.lookup_base_transform()
            if count:
                for box_index, detection in enumerate(detections):
                    confidence = detection["confidence"]
                    class_id = detection["class_id"]
                    detected_class = detection["class_name"]
                    x1, y1, x2, y2 = detection["bbox"]
                    u, v = detection["pixel"]
                    camera_position = self.position_from_pixel(u, v)
                    base_position = None
                    if camera_position is not None and base_transform is not None:
                        base_position = self.transform_position(
                            camera_position, base_transform
                        )
                    targets.append(
                        {
                            "frame_index": box_index,
                            "class_id": class_id,
                            "class_name": detected_class,
                            "confidence": confidence,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                            "target_pixel": {"u": u, "v": v},
                            "position_camera_m": camera_position,
                            "position_base_m": base_position,
                        }
                    )

                targets.sort(key=lambda item: item["confidence"], reverse=True)
                best = targets[0]
                max_confidence = float(best["confidence"])
                target_pixel = best["target_pixel"]
                position_camera_m = best["position_camera_m"]
                position_base_m = best["position_base_m"]

            now = time.monotonic()
            fps = 0.0
            if self.last_frame_time is not None:
                elapsed = now - self.last_frame_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
            self.last_frame_time = now

            cv2.putText(
                annotated,
                f"Target: {self.class_name} | Count: {count}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )
            for rank, target in enumerate(targets, start=1):
                pixel = target["target_pixel"]
                center = (pixel["u"], pixel["v"])
                marker_color = (0, 255, 255) if rank == 1 else (255, 180, 40)
                cv2.drawMarker(
                    annotated,
                    center,
                    marker_color,
                    cv2.MARKER_CROSS,
                    18,
                    2,
                )
                cv2.putText(
                    annotated,
                    f"#{rank} {target['confidence']:.2f}",
                    (max(8, center[0] - 35), max(18, center[1] - 34)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    marker_color,
                    2,
                    cv2.LINE_AA,
                )
                camera_position = target["position_camera_m"]
                if camera_position is not None:
                    position_text = (
                        f"X {camera_position['x']:.3f}  "
                        f"Y {camera_position['y']:.3f}  "
                        f"Z {camera_position['z']:.3f} m"
                    )
                    cv2.putText(
                        annotated,
                        position_text,
                        (max(8, center[0] - 150), max(58, center[1] - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        marker_color,
                        2,
                        cv2.LINE_AA,
                    )

            output = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            output.header = message.header
            self.image_publisher.publish(output)

            status_data = {
                "target": self.class_name,
                "count": count,
                "max_confidence": max_confidence,
                "targets": targets,
                "fps": fps,
                "depth_ready": self.depth_image is not None
                and self.camera_intrinsics is not None,
                "target_pixel": target_pixel,
                "position_camera_m": position_camera_m,
                "position_base_m": position_base_m,
                "coordinate_frame": self.coordinate_frame,
                "base_frame": BASE_FRAME,
                "tf_ready": base_transform is not None,
            }
            status = String()
            status.data = json.dumps(status_data)
            self.status_publisher.publish(status)

            encoded, jpeg = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if encoded:
                try:
                    self.ui_socket.send_multipart(
                        [status.data.encode("utf-8"), jpeg.tobytes()],
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    pass
        except Exception as error:
            self.get_logger().error(f"Image processing failed: {error}")

    def destroy_node(self) -> None:
        self.ui_socket.close(linger=0)
        self.zmq_context.term()
        super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic YOLO ROS 2 detector")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("Confidence must be greater than 0 and no more than 1")

    rclpy.init()
    node: GenericYoloDetector | None = None
    try:
        node = GenericYoloDetector(model_path, args.target, args.confidence)
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
