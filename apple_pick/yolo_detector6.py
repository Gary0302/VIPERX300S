#!/usr/bin/env python3

import os
from pathlib import Path

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data

from ultralytics import YOLO

import cv2
import numpy as np


class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.bridge = CvBridge()

        # ============================================================
        # YOLO11
        # ============================================================
        project_root = Path(__file__).resolve().parents[1]
        model_setting = os.environ.get('YOLO_MODEL', 'models/yolo11s.pt')
        model_path = Path(model_setting).expanduser()
        if not model_path.is_absolute():
            model_path = project_root / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f'找不到 YOLO 模型：{model_path}')

        self.apple_class_id = int(os.environ.get('APPLE_CLASS_ID', '47'))
        self.get_logger().info(f"⏳ 正在載入模型 {model_path}，蘋果類別 ID={self.apple_class_id}...")
        self.model = YOLO(str(model_path))
        self.get_logger().info("✅ 模型載入完成！")

        # ============================================================
        # Camera
        # ============================================================
        self.depth_image = None

        self.camera_model = PinholeCameraModel()

        self.camera_info_received = False

        # ============================================================
        # YOLO parameters
        # ============================================================
        self.confidence_threshold = 0.20

        # ============================================================
        # Image ROI
        # 避免畫面最邊緣的誤偵測
        # ============================================================
        self.roi_x_min_ratio = 0.02
        self.roi_x_max_ratio = 0.98
        self.roi_y_min_ratio = 0.02
        self.roi_y_max_ratio = 0.98

        # ============================================================
        # HSV threshold
        #
        # OpenCV HSV:
        # H = 0 ~ 179
        # S = 0 ~ 255
        # V = 0 ~ 255
        #
        # Red 因為橫跨 Hue 的頭尾，所以需要兩段
        # ============================================================

        self.red_lower_1 = np.array([0, 70, 50])
        self.red_upper_1 = np.array([10, 255, 255])

        self.red_lower_2 = np.array([170, 70, 50])
        self.red_upper_2 = np.array([179, 255, 255])

        self.green_lower = np.array([35, 50, 40])
        self.green_upper = np.array([90, 255, 255])

        # ============================================================
        # ROS subscribers
        # ============================================================

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            qos_profile_sensor_data
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.publisher = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)


    # ================================================================
    # Camera Info
    # ================================================================

    def info_callback(self, msg):

        if not self.camera_info_received:

            self.camera_model.fromCameraInfo(msg)

            self.camera_info_received = True

            self.get_logger().info(
                "✅ 成功接收到相機內參！"
            )


    # ================================================================
    # Depth callback
    # ================================================================

    def depth_callback(self, msg):

        try:

            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                "16UC1"
            )

        except Exception as e:

            self.get_logger().error(
                f"深度圖轉換失敗: {e}"
            )


    # ================================================================
    # HSV 顏色判斷
    # ================================================================

    def classify_apple_color(
        self,
        cv_image,
        x1,
        y1,
        x2,
        y2
    ):

        image_h, image_w = cv_image.shape[:2]

        # ------------------------------------------------------------
        # 不使用整個 Bounding Box
        #
        # 只取中間約 60%
        # 避免 Bounding Box 邊緣包含背景
        # ------------------------------------------------------------

        box_w = x2 - x1
        box_h = y2 - y1

        roi_x1 = int(x1 + box_w * 0.20)
        roi_x2 = int(x2 - box_w * 0.20)

        roi_y1 = int(y1 + box_h * 0.20)
        roi_y2 = int(y2 - box_h * 0.20)

        # 防止超出影像範圍

        roi_x1 = max(0, min(image_w - 1, roi_x1))
        roi_x2 = max(0, min(image_w, roi_x2))

        roi_y1 = max(0, min(image_h - 1, roi_y1))
        roi_y2 = max(0, min(image_h, roi_y2))

        # ROI 無效
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:

            return "unknown", 0.0, 0.0

        apple_roi = cv_image[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        # ------------------------------------------------------------
        # BGR → HSV
        # ------------------------------------------------------------

        hsv = cv2.cvtColor(
            apple_roi,
            cv2.COLOR_BGR2HSV
        )

        # ------------------------------------------------------------
        # RED mask
        # ------------------------------------------------------------

        red_mask_1 = cv2.inRange(
            hsv,
            self.red_lower_1,
            self.red_upper_1
        )

        red_mask_2 = cv2.inRange(
            hsv,
            self.red_lower_2,
            self.red_upper_2
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )

        # ------------------------------------------------------------
        # GREEN mask
        # ------------------------------------------------------------

        green_mask = cv2.inRange(
            hsv,
            self.green_lower,
            self.green_upper
        )

        # ------------------------------------------------------------
        # 計算比例
        # ------------------------------------------------------------

        total_pixels = apple_roi.shape[0] * apple_roi.shape[1]

        if total_pixels == 0:

            return "unknown", 0.0, 0.0

        red_pixels = cv2.countNonZero(
            red_mask
        )

        green_pixels = cv2.countNonZero(
            green_mask
        )

        red_ratio = red_pixels / total_pixels
        green_ratio = green_pixels / total_pixels

        # ------------------------------------------------------------
        # 判斷
        #
        # 不是一定要整顆都是紅色
        # 只要紅色比例明顯大於綠色即可
        # ------------------------------------------------------------

        if red_ratio > green_ratio and red_ratio > 0.08:

            apple_type = "ripe_apple"

        elif green_ratio > red_ratio and green_ratio > 0.08:

            apple_type = "green_apple"

        else:

            apple_type = "unknown"

        return (
            apple_type,
            red_ratio,
            green_ratio
        )


    # ================================================================
    # Depth ROI
    # ================================================================

    def get_median_depth(
        self,
        u,
        v,
        radius=10
    ):

        """
        舊版：
            只看 depth[v, u]

        新版：
            看中心附近 21 x 21 區域
            把有效深度拿出來
            取 median
        """

        if self.depth_image is None:

            return None

        h, w = self.depth_image.shape

        x1 = max(0, u - radius)
        x2 = min(w, u + radius + 1)

        y1 = max(0, v - radius)
        y2 = min(h, v + radius + 1)

        depth_roi = self.depth_image[
            y1:y2,
            x1:x2
        ]

        # 去除 0
        valid_depths = depth_roi[
            depth_roi > 0
        ]

        if valid_depths.size == 0:

            return None

        depth_mm = np.median(
            valid_depths
        )

        return float(depth_mm)


    # ================================================================
    # Color callback
    # ================================================================

    def color_callback(self, msg):

        if (
            self.depth_image is None
            or not self.camera_info_received
        ):

            self.get_logger().info(
                "⏳ 等待影像或相機內參中...",
                throttle_duration_sec=3.0
            )

            return

        try:

            # --------------------------------------------------------
            # ROS Image → OpenCV
            # --------------------------------------------------------

            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                "bgr8"
            )

            # --------------------------------------------------------
            # YOLO11
            #
            # 預設 COCO class 47 = apple；自訂模型可用 APPLE_CLASS_ID 指定
            # --------------------------------------------------------

            results = self.model(
                cv_image,
                verbose=False,
                classes=[self.apple_class_id],
                conf=self.confidence_threshold,
                iou=0.4
            )

            boxes = results[0].boxes

            annotated_frame = results[0].plot()

            ripe_count = 0
            green_count = 0
            unknown_count = 0

            if boxes is not None and len(boxes) > 0:

                h, w = self.depth_image.shape

                self.get_logger().info(
                    f"🔍 YOLO 本幀偵測到 {len(boxes)} 個 apple"
                )

                for box in boxes:

                    # ------------------------------------------------
                    # YOLO confidence
                    # ------------------------------------------------

                    confidence = float(
                        box.conf[0].cpu().item()
                    )

                    # ------------------------------------------------
                    # Bounding Box
                    # ------------------------------------------------

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    # 中心點仍然保留
                    # 主要拿來算 3D 位置

                    u = int(
                        (x1 + x2) / 2
                    )

                    v = int(
                        (y1 + y2) / 2
                    )

                    if not (
                        0 <= u < w
                        and
                        0 <= v < h
                    ):

                        continue

                    # ------------------------------------------------
                    # Whole image ROI filter
                    # ------------------------------------------------

                    roi_x_min = int(
                        w * self.roi_x_min_ratio
                    )

                    roi_x_max = int(
                        w * self.roi_x_max_ratio
                    )

                    roi_y_min = int(
                        h * self.roi_y_min_ratio
                    )

                    roi_y_max = int(
                        h * self.roi_y_max_ratio
                    )

                    if not (
                        roi_x_min <= u <= roi_x_max
                        and
                        roi_y_min <= v <= roi_y_max
                    ):

                        continue

                    # =================================================
                    # HSV COLOR
                    # =================================================

                    (
                        apple_type,
                        red_ratio,
                        green_ratio
                    ) = self.classify_apple_color(
                        cv_image,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    # =================================================
                    # MEDIAN DEPTH
                    # =================================================

                    depth_mm = self.get_median_depth(
                        u,
                        v,
                        radius=10
                    )

                    if depth_mm is None:

                        self.get_logger().warn(
                            f"⚠️ apple conf={confidence:.2f} "
                            f"沒有有效 Depth"
                        )

                        continue

                    depth_m = (
                        depth_mm / 1000.0
                    )

                    # =================================================
                    # Deprojection 2D → 3D
                    # =================================================

                    fx = self.camera_model.fx()
                    fy = self.camera_model.fy()
                    cx = self.camera_model.cx()
                    cy = self.camera_model.cy()

                    x_cam = (
                        (u - cx)
                        * depth_m
                        / fx
                    )

                    y_cam = (
                        (v - cy)
                        * depth_m
                        / fy
                    )

                    z_cam = depth_m

                    # =================================================
                    # Apple type
                    # =================================================

                    if apple_type == "ripe_apple":

                        ripe_count += 1

                        apple_id = ripe_count

                        display_text = (
                            f"Red#{apple_id} "
                            f"{confidence:.2f}"
                        )

                        color_ui = (
                            0,
                            0,
                            255
                        )

                    elif apple_type == "green_apple":

                        green_count += 1

                        apple_id = green_count

                        display_text = (
                            f"Green#{apple_id} "
                            f"{confidence:.2f}"
                        )

                        color_ui = (
                            0,
                            255,
                            0
                        )

                    else:

                        unknown_count += 1

                        apple_id = unknown_count

                        display_text = (
                            f"Unknown "
                            f"{confidence:.2f}"
                        )

                        color_ui = (
                            0,
                            255,
                            255
                        )

                    # =================================================
                    # UI
                    # =================================================

                    cv2.drawMarker(
                        annotated_frame,
                        (u, v),
                        color_ui,
                        cv2.MARKER_CROSS,
                        15,
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        display_text,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color_ui,
                        2
                    )

                    # 顯示 HSV 比例

                    hsv_text = (
                        f"R:{red_ratio:.2f} "
                        f"G:{green_ratio:.2f} "
                        f"D:{depth_m:.2f}m"
                    )

                    cv2.putText(
                        annotated_frame,
                        hsv_text,
                        (x1, min(h - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color_ui,
                        1
                    )

                    # =================================================
                    # TF
                    # =================================================

                    self.publish_target_tf(
                        x_cam,
                        y_cam,
                        z_cam,
                        apple_type,
                        apple_id,
                        msg.header.stamp
                    )

                    # =================================================
                    # Terminal debug
                    # =================================================

                    self.get_logger().info(
                        f"🍎 {apple_type} | "
                        f"conf={confidence:.2f} | "
                        f"red={red_ratio:.2f} | "
                        f"green={green_ratio:.2f} | "
                        f"depth={depth_m:.3f}m"
                    )

            else:

                self.get_logger().info(
                    "❌ YOLO 本幀沒有偵測到 apple",
                    throttle_duration_sec=2.0
                )

            # --------------------------------------------------------
            # Publish annotated image
            # --------------------------------------------------------

            annotated_msg = (
                self.bridge.cv2_to_imgmsg(
                    annotated_frame,
                    "bgr8"
                )
            )

            annotated_msg.header = msg.header

            self.publisher.publish(
                annotated_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"影像處理錯誤: {e}"
            )


    # ================================================================
    # TF
    # ================================================================

    def publish_target_tf(
        self,
        x,
        y,
        z,
        name,
        track_id,
        stamp
    ):

        t = TransformStamped()

        t.header.stamp = stamp

        t.header.frame_id = (
            'camera_color_optical_frame'
        )

        t.child_frame_id = (
            f'target_{name}_{track_id}'
        )

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(
            t
        )


def main(args=None):

    rclpy.init(args=args)

    node = YoloDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
