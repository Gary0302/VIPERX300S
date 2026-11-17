#!/usr/bin/env python3
"""Generic object picking UI prototype.

Robot motion requires a saved pose, a successful MoveIt plan, and explicit
operator confirmation.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QProcess, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "printed_apple" / "raw"
MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine"}
COLOR_TOPIC = "/camera/camera/color/image_raw"
ANNOTATED_TOPIC = "/object_detector/annotated_image"
STATUS_TOPIC = "/object_detector/status"
ROBOT_COMMAND_ENDPOINT = "tcp://127.0.0.1:5561"
ROBOT_STATUS_ENDPOINT = "tcp://127.0.0.1:5562"
BATCH_STABLE_DETECTION_COUNT = 6
BATCH_INITIAL_TARGET_TIMEOUT_SECONDS = 10.0
BATCH_REMAINING_TARGET_TIMEOUT_SECONDS = 10.0
APPLE_TARGETS = ("apple", "red_apple", "green_apple")


class VisionStreamWorker(QThread):
    """Receive the local detector stream without blocking the Qt event loop."""

    image_ready = pyqtSignal(QImage)
    status_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = True

    def run(self) -> None:
        context = None
        socket = None
        try:
            import cv2
            import numpy as np
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.connect("tcp://127.0.0.1:5560")
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)

            while self._running:
                events = dict(poller.poll(100))
                if socket not in events:
                    continue
                status_bytes, jpeg = socket.recv_multipart()
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format_RGB888,
                ).copy()
                self.image_ready.emit(image)
                try:
                    self.status_ready.emit(json.loads(status_bytes.decode("utf-8")))
                except json.JSONDecodeError as error:
                    self.error.emit(f"Invalid detector status: {error}")
        except Exception as error:
            self.error.emit(f"Vision stream failed: {error}")
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def stop(self) -> None:
        self._running = False


class RobotStatusWorker(QThread):
    status_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = True

    def run(self) -> None:
        context = None
        socket = None
        try:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.connect(ROBOT_STATUS_ENDPOINT)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while self._running:
                events = dict(poller.poll(100))
                if socket in events:
                    self.status_ready.emit(socket.recv_json())
        except Exception as error:
            self.error.emit(f"Robot status stream failed: {error}")
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def stop(self) -> None:
        self._running = False


class RobotCommandWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, command: dict) -> None:
        super().__init__()
        self.command = command

    def run(self) -> None:
        context = None
        socket = None
        response = {"success": False, "message": "Robot command failed."}
        try:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 120000)
            socket.connect(ROBOT_COMMAND_ENDPOINT)
            socket.send_json(self.command)
            response = socket.recv_json()
        except Exception as error:
            response = {"success": False, "message": str(error)}
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
        response["command"] = self.command.get("command")
        self.result_ready.emit(response)


class StatusBadge(QLabel):
    """Small status indicator used by the prototype."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(112)
        self.set_state("Offline", "offline")

    def set_state(self, text: str, state: str) -> None:
        colors = {
            "offline": ("#475569", "#e2e8f0"),
            "ready": ("#047857", "#d1fae5"),
            "warning": ("#b45309", "#fef3c7"),
            "error": ("#b91c1c", "#fee2e2"),
        }
        foreground, background = colors.get(state, colors["offline"])
        self.setText(f"{self.name} - {text}")
        self.setStyleSheet(
            f"background: {background}; color: {foreground};"
            "border-radius: 10px; padding: 5px 10px; font-weight: 600;"
        )


class ObjectPickPrototype(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("General Object Pick - UI Prototype")
        self.resize(1360, 860)
        self.setMinimumSize(1080, 700)
        self.camera_process: QProcess | None = None
        self.detector_process: QProcess | None = None
        self.robot_process: QProcess | None = None
        self.capture_process: QProcess | None = None
        self.capture_output_buffer = ""
        self.capture_session_dir: Path | None = None
        self.capture_count = 0
        self.image_worker: VisionStreamWorker | None = None
        self.robot_status_worker: RobotStatusWorker | None = None
        self.robot_command_workers: set[RobotCommandWorker] = set()
        self.last_image: QImage | None = None
        self.latest_detection_status: dict | None = None
        self.depth_ready_announced = False
        self.camera_started_by_ui = False
        self.camera_process_group_id: int | None = None
        self.last_robot_result_sequence = 0
        self.last_robot_history_sequence = 0
        self.last_robot_log_message = ""
        self.last_robot_log_time = 0.0
        self.robot_ready = False
        self.gripper_ready = False
        self.robot_active = False
        self.pick_plan_ready = False
        self.holding_object = False
        self.planned_pick_target: dict | None = None
        self.target_history = deque(maxlen=12)
        self.last_pick_completion_sequence = 0
        self.batch_active = False
        self.batch_state = "idle"
        self.batch_completed = 0
        self.batch_target_label = "apple"
        self.batch_target_history = deque(maxlen=BATCH_STABLE_DETECTION_COUNT)
        self.batch_current_target: dict | None = None
        self.batch_processed_targets: list[dict[str, float]] = []
        self.batch_next_detection_time = 0.0
        self.batch_wait_started = 0.0
        self._build_ui()
        self._apply_style()
        self.refresh_models()
        self.log("UI started. Robot motion requires a successful plan and confirmation.")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(22, 18, 22, 18)
        root_layout.setSpacing(14)

        title_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("General Object Pick Console")
        title.setObjectName("title")
        subtitle = QLabel("Select a model, confirm the target, and monitor the picking workflow")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        title_row.addLayout(title_block)
        title_row.addStretch()
        prototype = QLabel("PLAN FIRST - CONFIRM BEFORE MOTION")
        prototype.setObjectName("prototypeBadge")
        title_row.addWidget(prototype)
        root_layout.addLayout(title_row)

        status_row = QHBoxLayout()
        self.status_badges = {
            name: StatusBadge(name)
            for name in ("Camera", "Detector", "TF", "MoveIt", "Gripper")
        }
        for badge in self.status_badges.values():
            status_row.addWidget(badge)
        status_row.addStretch()
        root_layout.addLayout(status_row)

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._camera_panel(), stretch=7)
        body.addWidget(self._control_panel(), stretch=4)
        root_layout.addLayout(body, stretch=1)

        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(145)
        log_layout.addWidget(self.log_view)
        root_layout.addWidget(log_group)

        self.setCentralWidget(root)

    def _camera_panel(self) -> QGroupBox:
        group = QGroupBox("Camera and Detection View")
        layout = QVBoxLayout(group)

        self.camera_view = QLabel(
            "Waiting for a camera or annotated image\n\n"
            "Planned topic: /object_detector/annotated_image\n"
            "Detection boxes, labels, and confidence scores will appear here"
        )
        self.camera_view.setObjectName("cameraView")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_view.setMinimumHeight(390)
        layout.addWidget(self.camera_view)

        info = QHBoxLayout()
        self.image_topic_label = QLabel("Image: Offline")
        self.fps_label = QLabel("FPS: --")
        self.selected_detection_label = QLabel("Target: --")
        info.addWidget(self.image_topic_label)
        info.addStretch()
        info.addWidget(self.fps_label)
        info.addSpacing(20)
        info.addWidget(self.selected_detection_label)
        layout.addLayout(info)
        return group

    def _control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        vision = QGroupBox("1. Detection Settings")
        grid = QGridLayout(vision)
        grid.addWidget(QLabel("Backend"), 0, 0)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("YOLO (Phase 1)", "yolo")
        self.backend_combo.addItem("SAM 3 (Planned)", "sam3")
        self.backend_combo.currentIndexChanged.connect(self.backend_changed)
        grid.addWidget(self.backend_combo, 0, 1, 1, 2)

        grid.addWidget(QLabel("Model"), 1, 0)
        self.model_combo = QComboBox()
        grid.addWidget(self.model_combo, 1, 1)
        refresh = QPushButton("Rescan")
        refresh.clicked.connect(self.refresh_models)
        grid.addWidget(refresh, 1, 2)

        grid.addWidget(QLabel("Target object"), 2, 0)
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.addItems(
            ["apple", "red_apple", "green_apple", "bottle", "tweezers"]
        )
        self.target_combo.setCurrentText("apple")
        grid.addWidget(self.target_combo, 2, 1, 1, 2)

        self.backend_note = QLabel("YOLO is used first for real-time detection and workflow integration.")
        self.backend_note.setWordWrap(True)
        self.backend_note.setObjectName("hint")
        grid.addWidget(self.backend_note, 3, 0, 1, 3)
        layout.addWidget(vision)

        grasp = QGroupBox("2. Grasp Settings")
        grasp_grid = QGridLayout(grasp)
        grasp_grid.addWidget(QLabel("Grasp profile"), 0, 0)
        self.grasp_profile_combo = QComboBox()
        self.grasp_profile_combo.addItem("apple (Existing reference)", "apple")
        self.grasp_profile_combo.addItem("bottle (Needs calibration)", "bottle")
        self.grasp_profile_combo.addItem("tweezers (Planned)", "tweezers")
        self.grasp_profile_combo.currentIndexChanged.connect(
            lambda _index: self._update_pick_controls()
        )
        grasp_grid.addWidget(self.grasp_profile_combo, 0, 1)
        grasp_grid.addWidget(QLabel("Target position"), 1, 0)
        self.coordinate_label = QLabel("X --  /  Y --  /  Z --")
        self.coordinate_label.setObjectName("coordinate")
        grasp_grid.addWidget(self.coordinate_label, 1, 1)
        grasp_grid.addWidget(QLabel("Coordinate frame"), 2, 0)
        self.coordinate_frame_label = QLabel("--")
        grasp_grid.addWidget(self.coordinate_frame_label, 2, 1)
        grasp_grid.addWidget(QLabel("Orientation"), 3, 0)
        grasp_grid.addWidget(QLabel("Not calculated"), 3, 1)
        layout.addWidget(grasp)

        observation = QGroupBox("3. Observation Pose")
        observation_grid = QGridLayout(observation)
        observation_grid.addWidget(QLabel("Pose name"), 0, 0)
        self.pose_name_edit = QLineEdit("observation_default")
        observation_grid.addWidget(self.pose_name_edit, 0, 1, 1, 2)
        observation_grid.addWidget(QLabel("Saved pose"), 1, 0)
        self.pose_combo = QComboBox()
        self.pose_combo.currentIndexChanged.connect(
            lambda _index: self.execute_pose_button.setEnabled(False)
        )
        observation_grid.addWidget(self.pose_combo, 1, 1, 1, 2)
        self.save_pose_button = QPushButton("Save Current")
        self.plan_pose_button = QPushButton("Plan Only")
        self.execute_pose_button = QPushButton("Execute Plan")
        self.save_pose_button.setEnabled(False)
        self.plan_pose_button.setEnabled(False)
        self.execute_pose_button.setEnabled(False)
        self.save_pose_button.clicked.connect(self.save_observation_pose)
        self.plan_pose_button.clicked.connect(self.plan_observation_pose)
        self.execute_pose_button.clicked.connect(self.execute_observation_pose)
        observation_grid.addWidget(self.save_pose_button, 2, 0)
        observation_grid.addWidget(self.plan_pose_button, 2, 1)
        observation_grid.addWidget(self.execute_pose_button, 2, 2)
        layout.addWidget(observation)

        manual_gripper = QGroupBox("4. Manual Gripper")
        manual_gripper_layout = QGridLayout(manual_gripper)
        manual_gripper_layout.addWidget(QLabel("Last command"), 0, 0)
        self.gripper_state_label = QLabel("Unknown")
        manual_gripper_layout.addWidget(self.gripper_state_label, 0, 1)
        self.open_gripper_button = QPushButton("Open Gripper")
        self.close_gripper_button = QPushButton("Close Gripper")
        self.open_gripper_button.setEnabled(False)
        self.close_gripper_button.setEnabled(False)
        self.open_gripper_button.clicked.connect(self.open_gripper)
        self.close_gripper_button.clicked.connect(self.close_gripper)
        manual_gripper_layout.addWidget(self.open_gripper_button, 1, 0)
        manual_gripper_layout.addWidget(self.close_gripper_button, 1, 1)
        layout.addWidget(manual_gripper)

        workflow = QGroupBox("5. Workflow")
        workflow_layout = QGridLayout(workflow)
        workflow_layout.addWidget(QLabel("Pick mode"), 0, 0)
        self.pick_mode_combo = QComboBox()
        self.pick_mode_combo.addItem("Single Pick (Keep Holding)", "single")
        self.pick_mode_combo.addItem("Batch Pick and Release", "batch")
        self.pick_mode_combo.currentIndexChanged.connect(self.pick_mode_changed)
        workflow_layout.addWidget(self.pick_mode_combo, 0, 1)
        workflow_layout.addWidget(QLabel("Maximum batch picks"), 1, 0)
        self.batch_max_spin = QSpinBox()
        self.batch_max_spin.setRange(1, 20)
        self.batch_max_spin.setValue(3)
        workflow_layout.addWidget(self.batch_max_spin, 1, 1)
        self.vision_button = QPushButton("Start Vision")
        self.robot_button = QPushButton("Start Robot")
        self.preview_button = QPushButton("Preview Target")
        self.plan_pick_button = QPushButton("Plan Straight Pick")
        self.pick_button = QPushButton("Confirm and Pick")
        self.batch_pick_button = QPushButton("Confirm and Start Batch")
        self.stop_button = QPushButton("Stop Vision / Cancel Motion")
        self.emergency_button = QPushButton("Emergency Stop")
        self.plan_pick_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.batch_pick_button.setEnabled(False)
        self.emergency_button.setObjectName("emergencyButton")

        self.vision_button.clicked.connect(self.start_vision)
        self.robot_button.clicked.connect(self.start_robot_manager)
        self.preview_button.clicked.connect(self.preview_target)
        self.plan_pick_button.clicked.connect(self.plan_straight_pick)
        self.pick_button.clicked.connect(self.confirm_and_pick)
        self.batch_pick_button.clicked.connect(self.start_batch_pick)
        self.stop_button.clicked.connect(self.stop_workflow)
        self.emergency_button.clicked.connect(self.emergency_stop)

        workflow_layout.addWidget(self.vision_button, 2, 0)
        workflow_layout.addWidget(self.robot_button, 2, 1)
        workflow_layout.addWidget(self.preview_button, 3, 0)
        workflow_layout.addWidget(self.plan_pick_button, 3, 1)
        workflow_layout.addWidget(self.pick_button, 4, 0, 1, 2)
        workflow_layout.addWidget(self.batch_pick_button, 5, 0, 1, 2)
        workflow_layout.addWidget(self.stop_button, 6, 0)
        workflow_layout.addWidget(self.emergency_button, 6, 1)
        layout.addWidget(workflow)

        dataset = QGroupBox("6. Dataset Capture")
        dataset_grid = QGridLayout(dataset)
        dataset_grid.addWidget(QLabel("Session"), 0, 0)
        self.capture_session_edit = QLineEdit(
            datetime.now().strftime("printed_apple_%Y%m%d_%H%M%S")
        )
        dataset_grid.addWidget(self.capture_session_edit, 0, 1, 1, 2)
        dataset_grid.addWidget(QLabel("Capture category"), 1, 0)
        self.capture_label_combo = QComboBox()
        self.capture_label_combo.addItems(
            ["apple", "red_apple", "green_apple", "mixed_apples", "negative"]
        )
        dataset_grid.addWidget(self.capture_label_combo, 1, 1, 1, 2)
        dataset_grid.addWidget(QLabel("Interval"), 2, 0)
        self.capture_interval_spin = QDoubleSpinBox()
        self.capture_interval_spin.setRange(0.1, 10.0)
        self.capture_interval_spin.setSingleStep(0.1)
        self.capture_interval_spin.setValue(0.5)
        self.capture_interval_spin.setSuffix(" s")
        dataset_grid.addWidget(self.capture_interval_spin, 2, 1)
        dataset_grid.addWidget(QLabel("Maximum"), 2, 2)
        self.capture_max_spin = QSpinBox()
        self.capture_max_spin.setRange(1, 5000)
        self.capture_max_spin.setValue(200)
        self.capture_max_spin.setSuffix(" images")
        dataset_grid.addWidget(self.capture_max_spin, 2, 3)
        self.capture_status_label = QLabel("Stopped — 0 images")
        self.capture_status_label.setObjectName("hint")
        dataset_grid.addWidget(self.capture_status_label, 3, 0, 1, 4)
        self.capture_snapshot_button = QPushButton("Save Snapshot")
        self.capture_start_button = QPushButton("Start Auto Capture")
        self.capture_stop_button = QPushButton("Stop Capture")
        self.capture_stop_button.setEnabled(False)
        self.capture_snapshot_button.clicked.connect(
            lambda: self.start_dataset_capture(single=True)
        )
        self.capture_start_button.clicked.connect(
            lambda: self.start_dataset_capture(single=False)
        )
        self.capture_stop_button.clicked.connect(self.stop_dataset_capture)
        dataset_grid.addWidget(self.capture_snapshot_button, 4, 0)
        dataset_grid.addWidget(self.capture_start_button, 4, 1, 1, 2)
        dataset_grid.addWidget(self.capture_stop_button, 4, 3)
        note = QLabel("Saves raw color images without YOLO boxes. Annotation is a later step.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        dataset_grid.addWidget(note, 5, 0, 1, 4)
        layout.addWidget(dataset)
        layout.addStretch()
        panel.setMinimumHeight(930)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        return scroll

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f1f5f9; color: #172033; font-size: 14px; }
            QLabel#title { font-size: 27px; font-weight: 700; color: #0f172a; }
            QLabel#subtitle { color: #64748b; font-size: 14px; }
            QLabel#prototypeBadge {
                background: #fff7ed; color: #c2410c; border: 1px solid #fdba74;
                border-radius: 12px; padding: 7px 12px; font-weight: 700;
            }
            QGroupBox {
                background: #ffffff; border: 1px solid #dbe3ef; border-radius: 10px;
                margin-top: 10px; padding-top: 12px; font-weight: 700;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QLabel#cameraView {
                background: #111827; color: #cbd5e1; border: 1px solid #334155;
                border-radius: 8px; font-size: 15px;
            }
            QLabel#hint { color: #64748b; font-size: 12px; }
            QLabel#coordinate { color: #2563eb; font-weight: 700; }
            QComboBox, QTextEdit {
                background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
                padding: 7px;
            }
            QPushButton {
                background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 7px;
                padding: 9px 12px; font-weight: 600;
            }
            QPushButton:hover { background: #cbd5e1; }
            QPushButton:disabled { color: #94a3b8; background: #f1f5f9; }
            QPushButton#emergencyButton { background: #dc2626; color: white; border: none; }
            QPushButton#emergencyButton:hover { background: #b91c1c; }
            """
        )

    def refresh_models(self) -> None:
        current = self.model_combo.currentText() if hasattr(self, "model_combo") else ""
        models = []
        if MODEL_DIR.is_dir():
            models = sorted(
                path.name
                for path in MODEL_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
            )

        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            if current in models:
                self.model_combo.setCurrentText(current)
            self.log(f"Found {len(models)} model(s) in {MODEL_DIR}.")
        else:
            self.model_combo.addItem("No compatible model found")
            self.log(f"No compatible model found in {MODEL_DIR}.")

    def backend_changed(self) -> None:
        backend = self.backend_combo.currentData()
        if backend == "sam3":
            self.backend_note.setText(
                "SAM 3 supports text prompts and mask segmentation. It will run as a separate "
                "service to avoid dependency conflicts with ROS 2 Humble."
            )
        else:
            self.backend_note.setText(
                "YOLO is used first for real-time detection and workflow integration."
            )

    def preview_target(self) -> None:
        target = self.target_combo.currentText().strip()
        model = self.model_combo.currentText()
        if not target:
            QMessageBox.warning(self, "Missing Target", "Enter the object you want to pick.")
            return
        self.selected_detection_label.setText(f"Target: {target}")
        status = self.latest_detection_status or {}
        position = status.get("position_base_m") or status.get("position_camera_m")
        if position is None:
            self.log(
                f"Preview: model={model}, target={target}. "
                "No valid target position is available."
            )
            return
        frame = (
            status.get("base_frame", "vx300s/base_link")
            if status.get("position_base_m") is not None
            else status.get("coordinate_frame", "camera_color_optical_frame")
        )
        self.log(
            f"Preview: {target} in {frame}: "
            f"X={position['x']:.3f}, Y={position['y']:.3f}, Z={position['z']:.3f} m."
        )

    def start_vision(self) -> None:
        if self.detector_process is not None:
            self.log("Vision is already running.")
            return
        if self.backend_combo.currentData() != "yolo":
            QMessageBox.warning(self, "Backend Unavailable", "SAM 3 is not connected yet.")
            return

        model_path = MODEL_DIR / self.model_combo.currentText()
        target = self.target_combo.currentText().strip()
        if not model_path.is_file():
            QMessageBox.warning(self, "Missing Model", f"Model not found: {model_path}")
            return
        if not target:
            QMessageBox.warning(self, "Missing Target", "Enter the object you want to detect.")
            return

        self.start_button_state(True)
        self.status_badges["Camera"].set_state("Starting", "warning")
        self.status_badges["Detector"].set_state("Loading", "warning")
        self.image_topic_label.setText("Image: Waiting")
        self.camera_view.setText("Starting RealSense and loading the YOLO model...")

        self.image_worker = VisionStreamWorker()
        self.image_worker.image_ready.connect(self.update_image)
        self.image_worker.status_ready.connect(self.update_detection_status)
        self.image_worker.error.connect(self.vision_error)
        self.image_worker.start()

        if self.camera_topic_exists():
            self.log(f"Using the existing camera topic: {COLOR_TOPIC}")
        else:
            self.start_camera_process()

        self.detector_process = QProcess(self)
        self.detector_process.setProcessChannelMode(QProcess.MergedChannels)
        self.detector_process.readyReadStandardOutput.connect(self.read_detector_output)
        self.detector_process.finished.connect(self.detector_finished)
        self.detector_process.errorOccurred.connect(self.detector_process_error)
        detector_path = Path(__file__).with_name("generic_yolo_detector.py")
        detector_confidence = (
            0.55 if model_path.name == "printed_apple_yolo11s.pt" else 0.25
        )
        self.detector_process.start(
            "python3",
            [
                str(detector_path),
                "--model",
                str(model_path),
                "--target",
                target,
                "--confidence",
                f"{detector_confidence:.2f}",
            ],
        )
        self.log(
            f"Starting YOLO: model={model_path.name}, target={target}, "
            f"confidence={detector_confidence:.2f}"
        )

    def camera_topic_exists(self) -> bool:
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
            return result.returncode == 0 and COLOR_TOPIC in result.stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            self.log(f"Could not inspect ROS topics: {error}")
            return False

    def start_camera_process(self) -> None:
        self.camera_process = QProcess(self)
        self.camera_process.setProcessChannelMode(QProcess.MergedChannels)
        self.camera_process.readyReadStandardOutput.connect(self.read_camera_output)
        self.camera_process.finished.connect(self.camera_finished)
        self.camera_process.errorOccurred.connect(self.camera_process_error)
        self.camera_process.start(
            "setsid", [str(PROJECT_ROOT / "apple_pick" / "start_camera.sh")]
        )
        if self.camera_process.waitForStarted(3000):
            self.camera_process_group_id = int(self.camera_process.processId())
        self.camera_started_by_ui = True
        self.log("No active color topic was found. Starting RealSense.")

    def read_camera_output(self) -> None:
        if self.camera_process is None:
            return
        output = bytes(self.camera_process.readAllStandardOutput()).decode(errors="replace")
        for line in output.splitlines():
            if "No valid configuration file found" in line:
                self.log("Camera: no saved RealSense config; using defaults.")
            elif "ERROR" in line or "WARN" in line:
                self.log(f"Camera: {line.strip()}")

    def read_detector_output(self) -> None:
        if self.detector_process is None:
            return
        output = bytes(self.detector_process.readAllStandardOutput()).decode(errors="replace")
        for line in output.splitlines():
            line = line.strip()
            if line and ("ready:" in line or "ERROR" in line or "Traceback" in line):
                self.log(f"Detector: {line}")

    def camera_finished(self, exit_code: int) -> None:
        if self.camera_started_by_ui and self.detector_process is not None:
            self.stop_process_group(self.camera_process_group_id)
            self.status_badges["Camera"].set_state("Error", "error")
            self.log(f"Camera process stopped with exit code {exit_code}.")
        self.camera_process = None
        self.camera_started_by_ui = False
        self.camera_process_group_id = None

    def camera_process_error(self, _error) -> None:
        if self.camera_process is not None:
            self.log(f"Camera process error: {self.camera_process.errorString()}")
        self.status_badges["Camera"].set_state("Error", "error")

    def detector_process_error(self, _error) -> None:
        if self.detector_process is not None:
            self.log(f"Detector process error: {self.detector_process.errorString()}")
        self.status_badges["Detector"].set_state("Error", "error")

    def detector_finished(self, exit_code: int) -> None:
        was_running = self.detector_process is not None
        self._finish_batch("Detector stopped; batch workflow cancelled.", error=True)
        self.detector_process = None
        if was_running:
            if self.image_worker is not None:
                self.image_worker.stop()
                self.image_worker.wait(2000)
                self.image_worker = None
            if self.camera_process is not None and self.camera_started_by_ui:
                camera_process = self.camera_process
                self.camera_process = None
                self.camera_started_by_ui = False
                self.stop_qprocess(
                    camera_process, process_group_id=self.camera_process_group_id
                )
                self.camera_process_group_id = None
            if exit_code != 0:
                self.status_badges["Detector"].set_state("Error", "error")
                self.log(
                    f"Detector stopped with exit code {exit_code}. "
                    "Check whether the model contains the selected target class."
                )
        self.start_button_state(False)
        self._update_pick_controls()

    def update_image(self, image: QImage) -> None:
        self.last_image = image
        self.camera_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.status_badges["Camera"].set_state("Ready", "ready")
        self.status_badges["Detector"].set_state("Ready", "ready")
        self.image_topic_label.setText(f"Image: {ANNOTATED_TOPIC}")

    def update_detection_status(self, status: dict) -> None:
        self.latest_detection_status = status
        fps = float(status.get("fps", 0.0))
        count = int(status.get("count", 0))
        confidence = float(status.get("max_confidence", 0.0))
        target = str(status.get("target", "--"))
        self.fps_label.setText(f"FPS: {fps:.1f}")
        if count:
            self.selected_detection_label.setText(
                f"Target: {target} x{count} ({confidence:.2f})"
            )
        else:
            self.selected_detection_label.setText(f"Target: {target} x0")

        depth_ready = bool(status.get("depth_ready", False))
        if depth_ready and not self.depth_ready_announced:
            self.depth_ready_announced = True
            self.log("Aligned depth and camera intrinsics are ready.")

        tf_ready = bool(status.get("tf_ready", False))
        self.status_badges["TF"].set_state(
            "Ready" if tf_ready else "Waiting", "ready" if tf_ready else "warning"
        )
        position = status.get("position_base_m") or status.get("position_camera_m")
        frame = str(
            status.get("base_frame", "vx300s/base_link")
            if status.get("position_base_m") is not None
            else status.get("coordinate_frame", "--")
        )
        self.coordinate_frame_label.setText(frame if depth_ready else "Depth unavailable")
        if position is not None:
            self.coordinate_label.setText(
                f"X {position['x']:.3f} / Y {position['y']:.3f} / Z {position['z']:.3f} m"
            )
        else:
            self.coordinate_label.setText("X --  /  Y --  /  Z --")

        base_position = status.get("position_base_m")
        if count == 1 and tf_ready and base_position is not None:
            self.target_history.append(
                tuple(float(base_position[key]) for key in ("x", "y", "z"))
            )
        else:
            self.target_history.clear()
        self._update_batch_detection(status)
        self._update_pick_controls()

    def vision_error(self, message: str) -> None:
        self.log(message)
        self.status_badges["Detector"].set_state("Error", "error")

    def start_button_state(self, running: bool) -> None:
        self.vision_button.setEnabled(not running)
        self.backend_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.target_combo.setEnabled(not running)

    def start_robot_manager(self) -> None:
        if self.robot_process is not None:
            self.log("Robot manager is already running.")
            return
        self.status_badges["MoveIt"].set_state("Connecting", "warning")
        self.status_badges["Gripper"].set_state("Checking", "warning")
        self.robot_button.setEnabled(False)

        self.robot_status_worker = RobotStatusWorker()
        self.robot_status_worker.status_ready.connect(self.update_robot_status)
        self.robot_status_worker.error.connect(self.log)
        self.robot_status_worker.start()

        self.robot_process = QProcess(self)
        self.robot_process.setProcessChannelMode(QProcess.MergedChannels)
        self.robot_process.readyReadStandardOutput.connect(self.read_robot_output)
        self.robot_process.finished.connect(self.robot_manager_finished)
        self.robot_process.errorOccurred.connect(self.robot_manager_error)
        manager_path = Path(__file__).with_name("robot_pose_manager.py")
        self.robot_process.start("python3", [str(manager_path)])
        self.log("Connecting the UI to the existing MoveIt system.")

    def read_robot_output(self) -> None:
        if self.robot_process is None:
            return
        output = bytes(self.robot_process.readAllStandardOutput()).decode(errors="replace")
        for line in output.splitlines():
            if "ERROR" in line or "WARN" in line or "Traceback" in line:
                self.log(f"Robot manager: {line.strip()}")

    def robot_manager_error(self, _error) -> None:
        if self.robot_process is not None:
            self.log(f"Robot manager error: {self.robot_process.errorString()}")
        self.status_badges["MoveIt"].set_state("Error", "error")

    def robot_manager_finished(self, exit_code: int) -> None:
        was_running = self.robot_process is not None
        self._finish_batch("Robot manager stopped; batch workflow cancelled.", error=True)
        self.robot_process = None
        if self.robot_status_worker is not None:
            self.robot_status_worker.stop()
            self.robot_status_worker.wait(2000)
            self.robot_status_worker = None
        self.status_badges["MoveIt"].set_state("Offline", "offline")
        self.status_badges["Gripper"].set_state("Offline", "offline")
        self.robot_ready = False
        self.gripper_ready = False
        self.robot_active = False
        self.pick_plan_ready = False
        self.save_pose_button.setEnabled(False)
        self.plan_pose_button.setEnabled(False)
        self.execute_pose_button.setEnabled(False)
        self.plan_pick_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.open_gripper_button.setEnabled(False)
        self.close_gripper_button.setEnabled(False)
        self.gripper_state_label.setText("Unknown")
        self.robot_button.setEnabled(True)
        self._update_pick_controls()
        if was_running and exit_code != 0:
            self.log(f"Robot manager stopped with exit code {exit_code}.")

    def update_robot_status(self, status: dict) -> None:
        moveit_ready = bool(status.get("moveit_ready"))
        joints_ready = bool(status.get("joint_state_ready"))
        arm_ready = bool(status.get("arm_controller_ready"))
        gripper_ready = bool(status.get("gripper_ready"))
        self.gripper_ready = gripper_ready
        robot_ready = moveit_ready and joints_ready and arm_ready
        self.robot_ready = robot_ready and gripper_ready
        self.status_badges["MoveIt"].set_state(
            "Ready" if robot_ready else "Waiting",
            "ready" if robot_ready else "warning",
        )
        self.status_badges["Gripper"].set_state(
            "Ready" if gripper_ready else "Waiting",
            "ready" if gripper_ready else "warning",
        )
        active = status.get("active_command") is not None
        self.robot_active = active
        self.pick_plan_ready = bool(status.get("pick_plan_ready"))
        self.holding_object = bool(status.get("holding_object"))
        state = str(status.get("gripper_state", "unknown"))
        finger_position = status.get("finger_position_m")
        if finger_position is None:
            self.gripper_state_label.setText(state.capitalize())
        else:
            self.gripper_state_label.setText(
                f"{state.capitalize()} ({float(finger_position) * 1000:.1f} mm)"
            )
        self.save_pose_button.setEnabled(robot_ready and not active)
        self.plan_pose_button.setEnabled(
            robot_ready and not active and self.pose_combo.count() > 0
        )

        poses = [str(name) for name in status.get("poses", [])]
        current = self.pose_combo.currentText()
        existing = [self.pose_combo.itemText(i) for i in range(self.pose_combo.count())]
        if poses != existing:
            self.pose_combo.clear()
            self.pose_combo.addItems(poses)
            if current in poses:
                self.pose_combo.setCurrentText(current)

        history = status.get("result_history", [])
        if isinstance(history, list) and history:
            for entry in history:
                history_sequence = int(entry.get("sequence", 0))
                if history_sequence > self.last_robot_history_sequence:
                    message = str(entry.get("message", ""))
                    if message:
                        self.log_robot_message(message)
                    self.last_robot_history_sequence = history_sequence

        sequence = int(status.get("result_sequence", 0))
        if (
            not history
            and self.last_robot_result_sequence
            and sequence > self.last_robot_result_sequence
        ):
            message = str(status.get("last_result", ""))
            if message:
                self.log_robot_message(message)
        self.last_robot_result_sequence = max(self.last_robot_result_sequence, sequence)

        pick_completion_sequence = int(status.get("pick_completion_sequence", 0))
        if pick_completion_sequence > self.last_pick_completion_sequence:
            self.last_pick_completion_sequence = pick_completion_sequence
            if self.batch_active and self.batch_state == "executing":
                if bool(status.get("last_pick_success")):
                    self._batch_cycle_completed()
                else:
                    self._finish_batch(
                        "Batch stopped because the pick-and-release cycle failed.",
                        error=True,
                    )
        self._update_pick_controls()

    def send_robot_command(self, command: dict) -> None:
        if self.robot_process is None:
            self.log("Start Robot before sending a robot command.")
            return
        worker = RobotCommandWorker(command)
        self.robot_command_workers.add(worker)
        worker.result_ready.connect(self.robot_command_result)
        worker.finished.connect(lambda: self.robot_command_workers.discard(worker))
        worker.start()

    def robot_command_result(self, response: dict) -> None:
        message = str(response.get("message", "No response message."))
        command = response.get("command")
        self.log_robot_message(message)
        if command == "save_pose" and response.get("success"):
            poses = [str(name) for name in response.get("poses", [])]
            self.pose_combo.clear()
            self.pose_combo.addItems(poses)
            name = self.pose_name_edit.text().strip()
            if name in poses:
                self.pose_combo.setCurrentText(name)
        elif command == "plan_pose":
            self.execute_pose_button.setEnabled(bool(response.get("success")))
        elif command == "execute_pose":
            self.execute_pose_button.setEnabled(False)
        elif command == "plan_pick":
            self.pick_plan_ready = bool(response.get("success"))
            if not self.pick_plan_ready:
                self.planned_pick_target = None
            if self.batch_active and self.batch_state == "planning":
                if response.get("success") and self.batch_current_target is not None:
                    self.batch_state = "executing"
                    self.send_robot_command(
                        {
                            "command": "execute_pick",
                            "target": dict(self.batch_current_target),
                            "release_after_return": True,
                        }
                    )
                else:
                    self._finish_batch(
                        "Batch stopped because MoveIt could not plan the selected target.",
                        error=True,
                    )
            self._update_pick_controls()
        elif command == "execute_pick":
            self.pick_button.setEnabled(False)
            if (
                self.batch_active
                and self.batch_state == "executing"
                and not response.get("success")
            ):
                self._finish_batch(
                    "Batch stopped because robot execution was not accepted.",
                    error=True,
                )
        elif command in {"open_gripper", "close_gripper"}:
            self.open_gripper_button.setEnabled(False)
            self.close_gripper_button.setEnabled(False)

    def save_observation_pose(self) -> None:
        name = self.pose_name_edit.text().strip()
        self.send_robot_command({"command": "save_pose", "name": name})

    def log_robot_message(self, message: str) -> None:
        now = time.monotonic()
        if message == self.last_robot_log_message and now - self.last_robot_log_time < 1.5:
            return
        self.last_robot_log_message = message
        self.last_robot_log_time = now
        self.log(f"Robot: {message}")

    def plan_observation_pose(self) -> None:
        name = self.pose_combo.currentText().strip()
        if not name:
            self.log("Save or select an observation pose first.")
            return
        self.execute_pose_button.setEnabled(False)
        self.log(f"Requesting a plan-only path to '{name}'.")
        self.send_robot_command({"command": "plan_pose", "name": name})

    def execute_observation_pose(self) -> None:
        name = self.pose_combo.currentText().strip()
        answer = QMessageBox.question(
            self,
            "Execute Physical Motion",
            f"Move the physical arm to observation pose '{name}'?\n\n"
            "Confirm that the RViz plan is clear and the workspace has no obstacles.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.log("Observation-pose execution cancelled by the user.")
            return
        self.execute_pose_button.setEnabled(False)
        self.send_robot_command({"command": "execute_pose", "name": name})

    def _target_is_stable(self) -> bool:
        if len(self.target_history) < 10:
            return False
        axes = list(zip(*self.target_history))
        return max(max(axis) - min(axis) for axis in axes) <= 0.015

    @staticmethod
    def _positions_are_stable(history: deque, tolerance: float = 0.015) -> bool:
        if len(history) < history.maxlen:
            return False
        axes = list(zip(*history))
        return max(max(axis) - min(axis) for axis in axes) <= tolerance

    def _batch_candidates(self, status: dict) -> list[dict]:
        candidates = []
        raw_targets = status.get("targets", [])
        if not isinstance(raw_targets, list):
            return candidates
        for target in raw_targets:
            if not isinstance(target, dict):
                continue
            position = target.get("position_base_m")
            if not isinstance(position, dict):
                continue
            try:
                point = {key: float(position[key]) for key in ("x", "y", "z")}
                confidence = float(target.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            class_name = str(target.get("class_name", "")).strip().lower()
            if (
                self.batch_target_label != "apple"
                and class_name
                and class_name != self.batch_target_label
            ):
                continue
            if any(
                math.dist(
                    tuple(point[key] for key in ("x", "y", "z")),
                    tuple(processed[key] for key in ("x", "y", "z")),
                )
                <= 0.04
                for processed in self.batch_processed_targets
            ):
                continue
            candidates.append(
                {
                    "position": point,
                    "confidence": confidence,
                    "class_name": class_name or self.batch_target_label,
                }
            )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates

    def _update_batch_detection(self, status: dict) -> None:
        if (
            not self.batch_active
            or self.batch_state != "waiting_target"
            or self.robot_active
            or time.monotonic() < self.batch_next_detection_time
        ):
            return
        if not bool(status.get("tf_ready")):
            return
        if str(status.get("target", "")).strip().lower() != self.batch_target_label:
            return

        candidates = self._batch_candidates(status)
        now = time.monotonic()
        if (
            self.batch_completed > 0
            and now - self.batch_wait_started
            >= BATCH_REMAINING_TARGET_TIMEOUT_SECONDS
        ):
            self._finish_batch(
                f"Batch completed after {self.batch_completed} object(s); "
                "no remaining stable target was detected."
            )
            return
        if (
            self.batch_completed == 0
            and now - self.batch_wait_started >= BATCH_INITIAL_TARGET_TIMEOUT_SECONDS
        ):
            self._finish_batch(
                "No stable target was detected within "
                f"{BATCH_INITIAL_TARGET_TIMEOUT_SECONDS:.0f} seconds.",
                error=True,
            )
            return
        if not candidates:
            self.batch_target_history.clear()
            return

        if self.batch_target_history:
            anchor = self.batch_target_history[-1]
            selected = min(
                candidates,
                key=lambda item: math.dist(
                    anchor,
                    tuple(item["position"][key] for key in ("x", "y", "z")),
                ),
            )
            selected_point = tuple(
                selected["position"][key] for key in ("x", "y", "z")
            )
            if math.dist(anchor, selected_point) > 0.05:
                self.batch_target_history.clear()
                selected = candidates[0]
        else:
            selected = candidates[0]

        point = tuple(selected["position"][key] for key in ("x", "y", "z"))
        self.batch_target_history.append(point)
        if not self._positions_are_stable(self.batch_target_history):
            return

        axes = list(zip(*self.batch_target_history))
        self.batch_current_target = {
            key: sum(axis) / len(axis)
            for key, axis in zip(("x", "y", "z"), axes)
        }
        self.planned_pick_target = dict(self.batch_current_target)
        self.batch_state = "planning"
        self.batch_target_history.clear()
        self.log(
            f"Batch target {self.batch_completed + 1}: selected stable "
            f"{selected['class_name']} at "
            f"X={self.batch_current_target['x']:.3f}, "
            f"Y={self.batch_current_target['y']:.3f}, "
            f"Z={self.batch_current_target['z']:.3f} m "
            f"(confidence {selected['confidence']:.2f})."
        )
        self.send_robot_command(
            {
                "command": "plan_pick",
                "target": dict(self.batch_current_target),
                "target_label": self.batch_target_label,
                "profile": "apple",
            }
        )

    def _update_pick_controls(self) -> None:
        if not hasattr(self, "plan_pick_button"):
            return
        detection = self.latest_detection_status or {}
        single_mode = self.pick_mode_combo.currentData() == "single"
        batch_mode = self.pick_mode_combo.currentData() == "batch"
        selected_target = self.target_combo.currentText().strip().lower()
        correct_target = (
            selected_target in APPLE_TARGETS
            and str(detection.get("target", "")).strip().lower() == selected_target
        )
        profile_ready = self.grasp_profile_combo.currentData() == "apple"
        target_ready = (
            int(detection.get("count", 0)) == 1
            and detection.get("position_base_m") is not None
            and bool(detection.get("tf_ready"))
            and self._target_is_stable()
            and correct_target
            and profile_ready
        )
        self.plan_pick_button.setEnabled(
            single_mode
            and not self.batch_active
            and self.robot_ready
            and not self.robot_active
            and target_ready
        )
        planned_target_valid = False
        current_position = detection.get("position_base_m")
        if self.planned_pick_target is not None and current_position is not None:
            planned_target_valid = max(
                abs(float(current_position[key]) - float(self.planned_pick_target[key]))
                for key in ("x", "y", "z")
            ) <= 0.015
        self.pick_button.setEnabled(
            single_mode
            and not self.batch_active
            and self.robot_ready
            and not self.robot_active
            and self.pick_plan_ready
            and target_ready
            and planned_target_valid
        )
        batch_targets = detection.get("targets", [])
        if not isinstance(batch_targets, list):
            batch_targets = []
        batch_target_available = any(
            isinstance(target, dict)
            and isinstance(target.get("position_base_m"), dict)
            for target in batch_targets
        )
        self.batch_pick_button.setEnabled(
            batch_mode
            and not self.batch_active
            and self.robot_ready
            and not self.robot_active
            and self.detector_process is not None
            and bool(detection.get("tf_ready"))
            and correct_target
            and profile_ready
            and batch_target_available
        )
        self.pick_mode_combo.setEnabled(not self.batch_active)
        self.batch_max_spin.setEnabled(not self.batch_active)
        manual_gripper_ready = (
            self.robot_process is not None
            and self.gripper_ready
            and not self.robot_active
            and not self.batch_active
        )
        self.open_gripper_button.setEnabled(manual_gripper_ready)
        self.close_gripper_button.setEnabled(manual_gripper_ready)

    def pick_mode_changed(self, _index: int) -> None:
        self.pick_plan_ready = False
        self.planned_pick_target = None
        self.execute_pose_button.setEnabled(False)
        self._update_pick_controls()

    def start_batch_pick(self) -> None:
        if self.batch_active:
            return
        if self.pick_mode_combo.currentData() != "batch":
            self.log("Select Batch Pick and Release mode first.")
            return
        selected_target = self.target_combo.currentText().strip().lower()
        if selected_target not in APPLE_TARGETS:
            self.log(
                "Batch mode supports apple, red_apple, or green_apple targets only."
            )
            return
        if not self.robot_ready or self.robot_active:
            self.log("Wait until MoveIt and Gripper are ready before starting batch mode.")
            return

        maximum = self.batch_max_spin.value()
        display_target = selected_target.replace("_", " ")
        answer = QMessageBox.question(
            self,
            "Start Automatic Batch Pick",
            f"Automatically pick and release up to {maximum} "
            f"{display_target}(s)?\n\n"
            "Each cycle selects the highest-confidence stable target, picks it, retreats, "
            "returns to the arm position used for planning, and opens the gripper there. "
            "After release, old detection data is discarded and the scene is detected again.\n\n"
            "Confirm that a safe container is below the return position and released objects "
            "will not remain in the pick area.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.log("Batch pick cancelled by the user.")
            return

        self.batch_active = True
        self.batch_state = "waiting_target"
        self.batch_completed = 0
        self.batch_target_label = selected_target
        self.batch_current_target = None
        self.batch_processed_targets.clear()
        self.batch_target_history.clear()
        self.batch_next_detection_time = time.monotonic()
        self.batch_wait_started = time.monotonic()
        self.pick_plan_ready = False
        self.planned_pick_target = None
        self.log(
            f"Batch started: target={self.batch_target_label}, maximum {maximum}. "
            "Waiting for "
            f"{BATCH_STABLE_DETECTION_COUNT} fresh stable detections."
        )
        self._update_pick_controls()

    def _batch_cycle_completed(self) -> None:
        if not self.batch_active or self.batch_current_target is None:
            return
        self.batch_processed_targets.append(dict(self.batch_current_target))
        self.batch_completed += 1
        maximum = self.batch_max_spin.value()
        self.log(
            f"Batch cycle {self.batch_completed}/{maximum} completed; object released at "
            "the return position. Discarding the old detection snapshot."
        )
        if self.batch_completed >= maximum:
            self._finish_batch(
                f"Batch completed the requested {self.batch_completed} pick(s)."
            )
            return

        self.batch_state = "waiting_target"
        self.batch_current_target = None
        self.batch_target_history.clear()
        self.pick_plan_ready = False
        self.planned_pick_target = None
        self.batch_next_detection_time = time.monotonic() + 1.5
        self.batch_wait_started = self.batch_next_detection_time
        self.log("Waiting for a fresh scene and selecting the next stable target.")

    def _finish_batch(self, message: str, error: bool = False) -> None:
        if not self.batch_active:
            return
        self.batch_active = False
        self.batch_state = "idle"
        self.batch_current_target = None
        self.batch_target_history.clear()
        self.batch_next_detection_time = 0.0
        self.batch_wait_started = 0.0
        self.pick_plan_ready = False
        self.planned_pick_target = None
        self.log(("Batch error: " if error else "Batch: ") + message)
        self._update_pick_controls()

    def plan_straight_pick(self) -> None:
        status = self.latest_detection_status or {}
        target = status.get("position_base_m")
        if target is None or not self._target_is_stable():
            self.log("Wait for one stable target with depth and TF before planning.")
            return
        self.pick_plan_ready = False
        self.planned_pick_target = {
            key: float(target[key]) for key in ("x", "y", "z")
        }
        self.pick_button.setEnabled(False)
        self.log(
            f"Planning {self.target_combo.currentText().strip().lower()} pick: pregrasp, "
            "3 cm approach, close, retreat, and return."
        )
        self.send_robot_command(
            {
                "command": "plan_pick",
                "target": dict(self.planned_pick_target),
                "target_label": self.target_combo.currentText().strip().lower(),
                "profile": self.grasp_profile_combo.currentData(),
            }
        )

    def confirm_and_pick(self) -> None:
        target = (self.latest_detection_status or {}).get("position_base_m") or {}
        answer = QMessageBox.question(
            self,
            "Execute Straight Pick",
            "Execute the reviewed physical pick motion?\n\n"
            f"Last visible target: X={target.get('x', 0):.3f}, "
            f"Y={target.get('y', 0):.3f}, Z={target.get('z', 0):.3f} m\n"
            "The gripper will open first. The arm will move to pregrasp, wait 2.5 s, "
            "approach 3 cm, wait 2.5 s, close, retreat, and return to its start pose. "
            "Check whether the object was actually grasped.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.log("Straight-pick execution cancelled by the user.")
            return
        self.pick_button.setEnabled(False)
        self.send_robot_command(
            {
                "command": "execute_pick",
                "target": {key: float(target[key]) for key in ("x", "y", "z")},
            }
        )

    def open_gripper(self) -> None:
        answer = QMessageBox.question(
            self,
            "Open Physical Gripper",
            "Open the physical gripper at its current position?\n\n"
            "If it is holding an object, support the object or place it over a safe surface first.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.open_gripper_button.setEnabled(False)
            self.close_gripper_button.setEnabled(False)
            self.send_robot_command({"command": "open_gripper"})

    def close_gripper(self) -> None:
        answer = QMessageBox.question(
            self,
            "Close Physical Gripper",
            "Close the physical gripper at its current position?\n\n"
            "Keep fingers, cables, and unintended objects out of the gripper.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.open_gripper_button.setEnabled(False)
            self.close_gripper_button.setEnabled(False)
            self.send_robot_command({"command": "close_gripper"})

    def start_dataset_capture(self, single: bool = False) -> None:
        if self.capture_process is not None:
            self.log("Dataset capture is already running.")
            return
        if not self.camera_topic_exists():
            QMessageBox.warning(
                self,
                "Camera Topic Unavailable",
                f"No raw color stream was found at:\n{COLOR_TOPIC}\n\n"
                "Press Start Vision first, then try again.",
            )
            return

        session_name = self.capture_session_edit.text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_name):
            QMessageBox.warning(
                self,
                "Invalid Session Name",
                "Use only English letters, numbers, underscores, and hyphens.",
            )
            return

        dataset_root = DATASET_ROOT.resolve()
        output_dir = (dataset_root / session_name).resolve()
        if output_dir.parent != dataset_root:
            QMessageBox.warning(self, "Invalid Output", "Dataset output path is unsafe.")
            return

        self.capture_session_dir = output_dir
        self.capture_count = 0
        self.capture_output_buffer = ""
        self.capture_process = QProcess(self)
        self.capture_process.setProcessChannelMode(QProcess.MergedChannels)
        self.capture_process.readyReadStandardOutput.connect(self.read_capture_output)
        self.capture_process.finished.connect(self.capture_finished)
        self.capture_process.errorOccurred.connect(self.capture_process_error)
        capture_path = Path(__file__).with_name("dataset_capture.py")
        max_images = 1 if single else self.capture_max_spin.value()
        self.capture_process.start(
            "python3",
            [
                str(capture_path),
                "--output",
                str(output_dir),
                "--topic",
                COLOR_TOPIC,
                "--interval",
                str(self.capture_interval_spin.value()),
                "--max-images",
                str(max_images),
                "--scene-label",
                self.capture_label_combo.currentText(),
            ],
        )
        self.capture_snapshot_button.setEnabled(False)
        self.capture_start_button.setEnabled(False)
        self.capture_stop_button.setEnabled(True)
        self.capture_session_edit.setEnabled(False)
        self.capture_status_label.setText("Starting raw image capture...")
        mode = "one snapshot" if single else f"up to {max_images} images"
        self.log(f"Dataset capture starting: {mode}, session={session_name}.")

    def read_capture_output(self) -> None:
        if self.capture_process is None:
            return
        output = bytes(self.capture_process.readAllStandardOutput()).decode(
            errors="replace"
        )
        self.capture_output_buffer += output
        lines = self.capture_output_buffer.split("\n")
        self.capture_output_buffer = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if "Traceback" in line or "ERROR" in line:
                    self.log(f"Dataset capture: {line}")
                continue
            event_name = event.get("event")
            if event_name == "ready":
                existing = int(event.get("existing_images", 0))
                self.capture_status_label.setText(
                    f"Recording — {existing} existing images"
                )
                self.log(f"Saving raw images to {event.get('output', '--')}.")
            elif event_name == "saved":
                self.capture_count = int(event.get("count", 0))
                total = int(event.get("total", self.capture_count))
                free_gb = float(event.get("free_gb", 0.0))
                self.capture_status_label.setText(
                    f"Recording — {self.capture_count} new / {total} total "
                    f"({free_gb:.1f} GB free)"
                )
                if self.capture_count == 1 or self.capture_count % 25 == 0:
                    self.log(
                        f"Dataset capture saved {self.capture_count} new image(s); "
                        f"{total} total in this session."
                    )
            elif event_name == "error":
                self.log(f"Dataset capture error: {event.get('message', 'unknown error')}")
            elif event_name == "complete":
                total = int(event.get("total", self.capture_count))
                status = str(event.get("status", "completed"))
                label = (
                    "Completed"
                    if status == "completed"
                    else status.replace("_", " ").title()
                )
                self.capture_status_label.setText(
                    f"{label} — {self.capture_count} new / {total} total"
                )

    def capture_finished(self, exit_code: int, _exit_status=None) -> None:
        self.read_capture_output()
        process_existed = self.capture_process is not None
        process = self.capture_process
        self.capture_process = None
        self.capture_snapshot_button.setEnabled(True)
        self.capture_start_button.setEnabled(True)
        self.capture_stop_button.setEnabled(False)
        self.capture_session_edit.setEnabled(True)
        if process_existed:
            if exit_code == 0:
                self.log(
                    f"Dataset capture stopped after {self.capture_count} new image(s)."
                )
            else:
                self.capture_status_label.setText(f"Capture failed (exit {exit_code})")
                self.log(f"Dataset capture stopped with exit code {exit_code}.")
        if process is not None:
            process.deleteLater()

    def capture_process_error(self, _error) -> None:
        if self.capture_process is not None:
            self.capture_status_label.setText("Capture process error")
            self.log(f"Dataset capture: {self.capture_process.errorString()}")
            if self.capture_process.state() == QProcess.NotRunning:
                self.capture_finished(-1)

    def stop_dataset_capture(self) -> None:
        if self.capture_process is None:
            return
        process = self.capture_process
        self.capture_status_label.setText("Stopping capture...")
        self.stop_qprocess(process)

    def stop_workflow(self) -> None:
        self._finish_batch("Cancelled by the operator.")
        if self.robot_process is not None:
            self.send_robot_command({"command": "cancel"})
        self.stop_vision()

    def stop_vision(self) -> None:
        self._finish_batch("Vision stopped; batch workflow cancelled.", error=True)
        self.stop_dataset_capture()
        if self.detector_process is not None:
            detector_process = self.detector_process
            self.detector_process = None
            self.stop_qprocess(detector_process)
        if self.image_worker is not None:
            self.image_worker.stop()
            self.image_worker.wait(2000)
            self.image_worker = None
        if self.camera_process is not None and self.camera_started_by_ui:
            camera_process = self.camera_process
            self.camera_process = None
            self.camera_started_by_ui = False
            self.stop_qprocess(
                camera_process, process_group_id=self.camera_process_group_id
            )
            self.camera_process_group_id = None

        self.last_image = None
        self.latest_detection_status = None
        self.target_history.clear()
        self.planned_pick_target = None
        self.depth_ready_announced = False
        self.camera_view.clear()
        self.camera_view.setText("Vision stopped. Press Start Vision to reconnect.")
        self.status_badges["Camera"].set_state("Offline", "offline")
        self.status_badges["Detector"].set_state("Offline", "offline")
        self.status_badges["TF"].set_state("Offline", "offline")
        self.image_topic_label.setText("Image: Offline")
        self.fps_label.setText("FPS: --")
        self.coordinate_label.setText("X --  /  Y --  /  Z --")
        self.coordinate_frame_label.setText("--")
        self.start_button_state(False)
        self._update_pick_controls()
        self.log("Vision workflow stopped.")

    @staticmethod
    def stop_process_group(process_group_id: int | None, force: bool = False) -> None:
        if process_group_id is None:
            return
        try:
            os.killpg(process_group_id, signal.SIGKILL if force else signal.SIGINT)
        except ProcessLookupError:
            pass

    def stop_qprocess(
        self, process: QProcess, process_group_id: int | None = None
    ) -> None:
        if process_group_id is not None:
            self.stop_process_group(process_group_id)
        else:
            process.terminate()
        if not process.waitForFinished(3000):
            if process_group_id is not None:
                self.stop_process_group(process_group_id, force=True)
            else:
                process.kill()
            process.waitForFinished(1000)

    def prototype_action(self, action: str) -> None:
        self.log(f"{action} is a UI preview action. ROS 2 is not connected yet.")

    def emergency_stop(self) -> None:
        if self.robot_process is not None:
            self.send_robot_command({"command": "cancel"})
        self.stop_vision()
        self.log("Emergency Stop requested MoveIt cancellation and stopped vision.")
        QMessageBox.information(
            self,
            "Stop Requested",
            "MoveIt cancellation was requested and vision stopped. Use the physical emergency "
            "stop or cut motor power if motion does not stop.",
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.last_image is not None:
            self.camera_view.setPixmap(
                QPixmap.fromImage(self.last_image).scaled(
                    self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def closeEvent(self, event) -> None:
        self.stop_vision()
        if self.robot_process is not None:
            robot_process = self.robot_process
            self.robot_process = None
            self.stop_qprocess(robot_process)
        if self.robot_status_worker is not None:
            self.robot_status_worker.stop()
            self.robot_status_worker.wait(2000)
            self.robot_status_worker = None
        event.accept()

    def log(self, message: str) -> None:
        if not hasattr(self, "log_view"):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")


def main() -> int:
    app = QApplication(sys.argv)
    window = ObjectPickPrototype()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
