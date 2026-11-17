#!/usr/bin/env python3
"""Small PyQt5 reviewer for red/green printed-apple YOLO annotations."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "printed_apple"
ORGANIZED_ROOT = DATASET_ROOT / "organized"
LABEL_ROOT = DATASET_ROOT / "labels"
REVIEW_PATH = DATASET_ROOT / "annotation_review.json"
CLASS_NAMES = ("red_apple", "green_apple")
CLASS_COLORS = (QColor(239, 68, 68), QColor(34, 197, 94))
CATEGORIES = ("red_apple", "green_apple", "mixed_apples", "negative")


class AnnotationCanvas(QWidget):
    boxes_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.image = QImage()
        self.boxes: list[dict[str, float | int]] = []
        self.selected_index = -1
        self.active_class = 0
        self.drawing_start: QPointF | None = None
        self.drawing_end: QPointF | None = None

    def set_content(
        self, image: QImage, boxes: list[dict[str, float | int]]
    ) -> None:
        self.image = image
        self.boxes = boxes
        self.selected_index = -1
        self.drawing_start = None
        self.drawing_end = None
        self.update()

    def image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        scale = min(
            self.width() / self.image.width(), self.height() / self.image.height()
        )
        width = self.image.width() * scale
        height = self.image.height() * scale
        return QRectF(
            (self.width() - width) / 2.0,
            (self.height() - height) / 2.0,
            width,
            height,
        )

    def box_rect(self, box: dict[str, float | int]) -> QRectF:
        image_rect = self.image_rect()
        return QRectF(
            image_rect.x() + (float(box["x"]) - float(box["w"]) / 2) * image_rect.width(),
            image_rect.y() + (float(box["y"]) - float(box["h"]) / 2) * image_rect.height(),
            float(box["w"]) * image_rect.width(),
            float(box["h"]) * image_rect.height(),
        )

    def widget_to_normalized(self, point: QPointF) -> QPointF:
        rect = self.image_rect()
        x = min(1.0, max(0.0, (point.x() - rect.x()) / rect.width()))
        y = min(1.0, max(0.0, (point.y() - rect.y()) / rect.height()))
        return QPointF(x, y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 23, 42))
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image")
            return
        image_rect = self.image_rect()
        painter.drawPixmap(image_rect.toRect(), QPixmap.fromImage(self.image))
        painter.setFont(QFont("Sans", 10, QFont.Bold))
        for index, box in enumerate(self.boxes):
            class_id = int(box["class_id"])
            color = CLASS_COLORS[class_id] if class_id in (0, 1) else QColor(251, 191, 36)
            pen = QPen(QColor(250, 204, 21) if index == self.selected_index else color)
            pen.setWidth(4 if index == self.selected_index else 2)
            painter.setPen(pen)
            rect = self.box_rect(box)
            painter.drawRect(rect)
            label = CLASS_NAMES[class_id] if class_id in (0, 1) else f"class {class_id}"
            text_rect = QRectF(rect.x(), max(image_rect.y(), rect.y() - 22), 130, 22)
            painter.fillRect(text_rect, QColor(15, 23, 42, 210))
            painter.drawText(text_rect, Qt.AlignCenter, label)

        if self.drawing_start is not None and self.drawing_end is not None:
            painter.setPen(QPen(CLASS_COLORS[self.active_class], 2, Qt.DashLine))
            painter.drawRect(QRectF(self.drawing_start, self.drawing_end).normalized())

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.image.isNull():
            return
        point = QPointF(event.pos())
        if not self.image_rect().contains(point):
            return
        if not (event.modifiers() & Qt.ShiftModifier):
            for index in range(len(self.boxes) - 1, -1, -1):
                if self.box_rect(self.boxes[index]).contains(point):
                    self.selected_index = index
                    self.selection_changed.emit(index)
                    self.update()
                    return
        self.selected_index = -1
        self.selection_changed.emit(-1)
        self.drawing_start = point
        self.drawing_end = point
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.drawing_start is not None:
            rect = self.image_rect()
            point = QPointF(
                min(rect.right(), max(rect.left(), event.pos().x())),
                min(rect.bottom(), max(rect.top(), event.pos().y())),
            )
            self.drawing_end = point
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.drawing_start is None:
            return
        start = self.widget_to_normalized(self.drawing_start)
        end = self.widget_to_normalized(self.drawing_end or self.drawing_start)
        self.drawing_start = None
        self.drawing_end = None
        left, right = sorted((start.x(), end.x()))
        top, bottom = sorted((start.y(), end.y()))
        if right - left >= 0.01 and bottom - top >= 0.01:
            self.boxes.append(
                {
                    "class_id": self.active_class,
                    "x": (left + right) / 2.0,
                    "y": (top + bottom) / 2.0,
                    "w": right - left,
                    "h": bottom - top,
                }
            )
            self.selected_index = len(self.boxes) - 1
            self.boxes_changed.emit()
            self.selection_changed.emit(self.selected_index)
        self.update()

    def delete_selected(self) -> None:
        if 0 <= self.selected_index < len(self.boxes):
            del self.boxes[self.selected_index]
            self.selected_index = -1
            self.boxes_changed.emit()
            self.selection_changed.emit(-1)
            self.update()

    def set_selected_class(self, class_id: int) -> None:
        self.active_class = class_id
        if 0 <= self.selected_index < len(self.boxes):
            self.boxes[self.selected_index]["class_id"] = class_id
            self.boxes_changed.emit()
        self.update()

    def clear_boxes(self) -> None:
        if self.boxes:
            self.boxes.clear()
            self.selected_index = -1
            self.boxes_changed.emit()
            self.selection_changed.emit(-1)
            self.update()


class DatasetAnnotator(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Printed Apple YOLO Annotation Review")
        self.resize(1280, 820)
        self.all_images = [
            path
            for category in CATEGORIES
            for path in sorted((ORGANIZED_ROOT / category).glob("*.jpg"))
        ]
        self.review = self.load_review()
        self.filtered_images: list[Path] = []
        self.current_index = -1
        self.current_path: Path | None = None
        self.dirty = False
        self._build_ui()
        self.apply_filter()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        top.addWidget(QLabel("Category"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All", "all")
        for category in CATEGORIES:
            self.filter_combo.addItem(category, category)
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        top.addWidget(self.filter_combo)
        self.unreviewed_check = QCheckBox("Unreviewed only")
        self.unreviewed_check.stateChanged.connect(self.apply_filter)
        top.addWidget(self.unreviewed_check)
        top.addStretch()
        self.progress_label = QLabel("0 / 0")
        top.addWidget(self.progress_label)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        self.canvas = AnnotationCanvas()
        self.canvas.boxes_changed.connect(self.boxes_changed)
        self.canvas.selection_changed.connect(self.selection_changed)
        splitter.addWidget(self.canvas)

        side = QFrame()
        side.setMinimumWidth(300)
        side_layout = QVBoxLayout(side)
        self.filename_label = QLabel("--")
        self.filename_label.setWordWrap(True)
        side_layout.addWidget(self.filename_label)
        self.status_label = QLabel("Not reviewed")
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(QLabel("Draw / selected class"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("red_apple (0)", 0)
        self.class_combo.addItem("green_apple (1)", 1)
        self.class_combo.currentIndexChanged.connect(self.class_changed)
        side_layout.addWidget(self.class_combo)
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.list_selection_changed)
        side_layout.addWidget(self.box_list, stretch=1)

        delete_button = QPushButton("Delete Selected Box")
        delete_button.clicked.connect(self.canvas.delete_selected)
        clear_button = QPushButton("Clear All Boxes")
        clear_button.clicked.connect(self.confirm_clear)
        side_layout.addWidget(delete_button)
        side_layout.addWidget(clear_button)
        instruction = QLabel(
            "Draw: drag on empty image area\n"
            "Overlapping box: hold Shift and drag\n"
            "Select: click inside a box\n"
            "Correct a box: delete it, then redraw\n\n"
            "Shortcuts: ←/→ navigate, R/G class, Delete box, Ctrl+S save, Space save+next"
        )
        instruction.setWordWrap(True)
        side_layout.addWidget(instruction)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, stretch=1)

        buttons = QHBoxLayout()
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(lambda: self.navigate(-1))
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_current)
        next_button = QPushButton("Next (not reviewed)")
        next_button.clicked.connect(lambda: self.navigate(1))
        review_next_button = QPushButton("Mark Reviewed + Next")
        review_next_button.clicked.connect(self.review_and_next)
        buttons.addWidget(previous_button)
        buttons.addWidget(save_button)
        buttons.addWidget(next_button)
        buttons.addStretch()
        buttons.addWidget(review_next_button)
        layout.addLayout(buttons)
        self.setCentralWidget(root)

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.navigate(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.navigate(1))
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self.canvas.delete_selected)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current)
        QShortcut(QKeySequence("R"), self, activated=lambda: self.choose_class(0))
        QShortcut(QKeySequence("G"), self, activated=lambda: self.choose_class(1))
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.review_and_next)

    def load_review(self) -> dict:
        if REVIEW_PATH.is_file():
            try:
                value = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        return {"schema_version": 1, "images": {}}

    def write_review(self) -> None:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = REVIEW_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(REVIEW_PATH)

    def image_key(self, path: Path) -> str:
        return str(path.relative_to(PROJECT_ROOT))

    def is_reviewed(self, path: Path) -> bool:
        return bool(self.review.get("images", {}).get(self.image_key(path), {}).get("reviewed"))

    def apply_filter(self, *_args) -> None:
        if self.dirty:
            self.save_current()
        category = self.filter_combo.currentData()
        self.filtered_images = [
            path
            for path in self.all_images
            if (category == "all" or path.parent.name == category)
            and (not self.unreviewed_check.isChecked() or not self.is_reviewed(path))
        ]
        self.current_index = 0 if self.filtered_images else -1
        self.load_current()

    def label_path(self, image_path: Path) -> Path:
        return LABEL_ROOT / image_path.parent.name / f"{image_path.stem}.txt"

    def read_boxes(self, image_path: Path) -> list[dict[str, float | int]]:
        boxes = []
        label_path = self.label_path(image_path)
        if not label_path.is_file():
            return boxes
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                x, y, width, height = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            if class_id in (0, 1):
                boxes.append(
                    {"class_id": class_id, "x": x, "y": y, "w": width, "h": height}
                )
        return boxes

    def load_current(self) -> None:
        if not (0 <= self.current_index < len(self.filtered_images)):
            self.current_path = None
            self.canvas.set_content(QImage(), [])
            self.filename_label.setText("No images match this filter.")
            self.progress_label.setText("0 / 0")
            self.box_list.clear()
            return
        self.current_path = self.filtered_images[self.current_index]
        image = QImage(str(self.current_path))
        if image.isNull():
            QMessageBox.warning(self, "Image Error", f"Cannot read {self.current_path}")
            return
        boxes = self.read_boxes(self.current_path)
        self.canvas.set_content(image, boxes)
        self.dirty = False
        self.filename_label.setText(
            f"{self.current_path.parent.name}\n{self.current_path.name}"
        )
        self.status_label.setText(
            "Reviewed" if self.is_reviewed(self.current_path) else "Needs manual review"
        )
        self.update_box_list()
        reviewed_count = sum(self.is_reviewed(path) for path in self.all_images)
        self.progress_label.setText(
            f"{self.current_index + 1} / {len(self.filtered_images)}   "
            f"reviewed {reviewed_count} / {len(self.all_images)}"
        )

    def update_box_list(self) -> None:
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for index, box in enumerate(self.canvas.boxes):
            class_id = int(box["class_id"])
            self.box_list.addItem(f"{index + 1}. {CLASS_NAMES[class_id]}")
        self.box_list.setCurrentRow(self.canvas.selected_index)
        self.box_list.blockSignals(False)

    def boxes_changed(self) -> None:
        self.dirty = True
        self.status_label.setText("Unsaved changes")
        self.update_box_list()

    def selection_changed(self, index: int) -> None:
        self.box_list.blockSignals(True)
        self.box_list.setCurrentRow(index)
        self.box_list.blockSignals(False)
        if 0 <= index < len(self.canvas.boxes):
            self.choose_class(int(self.canvas.boxes[index]["class_id"]), update_box=False)

    def list_selection_changed(self, index: int) -> None:
        self.canvas.selected_index = index
        if 0 <= index < len(self.canvas.boxes):
            self.choose_class(int(self.canvas.boxes[index]["class_id"]), update_box=False)
        self.canvas.update()

    def choose_class(self, class_id: int, update_box: bool = True) -> None:
        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentIndex(class_id)
        self.class_combo.blockSignals(False)
        if update_box:
            self.canvas.set_selected_class(class_id)
        else:
            self.canvas.active_class = class_id

    def class_changed(self, _index: int) -> None:
        self.canvas.set_selected_class(int(self.class_combo.currentData()))

    def save_current(self) -> None:
        if self.current_path is None:
            return
        label_path = self.label_path(self.current_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{int(box['class_id'])} {float(box['x']):.6f} {float(box['y']):.6f} "
            f"{float(box['w']):.6f} {float(box['h']):.6f}"
            for box in self.canvas.boxes
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty = False
        self.status_label.setText(
            "Reviewed" if self.is_reviewed(self.current_path) else "Saved; still needs review"
        )

    def review_and_next(self) -> None:
        if self.current_path is None:
            return
        self.save_current()
        key = self.image_key(self.current_path)
        self.review.setdefault("images", {})[key] = {
            "reviewed": True,
            "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "box_count": len(self.canvas.boxes),
        }
        self.write_review()
        if self.unreviewed_check.isChecked():
            del self.filtered_images[self.current_index]
            if self.current_index >= len(self.filtered_images):
                self.current_index = max(0, len(self.filtered_images) - 1)
            self.load_current()
        else:
            self.navigate(1, save=False)

    def navigate(self, offset: int, save: bool = True) -> None:
        if not self.filtered_images:
            return
        if save and self.dirty:
            self.save_current()
        self.current_index = max(
            0, min(len(self.filtered_images) - 1, self.current_index + offset)
        )
        self.load_current()

    def confirm_clear(self) -> None:
        if not self.canvas.boxes:
            return
        answer = QMessageBox.question(
            self,
            "Clear Boxes",
            "Remove every box from this image?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.canvas.clear_boxes()

    def closeEvent(self, event) -> None:
        if self.dirty:
            self.save_current()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = DatasetAnnotator()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
