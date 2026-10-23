import math
import sys
import logging
from typing import Optional
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from main_app.controllers.camera_controller import CameraController
from main_app.services import (
    CONNECTING,
    LIVE,
    RECONNECTING,
    STOPPED,
    load_config,
    setup_logging,
)

logger = logging.getLogger("ui.main_window")
FRAME_INTERVAL_MS = 33  # ~30 FPS làm mới giao diện
MAX_CAMERAS = 16

STYLE = """
QMainWindow, QWidget { background: #1e1f22; color: #dcdcdc; font-size: 13px; }
QGroupBox { border: 1px solid #3a3c40; border-radius: 6px; margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa0a6; }
QFrame#cameraView { background: #2b2d31; border: 1px solid #3a3c40; border-radius: 6px; }
QLabel#camTitle { font-weight: bold; }
QLabel#camStatus { color: #ef5350; font-size: 11px; }
QLabel#videoArea { background: #000; color: #6b6f76; border-radius: 4px; }
QSpinBox { background: #2b2d31; border: 1px solid #3a3c40; border-radius: 4px; padding: 4px; }
QPushButton { background: #3a3c40; border: none; border-radius: 4px; padding: 8px; }
QPushButton:disabled { color: #6b6f76; }
QPushButton#startBtn { background: #2e7d32; }
QPushButton#stopBtn { background: #c62828; }
QTableWidget { background: #2b2d31; gridline-color: #3a3c40; border: none; }
QHeaderView::section { background: #3a3c40; border: none; padding: 4px; }
QMenuBar { background: #2b2d31; }
QMenuBar::item { padding: 6px 12px; background: transparent; }
QMenuBar::item:selected { background: #3a3c40; }
"""

STATE_VIEW = {
    STOPPED: ("OFFLINE", "#ef5350", "Chưa kết nối"),
    CONNECTING: ("CONNECTING", "#ffb300", "Đang kết nối..."),
    LIVE: ("LIVE", "#66bb6a", ""),
    RECONNECTING: ("RECONNECTING", "#ffb300", "Mất kết nối, đang thử lại..."),
}


class CameraView(QFrame):
    """Một ô hiển thị camera trên lưới với tối ưu hiển thị nhanh."""

    def __init__(self, name: str, source: str) -> None:
        super().__init__()
        self.setObjectName("cameraView")
        self.setToolTip(f"Nguồn: {source}")
        self._state = STOPPED

        title = QLabel(name)
        title.setObjectName("camTitle")
        self.status = QLabel("OFFLINE")
        self.status.setObjectName("camStatus")

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status)

        self.video = QLabel("No signal")
        self.video.setObjectName("videoArea")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(240, 135)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addWidget(self.video, 1)

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        badge, color, _ = STATE_VIEW.get(state, ("UNKNOWN", "#999", ""))
        self.status.setText(badge)
        self.status.setStyleSheet(f"color: {color};")

    def show_frame(self, frame) -> None:
        """Hiển thị frame với tối ưu hóa resize trước khi chuyển đổi sang Qt."""
        h, w = frame.shape[:2]
        vw = max(self.video.width(), 320)
        vh = max(self.video.height(), 180)

        # Nếu frame quá to so với widget (như 1080p), resize nhanh bằng OpenCV
        if w > vw:
            scale = vw / float(w)
            target_h = int(h * scale)
            frame = cv2.resize(frame, (vw, target_h), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]

        image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.video.setPixmap(QPixmap.fromImage(image))

    def clear_frame(self, message: str) -> None:
        if self.video.text() != message:
            self.video.setText(message)


class MainWindow(QMainWindow):
    """Giao diện Qt6 để quan sát trực tiếp các luồng camera khi cần."""

    def __init__(self, controller: Optional[CameraController] = None) -> None:
        super().__init__()

        if controller is not None:
            self.controller = controller
        else:
            config = load_config("config.yaml")
            self.controller = CameraController(config)

        self.views: list[CameraView] = []
        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_INTERVAL_MS)
        self.timer.timeout.connect(self._update_frames)

        self.setWindowTitle("Multi-Camera Real-Time Monitor (Qt6)")
        self.resize(1440, 820)
        self.setStyleSheet(STYLE)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_left_panel())
        layout.addWidget(self._build_camera_grid(), 1)
        layout.addWidget(self._build_right_panel())
        self.setCentralWidget(root)

        self._build_menu_bar()
        self._rebuild_grid()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Điều khiển")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(QLabel("Số camera hiển thị:"))

        self.spin_cameras = QSpinBox()
        self.spin_cameras.setRange(1, min(MAX_CAMERAS, len(self.controller.camera_order) or 1))
        self.spin_cameras.setValue(len(self.controller.camera_order))
        self.spin_cameras.valueChanged.connect(self._on_num_cameras_changed)
        group_layout.addWidget(self.spin_cameras)

        self.btn_start = QPushButton("Start All")
        self.btn_start.setObjectName("startBtn")
        self.btn_stop = QPushButton("Stop All")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._start_cameras)
        self.btn_stop.clicked.connect(self._stop_cameras)

        layout.addWidget(group)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addStretch()
        return panel

    def _build_camera_grid(self) -> QWidget:
        panel = QWidget()
        self.grid = QGridLayout(panel)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        counts_group = QGroupBox("Thống kê Đếm vạch (F05)")
        counts_layout = QVBoxLayout(counts_group)
        self.counts_table = self._make_table(["Camera", "Vào (IN)", "Ra (OUT)"])
        counts_layout.addWidget(self.counts_table)

        events_group = QGroupBox("Nhật ký sự kiện gần nhất (F07)")
        events_layout = QVBoxLayout(events_group)
        self.events_table = self._make_table(["Thời gian", "Camera", "Track ID", "Loại sự kiện"])
        events_layout.addWidget(self.events_table)

        layout.addWidget(counts_group, 1)
        layout.addWidget(events_group, 2)
        return panel

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        app_menu = menu_bar.addMenu("Ứng dụng")

        exit_action = QAction("Thoát", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        app_menu.addAction(exit_action)

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def _on_num_cameras_changed(self, value: int) -> None:
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        n = self.spin_cameras.value()
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.views = []
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        for i in range(n):
            if i < len(self.controller.camera_order):
                cam_id = self.controller.camera_order[i]
                pipe = self.controller.pipelines[cam_id]
                view = CameraView(pipe.name, str(pipe.config.source))
            else:
                view = CameraView(f"CAM {i + 1}", "N/A")

            self.views.append(view)
            self.grid.addWidget(view, i // cols, i % cols)

        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(self.grid.columnCount()):
            self.grid.setColumnStretch(c, 1 if c < cols else 0)

        self._refresh_tables()

    def _refresh_tables(self) -> None:
        self.counts_table.setRowCount(len(self.controller.camera_order))
        for row, cid in enumerate(self.controller.camera_order):
            pipe = self.controller.pipelines[cid]
            c_in = pipe.line_counter.count_in if pipe.line_counter else 0
            c_out = pipe.line_counter.count_out if pipe.line_counter else 0
            self.counts_table.setItem(row, 0, QTableWidgetItem(pipe.name))
            self.counts_table.setItem(row, 1, QTableWidgetItem(str(c_in)))
            self.counts_table.setItem(row, 2, QTableWidgetItem(str(c_out)))

        recent_events = self.controller.db.get_events(limit=10)
        self.events_table.setRowCount(len(recent_events))
        for row, ev in enumerate(recent_events):
            time_part = str(ev.get("created_at", ""))[-8:]
            self.events_table.setItem(row, 0, QTableWidgetItem(time_part))
            self.events_table.setItem(row, 1, QTableWidgetItem(str(ev.get("camera_id"))))
            self.events_table.setItem(row, 2, QTableWidgetItem(f"#{ev.get('track_id')}"))
            self.events_table.setItem(row, 3, QTableWidgetItem(str(ev.get("event_type"))))

    def _start_cameras(self) -> None:
        self.controller.start_all()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()

    def _stop_cameras(self) -> None:
        self.timer.stop()
        self.controller.stop_all()
        for view in self.views:
            view.set_state(STOPPED)
            view.clear_frame("Đã dừng")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event) -> None:
        self._stop_cameras()
        super().closeEvent(event)

    def _update_frames(self) -> None:
        for index, view in enumerate(self.views):
            state = self.controller.get_state_by_index(index)
            view.set_state(state)
            frame = self.controller.get_frame_by_index(index)
            if frame is not None:
                view.show_frame(frame)

        # Cập nhật số liệu mỗi giây
        if self.timer.isActive() and (self.timer.remainingTime() < 40):
            self._refresh_tables()


def run_ui():
    setup_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()
