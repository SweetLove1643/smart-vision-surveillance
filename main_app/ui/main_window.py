import json
import math
from pathlib import Path
import sys
import logging

from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QAction, QKeySequence, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
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

from main_app.controllers.camera_controller import CONNECTING, LIVE, RECONNECTING, STOPPED, CameraController
from main_app.services.logging_setup import mask_source

logger = logging.getLogger("main_window")

FRAME_INTERVAL_MS = 33  # ~30 lần/giây lấy frame đưa lên giao diện

# Hai file config nằm cạnh file này, không phụ thuộc thư mục chạy.
UI_DIR = Path(__file__).parent
UI_CONFIG_PATH = UI_DIR / "ui_config.json"  # cài đặt giao diện (số camera hiển thị)
CAMERAS_PATH = UI_DIR / "cameras.json"      # danh sách nguồn camera

MAX_CAMERAS = 16

STYLE = """
QMainWindow, QWidget { background: #1e1f22; color: #dcdcdc; font-size: 13px; }
QGroupBox { border: 1px solid #3a3c40; border-radius: 6px; margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa0a6; }
QFrame#cameraView { background: #2b2d31; border: 1px solid #3a3c40; border-radius: 6px; }
QLabel#camTitle { font-weight: bold; }
QLabel#camStatus { color: #ef5350; font-size: 11px; }
QLabel#videoArea { background: #000; color: #6b6f76; border-radius: 4px; }
QLabel#hint { color: #6b6f76; font-size: 11px; }
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
QMenu { background: #2b2d31; border: 1px solid #3a3c40; }
QMenu::item { padding: 6px 28px; }
QMenu::item:selected { background: #3d5afe; }
"""

# trạng thái reader -> (chữ trên nhãn, màu, thông báo khi chưa có hình)
STATE_VIEW = {
    STOPPED: ("OFFLINE", "#ef5350", "No signal"),
    CONNECTING: ("CONNECTING", "#ffb300", "Đang kết nối..."),
    LIVE: ("LIVE", "#66bb6a", ""),
    RECONNECTING: ("RECONNECTING", "#ffb300", "Mất kết nối, đang thử lại..."),
}

def load_json(path: Path, default):
    """Đọc JSON; file thiếu hoặc hỏng thì dùng giá trị mặc định."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Không đọc được %s (%s), dùng mặc định", path.name, exc)
        return default


def save_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.exception("Không ghi được %s", path.name)


class CameraView(QFrame):
    """Một ô camera: tiêu đề + trạng thái + vùng hiển thị video."""

    def __init__(self, name: str, source) -> None:
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
        if state == self._state:  # chỉ cập nhật khi đổi, tránh dựng lại style 30 lần/giây
            return
        self._state = state
        badge, color, _ = STATE_VIEW[state]
        self.status.setText(badge)
        self.status.setStyleSheet(f"color: {color};")
        
    def show_frame(self, frame) -> None:
        """Hiển thị frame BGR (numpy) của OpenCV, giữ nguyên tỉ lệ."""
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        )
        self.video.setPixmap(pixmap)

    def clear_frame(self, message: str) -> None:
        if self.video.text() != message:  # setText cũng xoá luôn hình cũ
            self.video.setText(message)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.camera_controller = CameraController()
        self.views: list[CameraView] = []
        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_INTERVAL_MS)
        self.timer.timeout.connect(self._update_frames)

        self.setWindowTitle("Multi-Camera Monitor")
        self.resize(1440, 820)
        self.setStyleSheet(STYLE)

        self.ui_config = {"num_cameras": 4, **load_json(UI_CONFIG_PATH, {})}
        self.camera_sources: list[dict] = load_json(CAMERAS_PATH, [])

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

    # ---------- Dựng giao diện ----------
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Hiển thị")
        group_layout = QVBoxLayout(group)
        group_layout.addWidget(QLabel("Số camera:"))

        self.spin_cameras = QSpinBox()
        self.spin_cameras.setRange(1, MAX_CAMERAS)
        self.spin_cameras.setValue(self.ui_config["num_cameras"])
        self.spin_cameras.valueChanged.connect(self._on_num_cameras_changed)
        group_layout.addWidget(self.spin_cameras)

        # hint = QLabel("Nguồn camera: sửa file ui/cameras.json")
        # hint.setObjectName("hint")
        # hint.setWordWrap(True)
        # group_layout.addWidget(hint)

        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("startBtn")
        self.btn_stop = QPushButton("Stop")
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
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        counts_group = QGroupBox("Đếm qua vạch")
        counts_layout = QVBoxLayout(counts_group)
        self.counts_table = self._make_table(["Camera", "Vào", "Ra"])
        counts_layout.addWidget(self.counts_table)

        events_group = QGroupBox("Sự kiện")
        events_layout = QVBoxLayout(events_group)
        self.events_table = self._make_table(["Thời gian", "Camera", "ID", "Loại"])
        events_layout.addWidget(self.events_table)

        layout.addWidget(counts_group, 1)
        layout.addWidget(events_group, 2)
        return panel
    
    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        app_menu = menu_bar.addMenu("Ứng dụng")
        restart_action = QAction("Khởi động lại", self)
        restart_action.setShortcut(QKeySequence("Ctrl+R"))
        restart_action.triggered.connect(self._restart_app)
        app_menu.addAction(restart_action)
        app_menu.addSeparator()
        exit_action = QAction("Thoát", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        app_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu("Trợ giúp")
        about_action = QAction("Giới thiệu", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _show_about(self) -> None:
        QMessageBox.information(self, "Giới thiệu", "Multi-Camera Monitor\nPhiên bản 0.1")

    def _restart_app(self) -> None:
        answer = QMessageBox.question(self, "Khởi động lại", "Khởi động lại ứng dụng?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        logger.info("Khởi động lại ứng dụng")
        self._stop_cameras()  # trả webcam/kết nối về trước, tiến trình mới mới mở lại được
        # Mở một tiến trình mới chạy đúng lệnh hiện tại (python main.py), rồi đóng tiến trình này.
        started, pid = QProcess.startDetached(sys.executable, sys.argv)
        if not started:  # không mở được tiến trình mới thì KHÔNG được thoát, kẻo mất luôn ứng dụng
            logger.error("Không mở được tiến trình mới: %s %s", sys.executable, sys.argv)
            QMessageBox.warning(self, "Khởi động lại", "Không mở được tiến trình mới. Xem logs/app.log")
            return
        logger.info("Đã mở tiến trình mới (pid=%s), đóng tiến trình này", pid)
        QApplication.quit()

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    # ---------- Xử lý ----------
    def _on_num_cameras_changed(self, value: int) -> None:
        logger.info("Đổi số camera hiển thị: %d", value)
        self.ui_config["num_cameras"] = value
        save_json(UI_CONFIG_PATH, self.ui_config)
        self._rebuild_grid()

    def _camera_info(self, index: int) -> tuple[str, object]:
        """Tên + nguồn của camera thứ index; thiếu trong cameras.json thì dùng tên mặc định."""
        if index < len(self.camera_sources):
            info = self.camera_sources[index]
            logger.info(info)
            return info.get("name", f"CAM {index + 1}"), info.get("source")
        return f"CAM {index + 1}", None

    def _rebuild_grid(self) -> None:
        n = self.spin_cameras.value()

        # Xoá các ô cũ
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        # Chia lưới gần vuông: 1 -> 1x1, 2 -> 2x1, 3-4 -> 2x2, 5-6 -> 3x2, 7-9 -> 3x3, ...
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        for i in range(n):
            name, source = self._camera_info(i)
            view = CameraView(name, source)
            self.views.append(view)
            self.grid.addWidget(view, i // cols, i % cols)

        # Hàng/cột đang dùng chia đều; hàng/cột thừa của lần trước về 0.
        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(self.grid.columnCount()):
            self.grid.setColumnStretch(c, 1 if c < cols else 0)

        # Bảng đếm: mỗi camera một dòng
        self.counts_table.setRowCount(n)
        for row in range(n):
            for col, text in enumerate([self._camera_info(row)[0], "0", "0"]):
                self.counts_table.setItem(row, col, QTableWidgetItem(text))

        self.statusBar().showMessage(f"Hiển thị {n} camera")
        logger.info("Dựng lưới %d camera (%d hàng x %d cột)", n, rows, cols)


        
    # --- các hàm điều khiển ---
    def _start_cameras(self) -> None:
        cameras = [self._camera_info(i) for i in range(len(self.views))]
        if all(source is None for _, source in cameras):
            logger.warning("Start bị từ chối: chưa camera nào có nguồn trong cameras.json")
            QMessageBox.information(self, "Chưa có nguồn", "Chưa camera nào có nguồn.\nHãy thêm vào ui/cameras.json")
            return
        for i, (name, source) in enumerate(cameras):
            logger.info("Camera %d '%s' nguồn: %s", i, name, mask_source(source))  # che user:pass của RTSP
        self.camera_controller.start(cameras)
        self._set_running(True)
        self.timer.start()

    def _stop_cameras(self) -> None:
        self.timer.stop()
        self.camera_controller.stop()
        for view in self.views:
            view.set_state(STOPPED)
            view.clear_frame(STATE_VIEW[STOPPED][2])
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.spin_cameras.setEnabled(not running)  # đang chạy thì không đổi số camera

    def closeEvent(self, event) -> None:
        logger.info("Đóng cửa sổ")
        self._stop_cameras()  # đóng cửa sổ cũng phải trả webcam / kết nối
        super().closeEvent(event)

    def _update_frames(self) -> None:
        try:
            for index, view in enumerate(self.views):
                state = self.camera_controller.get_state(index)
                view.set_state(state)
                frame = self.camera_controller.get_frame(index)
                if frame is not None:
                    view.show_frame(frame)
                elif state != LIVE:  # LIVE mà chưa kịp có frame: giữ hình cũ, tránh nhấp nháy
                    view.clear_frame(STATE_VIEW[state][2])
        except Exception:
            # Hàm này chạy 30 lần/giây: nếu lỗi mà cứ tiếp tục thì log ngập 30 dòng/giây. Ghi 1 lần rồi dừng.
            logger.exception("Lỗi khi cập nhật khung hình, dừng toàn bộ camera")
            self._stop_cameras()
            self.statusBar().showMessage("Lỗi khi hiển thị, xem logs/app.log")