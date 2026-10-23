import logging
import os
import threading
import time
from typing import Optional, Union

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2
import numpy as np

logger = logging.getLogger("services.capture")

STOPPED = "stopped"
CONNECTING = "connecting"
LIVE = "live"
RECONNECTING = "reconnecting"


class CameraReader:
    """Đọc nguồn video từ RTSP / File / Webcam trong thread riêng (F01, F02).
    
    Tối ưu:
    - Điều tiết tốc độ đọc file video theo đúng FPS tự nhiên (tránh việc giải mã 150-200 FPS gây quá tải CPU).
    - Cơ chế zero-latency: luôn chỉ giữ 1 frame mới nhất.
    - Tự động reconnect với exponential backoff khi mất kết nối.
    """

    def __init__(self, camera_id: str, source: Union[int, str], name: str = "") -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.camera_id = camera_id
        self.source = source
        self.name = name or camera_id
        self.state = STOPPED
        self.reconnect_count = 0

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fps_count = 0
        self._fps_start_time = time.time()
        self.fps: float = 0.0

    def start(self) -> "CameraReader":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"capture-{self.camera_id}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.state = STOPPED

    def read_latest_frame(self) -> Optional[np.ndarray]:
        """Lấy frame mới nhất, không bao giờ bị nghẽn queue (F02)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        logger.info("[%s] Đang kết nối tới nguồn: %s", self.camera_id, self.source)
        self.state = CONNECTING

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.warning("[%s] Không mở được nguồn: %s", self.camera_id, self.source)
            cap.release()
            return None

        self.state = LIVE
        logger.info("[%s] Kết nối camera thành công!", self.camera_id)
        return cap

    def _run_loop(self) -> None:
        reconnect_delay = 1.0
        max_reconnect_delay = 10.0

        is_file = isinstance(self.source, str) and not self.source.startswith("rtsp://")

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                self.state = RECONNECTING
                self.reconnect_count += 1
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                continue

            reconnect_delay = 1.0
            consecutive_failures = 0

            # Lấy FPS của nguồn để điều tiết tốc độ phát cho file video
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if native_fps <= 0 or native_fps > 60:
                native_fps = 25.0
            frame_interval = 1.0 / native_fps

            while not self._stop_event.is_set():
                t_frame_start = time.time()
                ret, frame = cap.read()

                if not ret or frame is None:
                    # File video: lặp lại liên tục từ đầu
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue

                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.warning("[%s] Mất tín hiệu 5 frame liên tiếp.", self.camera_id)
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                with self._lock:
                    self._frame = frame

                self._fps_count += 1
                elapsed_fps = time.time() - self._fps_start_time
                if elapsed_fps >= 1.0:
                    self.fps = round(self._fps_count / elapsed_fps, 1)
                    self._fps_count = 0
                    self._fps_start_time = time.time()

                # TỐI ƯU CỰC KỲ QUAN TRỌNG: Điều tiết tốc độ đọc file theo FPS tự nhiên (25-30 FPS)
                # Tránh tình trạng OpenCV đọc với tốc độ 150-200 FPS làm quá tải 100% CPU
                if is_file:
                    proc_time = time.time() - t_frame_start
                    if proc_time < frame_interval:
                        time.sleep(frame_interval - proc_time)

            cap.release()
            if not self._stop_event.is_set():
                self.state = RECONNECTING
                self.reconnect_count += 1
                logger.info("[%s] Thử kết nối lại lần %d...", self.camera_id, self.reconnect_count)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
