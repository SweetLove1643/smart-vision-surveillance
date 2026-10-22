import logging
import os
import sys
import threading
import time

# Phải đặt TRƯỚC khi import cv2: ép RTSP chạy qua TCP (UDP dễ rớt gói gây vỡ hình).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger("capture")

# Trạng thái của một camera
STOPPED = "stopped"
CONNECTING = "connecting"
LIVE = "live"
RECONNECTING = "reconnecting"

RECONNECT_MIN_DELAY = 1.0   # giây, lần thử lại đầu tiên
RECONNECT_MAX_DELAY = 10.0  # giây, chờ tối đa giữa hai lần thử
MAX_READ_FAILS = 3          # đọc hỏng liên tiếp bao nhiêu lần thì coi là mất kết nối
OPEN_TIMEOUT_MS = 5000      # RTSP: chờ mở kết nối tối đa
READ_TIMEOUT_MS = 5000      # RTSP: chờ frame tối đa

class CameraReader:
    """Đọc một nguồn video (RTSP / file / webcam) ở thread riêng.

    - Luôn giữ frame MỚI NHẤT, frame cũ bị ghi đè -> không dồn hàng đợi (F02).
    - Mất kết nối / không mở được -> tự thử lại, chờ tăng dần 1s, 2s, 4s... tối đa 10s.
    - Mỗi camera một thread và mọi lỗi đều bị bắt tại chỗ -> camera lỗi không kéo camera khác (F01).
    - File video: hết file thì phát lại từ đầu (để test như camera chạy liên tục).
    """

    def __init__(self, source: int | str, name: str = "camera") -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # "0" trong config -> webcam số 0
        self.source = source
        self.name = name

        if isinstance(source, int):
            self.kind = "webcam"
        elif source.lower().startswith("rtsp://"):
            self.kind = "rtsp"
        else:
            self.kind = "file"

        self.state = STOPPED
        self.reconnect_count = 0

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- API công khai ----------
    def start(self) -> "CameraReader":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"reader-{self.name}", daemon=True)
        self._thread.start()
        return self

    def request_stop(self) -> None:
        """Báo thread dừng nhưng không chờ. Dùng để báo nhiều camera cùng lúc rồi mới chờ."""
        self._stop.set()

    def stop(self) -> None:
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def read(self) -> np.ndarray | None:
        """Frame mới nhất (bản sao), hoặc None nếu chưa có / đang mất kết nối."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ---------- Bên trong ----------
    def _set_state(self, state: str) -> None:
        if state != self.state:
            log.info("[%s] %s -> %s", self.name, self.state, state)
            self.state = state

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            got_frames = False
            try:
                cap = self._open()
                if cap is not None:
                    try:
                        got_frames = self._read_loop(cap)
                    finally:
                        cap.release()
            except Exception:  # lỗi bất ngờ cũng không được làm chết thread
                log.exception("[%s] lỗi bất ngờ khi đọc", self.name)

            with self._lock:
                self._frame = None  # mất kết nối thì xoá frame cũ, UI hiện "No signal"
            if self._stop.is_set():
                break

            self._set_state(RECONNECTING)
            self.reconnect_count += 1
            if got_frames:
                failures = 0  # đã chạy được một lúc rồi mới rớt -> thử lại ngay từ mức thấp
            wait = min(RECONNECT_MIN_DELAY * 2**failures, RECONNECT_MAX_DELAY)
            failures += 1
            log.warning("[%s] mất kết nối, thử lại sau %.1fs (lần %d)", self.name, wait, self.reconnect_count)
            self._stop.wait(wait)  # wait thay cho sleep: stop() đánh thức ngay

        self._set_state(STOPPED)

    def _open(self) -> "cv2.VideoCapture | None":
        self._set_state(CONNECTING)
        if self.kind == "webcam":
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # DSHOW mở nhanh hơn trên Windows
            cap = cv2.VideoCapture(self.source, backend)
        elif self.kind == "rtsp":
            cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS],
            )
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer nhỏ nhất, giảm độ trễ
        return cap

    def _read_loop(self, cap: "cv2.VideoCapture") -> bool:
        """Đọc đến khi mất kết nối hoặc bị stop. Trả về True nếu đã đọc được ít nhất 1 frame."""
        is_file = self.kind == "file"
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0  # file: phát đúng tốc độ gốc
        next_time = time.monotonic()
        got_frames = False
        fails = 0

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= MAX_READ_FAILS:
                    return got_frames
                if is_file:  # hết file -> tua về đầu
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            fails = 0
            if not got_frames:
                got_frames = True
                self._set_state(LIVE)
            with self._lock:
                self._frame = frame

            if is_file:
                next_time += period
                self._stop.wait(max(0.0, next_time - time.monotonic()))
        return got_frames


def _demo() -> None:
    """Chạy thử: python -m main_app.services.capture   (cần opencv-python, nhấn q để thoát)."""
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cameras = json.loads((Path(__file__).parents[1] / "ui" / "cameras.json").read_text(encoding="utf-8"))
    readers = [CameraReader(c["source"], c["name"]).start() for c in cameras]
    try:
        while True:
            for reader in readers:
                frame = reader.read()
                if frame is not None:
                    cv2.imshow(reader.name, frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        for reader in readers:
            reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()