import logging
import queue
import shutil
import socket
import subprocess
import threading
import time
from typing import Optional
from urllib.parse import urlparse
import numpy as np

from main_app.services.logging_service import mask_source

logger = logging.getLogger("services.streamer")


class FFmpegRTSPStreamer:
    """Đẩy luồng video ra RTSP qua FFmpeg (F08).
    
    Cải tiến tối ưu hoá:
    - Bất đồng bộ hoàn toàn (Asynchronous): `push_frame` đẩy vào queue và trả về ngay tức thì,
      không bao giờ làm nghẽn (freeze) luồng nhận diện AI của camera.
    - Tự động kiểm tra cổng kết nối của RTSP server: nếu server RTSP (vd MediaMTX trên port 8554)
      chưa bật, tiến trình FFmpeg sẽ không bị bật/tắt liên tục gây tràn bộ đệm pipe và đơ máy.
    - Tự động hồi sinh FFmpeg khi có sự cố mạng.
    """

    def __init__(self, rtsp_url: str, fps: int = 25, enabled: bool = True) -> None:
        self.rtsp_url = rtsp_url
        if fps < 1:
            raise ValueError("FPS của RTSP streamer phải lớn hơn 0.")
        self.fps = fps
        self.enabled = enabled
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._ffmpeg_available = self._ffmpeg_path is not None

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None

        self._host, self._port = self._parse_host_port(rtsp_url)
        self._last_warn_time = 0.0
        self._last_process_exit: Optional[int] = None

        if not self._ffmpeg_available:
            logger.warning("Không tìm thấy 'ffmpeg' trong PATH. Chức năng RTSP tạm thời tắt.")
        elif self.enabled:
            logger.info(
                "RTSP output được bật tới %s. Ứng dụng chỉ publish; cần một RTSP "
                "server riêng để listen trên cổng đích.",
                mask_source(self.rtsp_url),
            )

    def start(self) -> None:
        if not self.enabled or not self._ffmpeg_available:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop, name=f"streamer-{self._port}", daemon=True
        )
        self._thread.start()

    def _parse_host_port(self, url: str) -> tuple[str, int]:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            raise ValueError(f"RTSP output URL không hợp lệ: {mask_source(url)}")
        try:
            port = parsed.port or 554
        except ValueError as exc:
            raise ValueError(f"Port trong RTSP URL không hợp lệ: {mask_source(url)}") from exc
        return parsed.hostname, port

    def _is_server_listening(self) -> bool:
        """Kiểm tra nhanh cổng RTSP server có đang mở không để tránh lỗi pipe."""
        try:
            with socket.create_connection((self._host, self._port), timeout=0.3):
                return True
        except OSError:
            return False

    def push_frame(self, frame: np.ndarray) -> bool:
        """Đẩy frame vào hàng đợi stream (non-blocking). Không bao giờ làm chậm camera."""
        if not self.enabled or not self._ffmpeg_available:
            return False

        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            return False

        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def _start_process(self, width: int, height: int) -> Optional[subprocess.Popen]:
        # Avoid quantizing adjacent wall-clock timestamps to the same input tick.
        timestamp_rate = max(30, self.fps * 2)
        command = [
            self._ffmpeg_path or "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-nostats",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-framerate", str(timestamp_rate),
            "-use_wallclock_as_timestamps", "1",
            "-i", "-",
            "-c:v", "libx264",
            "-g", str(self.fps),
            "-keyint_min", str(self.fps),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-fps_mode", "vfr",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info(
                "Đã khởi chạy FFmpeg publisher (pid=%d) tới %s; đang chờ xác nhận "
                "publish từ FFmpeg.",
                proc.pid,
                mask_source(self.rtsp_url),
            )
            threading.Thread(
                target=self._log_process_errors,
                args=(proc,),
                name=f"ffmpeg-stderr-{self._port}",
                daemon=True,
            ).start()
            return proc
        except Exception as exc:
            logger.exception(
                "Không khởi chạy được FFmpeg publisher tới %s: %s",
                mask_source(self.rtsp_url),
                exc,
            )
            return None

    def _log_process_errors(self, process: subprocess.Popen) -> None:
        if process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            message = raw_line.decode("utf-8", errors="replace").strip()
            if message:
                logger.warning(
                    "FFmpeg (%s): %s",
                    mask_source(self.rtsp_url),
                    mask_source(message),
                )

    def _worker_loop(self) -> None:
        last_check_time = 0.0
        server_ok = False
        width, height = 0, 0

        while not self._stop_event.is_set():
            # 1. Định kỳ 5s kiểm tra xem RTSP server đích (vd port 8554) có đang sống không
            now = time.time()
            if now - last_check_time > 5.0:
                server_ok = self._is_server_listening()
                last_check_time = now
                if not server_ok and (now - self._last_warn_time > 30.0):
                    logger.warning(
                        "Không có RTSP server listen tại %s:%d; FFmpeg chưa được chạy. "
                        "Ứng dụng này chỉ publish, không mở cổng RTSP. Hãy khởi chạy "
                        "MediaMTX hoặc một RTSP server khác trước khi kết nối VLC (%s).",
                        self._host,
                        self._port,
                        mask_source(self.rtsp_url),
                    )
                    self._last_warn_time = now

            if not server_ok:
                if self.process is not None:
                    self.close_process()
                time.sleep(1.0)
                continue

            # 2. Lấy frame từ queue để bắn sang FFmpeg
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]

            if self.process is not None and self.process.poll() is not None:
                return_code = self.process.returncode
                if return_code != self._last_process_exit:
                    logger.error(
                        "FFmpeg publisher tới %s đã thoát (exit code=%s). "
                        "Kiểm tra lỗi FFmpeg ở các log ngay phía trên.",
                        mask_source(self.rtsp_url),
                        return_code,
                    )
                    self._last_process_exit = return_code
                self.close_process()

            if self.process is None or w != width or h != height:
                self.close_process()
                self.process = self._start_process(w, h)
                width, height = w, h

            if self.process is None:
                time.sleep(1.0)
                continue

            if self.process and self.process.stdin:
                try:
                    self.process.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    logger.error(
                        "Không ghi được frame vào FFmpeg publisher tới %s: %s",
                        mask_source(self.rtsp_url),
                        exc,
                    )
                    self.close_process()
                    server_ok = False
                    last_check_time = 0.0
                    time.sleep(1.0)

        self.close_process()

    def close_process(self) -> None:
        if self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("FFmpeg streamer thread chưa dừng sau 3 giây.")
            else:
                self._thread = None
        self.close_process()
