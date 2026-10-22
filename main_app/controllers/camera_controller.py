import logging
import numpy as np
from main_app.services.capture import CONNECTING, LIVE, RECONNECTING, STOPPED, CameraReader

__all__ = ["CameraController", "CONNECTING", "LIVE", "RECONNECTING", "STOPPED"]  # UI lấy hằng trạng thái qua đây

log = logging.getLogger("controller")

class CameraController:
    """Điều khiển nhóm camera. UI chỉ gọi lớp này, không biết CameraReader / thread / OpenCV.

    Camera được đánh số theo vị trí (index) trên lưới: 0, 1, 2, ...
    """

    def __init__(self) -> None:
        # Cùng thứ tự với các ô trên lưới; None = ô đó không có nguồn nên không có reader.
        self._readers: list[CameraReader | None] = []
        self.running = False

    # ---------- Bật / tắt ----------
    def start(self, cameras: list[tuple[str, int | str | None]]) -> None:
        """cameras: danh sách (tên, nguồn). Nguồn None thì bỏ qua ô đó."""
        self.stop()  # bấm Start hai lần cũng không bị chạy trùng
        self._readers = [
            CameraReader(source, name).start() if source is not None else None
            for name, source in cameras
        ]
        self.running = True
        log.info("Đã bật %d camera", sum(r is not None for r in self._readers))

    def stop(self) -> None:
        active = [r for r in self._readers if r is not None]
        for reader in active:
            reader.request_stop()  # báo dừng tất cả trước...
        for reader in active:
            reader.stop()          # ...rồi mới chờ từng thread, tổng thời gian chờ ngắn hơn
        if self.running:
            log.info("Đã dừng %d camera", len(active))
        self._readers = []
        self.running = False

    # ---------- Lấy dữ liệu cho UI ----------
    def get_state(self, index: int) -> str:
        """Trạng thái camera thứ index: STOPPED / CONNECTING / LIVE / RECONNECTING."""
        reader = self._reader(index)
        return reader.state if reader is not None else STOPPED

    def get_frame(self, index: int) -> np.ndarray | None:
        """Frame mới nhất của camera thứ index, hoặc None nếu chưa có / đang mất kết nối."""
        reader = self._reader(index)
        return reader.read() if reader is not None else None

    def get_reconnect_count(self, index: int) -> int:
        reader = self._reader(index)
        return reader.reconnect_count if reader is not None else 0

    def _reader(self, index: int) -> CameraReader | None:
        return self._readers[index] if 0 <= index < len(self._readers) else None