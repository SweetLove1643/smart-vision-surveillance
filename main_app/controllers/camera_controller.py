import logging
import threading
import time
from typing import Dict, List, Optional, Set
import cv2
import numpy as np

from main_app.services import (
    AppConfig,
    CameraConfig,
    CameraReader,
    Detection,
    EventDatabase,
    FFmpegRTSPStreamer,
    LineCrossingCounter,
    YoloTracker,
    ZoneIntrusionDetector,
    LIVE,
    STOPPED,
    CONNECTING,
    RECONNECTING,
)
from main_app.services.config_service import ModelConfig

logger = logging.getLogger("controllers.camera")


class SingleCameraPipeline:
    """Pipeline xử lý độc lập cho từng camera: Capture -> AI -> Logic -> RTSP.
    
    Tối ưu hiệu năng:
    - Mỗi camera sở hữu riêng một instance YoloTracker (tránh xung đột dữ liệu bám vết ByteTrack).
    - Hỗ trợ skip_frames: suy luận AI xen kẽ frame và giữ lại box cho frame sau -> video mượt mà không quá tải CPU.
    - Điều tiết tốc độ khung hình (target_fps) để duy trì chuẩn >= 10 FPS và độ trễ < 1 giây.
    """

    def __init__(
        self,
        config: CameraConfig,
        tracker: YoloTracker,
        db: EventDatabase,
        model_config: ModelConfig,
    ) -> None:
        self.config = config
        self.camera_id = config.id
        self.name = config.name
        self.tracker = tracker
        self.db = db
        self.model_config = model_config

        self.reader = CameraReader(config.id, config.source, config.name)
        self.skip_frames = model_config.skip_frames
        self.target_fps = model_config.target_fps
        self.streamer = FFmpegRTSPStreamer(config.rtsp_out, fps=self.target_fps)

        self.line_counter: Optional[LineCrossingCounter] = None
        if config.line_crossing.enabled:
            persisted_counts = db.get_counts().get(config.id, {})
            self.line_counter = LineCrossingCounter(
                config.line_crossing.point1,
                config.line_crossing.point2,
                count_in=persisted_counts.get("count_in", 0),
                count_out=persisted_counts.get("count_out", 0),
            )

        self.zone_detector: Optional[ZoneIntrusionDetector] = None
        if config.restricted_zone.enabled and len(config.restricted_zone.polygon) >= 3:
            self.zone_detector = ZoneIntrusionDetector(config.restricted_zone.polygon)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Lưu frame đã vẽ gần nhất để UI Qt hoặc API lấy xem
        self._annotated_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._frame_count = 0
        self._last_detections: List[Detection] = []

        self.output_fps: float = 0.0
        self.latency_ms: float = 0.0
        self.inference_fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self._fps_count = 0
        self._fps_time = time.perf_counter()
        self._metrics_started_at = self._fps_time
        self._inference_count = 0
        self._inference_latency_total_ms = 0.0
        self.is_running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.is_running = True
        self._stop_event.clear()
        self._frame_count = 0
        self._last_detections.clear()
        self._fps_count = 0
        self._inference_count = 0
        self._inference_latency_total_ms = 0.0
        self._fps_time = time.perf_counter()
        self._metrics_started_at = self._fps_time
        self.output_fps = 0.0
        self.inference_fps = 0.0
        self.latency_ms = 0.0
        self.inference_latency_ms = 0.0
        if self.line_counter:
            self.line_counter.reset_tracks()
        if self.zone_detector:
            self.zone_detector.reset_tracks()
        self.streamer.start()
        self.reader.start()

        self._thread = threading.Thread(
            target=self._process_loop, name=f"pipe-{self.camera_id}", daemon=True
        )
        self._thread.start()
        logger.info("[%s] Khởi động camera pipeline.", self.camera_id)

    def stop(self) -> None:
        if not self.is_running and (
            self._thread is None or not self._thread.is_alive()
        ):
            return
        self._stop_event.set()
        self.reader.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "[%s] Pipeline thread chưa kết thúc sau 5 giây; "
                    "sẽ không tạo thread mới cho đến khi thread hiện tại dừng.",
                    self.camera_id,
                )
            else:
                self._thread = None
        self.streamer.close()
        if self._thread is None:
            self.is_running = False
            logger.info("[%s] Dừng camera pipeline an toàn.", self.camera_id)

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._annotated_frame is None else self._annotated_frame.copy()

    def _process_loop(self) -> None:
        min_frame_time = 1.0 / self.target_fps
        try:
            while not self._stop_event.is_set():
                t_start = time.perf_counter()
                raw_frame = self.reader.read_latest_frame()

                if raw_frame is None:
                    time.sleep(0.01)
                    continue

                should_infer = self._frame_count % (self.skip_frames + 1) == 0
                self._frame_count += 1

                if should_infer:
                    inference_started_at = time.perf_counter()
                    detections = self.tracker.track(raw_frame)
                    inference_elapsed = time.perf_counter() - inference_started_at
                    self._inference_latency_total_ms += inference_elapsed * 1000
                    self._inference_count += 1
                    self.inference_latency_ms = round(
                        self._inference_latency_total_ms / self._inference_count, 1
                    )
                    self._last_detections = detections
                    active_ids = {d.track_id for d in detections}
                    self._process_events(raw_frame, detections, active_ids)
                else:
                    detections = self._last_detections

                annotated = self._draw(raw_frame.copy(), detections)
                with self._frame_lock:
                    self._annotated_frame = annotated

                self.streamer.push_frame(annotated)

                self.latency_ms = round(
                    (time.perf_counter() - t_start) * 1000, 1
                )
                self._fps_count += 1
                now = time.perf_counter()
                fps_elapsed = now - self._fps_time
                if fps_elapsed >= 1.0:
                    self.output_fps = round(self._fps_count / fps_elapsed, 1)
                    self._fps_count = 0
                    self._fps_time = now

                metrics_elapsed = now - self._metrics_started_at
                if metrics_elapsed >= self.model_config.metrics_log_interval_seconds:
                    self.inference_fps = round(
                        self._inference_count / metrics_elapsed, 2
                    )
                    logger.info(
                        "[%s] Performance: input_fps=%.1f output_fps=%.1f "
                        "inference_fps=%.2f inference_latency_ms=%.1f "
                        "processing_latency_ms=%.1f",
                        self.camera_id,
                        self.reader.fps,
                        self.output_fps,
                        self.inference_fps,
                        self.inference_latency_ms,
                        self.latency_ms,
                    )
                    self._inference_count = 0
                    self._inference_latency_total_ms = 0.0
                    self._metrics_started_at = now

                elapsed = time.perf_counter() - t_start
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
        finally:
            self.is_running = False

    def _process_events(
        self, frame: np.ndarray, detections: List[Detection], active_ids: Set[int]
    ) -> None:
        for det in detections:
            # Đếm cắt vạch
            if self.line_counter:
                direction = self.line_counter.update(det.track_id, det.bottom_center)
                if direction:
                    _, count_in, count_out = self.db.save_line_crossing(
                        camera_id=self.camera_id,
                        track_id=det.track_id,
                        direction=direction,
                        details=f"Direction: {direction}, Class: {det.class_name}",
                        frame=frame,
                    )
                    self.line_counter.count_in = count_in
                    self.line_counter.count_out = count_out

            # Xâm nhập vùng cấm
            if self.zone_detector:
                if self.zone_detector.update(det.track_id, det.bottom_center):
                    self.db.save_event(
                        camera_id=self.camera_id,
                        track_id=det.track_id,
                        event_type="zone_intrusion",
                        details=f"Class: {det.class_name}, Conf: {det.conf}",
                        frame=frame,
                    )

        if self.line_counter:
            self.line_counter.clean_old_tracks(active_ids)
        if self.zone_detector:
            self.zone_detector.clean_old_tracks(active_ids)

    def _draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        # Vẽ vùng cấm
        if self.config.restricted_zone.enabled and len(self.config.restricted_zone.polygon) >= 3:
            pts = np.array(self.config.restricted_zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "RESTRICTED ZONE", (pts[0][0][0], pts[0][0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Vẽ vạch đếm
        if self.config.line_crossing.enabled:
            p1, p2 = self.config.line_crossing.point1, self.config.line_crossing.point2
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
            if self.line_counter:
                count_str = f"IN: {self.line_counter.count_in} | OUT: {self.line_counter.count_out}"
                cv2.putText(frame, count_str, (p1[0], p1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Vẽ bounding box & ID
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = (0, 255, 0) if det.class_name == "person" else (255, 120, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{det.track_id} {det.class_name} {det.conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị thông số trên góc
        info = f"{self.name} | FPS: {self.output_fps} | {self.latency_ms}ms"
        cv2.putText(frame, info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def get_status(self) -> dict:
        count_in = self.line_counter.count_in if self.line_counter else 0
        count_out = self.line_counter.count_out if self.line_counter else 0
        return {
            "id": self.camera_id,
            "name": self.name,
            "enabled": self.is_running,
            "state": self.reader.state,
            "input_fps": self.reader.fps,
            "output_fps": self.output_fps,
            "inference_fps": self.inference_fps,
            "inference_latency_ms": self.inference_latency_ms,
            "latency_ms": self.latency_ms,
            "count_in": count_in,
            "count_out": count_out,
            "reconnect_count": self.reader.reconnect_count,
            "rtsp_out": self.config.rtsp_out,
        }


class CameraController:
    """Bộ điều khiển quản lý toàn bộ các Camera Pipelines cho cả UI Qt và FastAPI."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.db = EventDatabase(config.database.path, config.database.snapshot_dir)
        self.pipelines: Dict[str, SingleCameraPipeline] = {}
        self.camera_order: List[str] = []

        # Mỗi camera khởi tạo một instance YoloTracker riêng biệt để độc lập trạng thái bám vết (F04)
        for c_cfg in config.cameras:
            self.db.ensure_camera(c_cfg.id)
            tracker = YoloTracker(
                weights=config.model.weights,
                conf=config.model.conf,
                iou=config.model.iou,
                imgsz=config.model.imgsz,
                classes=config.model.classes,
                device=config.model.device,
            )
            self.pipelines[c_cfg.id] = SingleCameraPipeline(c_cfg, tracker, self.db, config.model)
            self.camera_order.append(c_cfg.id)

    def start_all(self) -> None:
        for pipe in self.pipelines.values():
            if pipe.config.enabled:
                pipe.start()

    def stop_all(self) -> None:
        for pipe in self.pipelines.values():
            pipe.stop()

    def start_camera(self, camera_id: str) -> bool:
        pipe = self.pipelines.get(camera_id)
        if pipe:
            pipe.start()
            return True
        return False

    def stop_camera(self, camera_id: str) -> bool:
        pipe = self.pipelines.get(camera_id)
        if pipe:
            pipe.stop()
            return True
        return False

    def get_all_status(self) -> List[dict]:
        return [self.pipelines[cid].get_status() for cid in self.camera_order]

    def get_frame_by_id(self, camera_id: str) -> Optional[np.ndarray]:
        pipe = self.pipelines.get(camera_id)
        return pipe.get_latest_annotated_frame() if pipe else None

    def get_frame_by_index(self, index: int) -> Optional[np.ndarray]:
        if 0 <= index < len(self.camera_order):
            return self.get_frame_by_id(self.camera_order[index])
        return None

    def get_state_by_index(self, index: int) -> str:
        if 0 <= index < len(self.camera_order):
            pipe = self.pipelines[self.camera_order[index]]
            return pipe.reader.state if pipe.is_running else STOPPED
        return STOPPED

    def get_reconnect_count_by_index(self, index: int) -> int:
        if 0 <= index < len(self.camera_order):
            return self.pipelines[self.camera_order[index]].reader.reconnect_count
        return 0
