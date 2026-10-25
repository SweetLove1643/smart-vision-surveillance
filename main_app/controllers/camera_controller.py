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
    CameraTracker,
    Detection,
    EventDatabase,
    FFmpegRTSPStreamer,
    LineCrossingCounter,
    YoloInferenceService,
    ZoneIntrusionDetector,
    STOPPED,
)
from main_app.services.config_service import ModelConfig
from main_app.services.detector_service import (
    InferenceQueueFullError,
    InferenceUnavailableError,
)
from main_app.services.inference_api import YoloInferenceAPIServer
from main_app.services.inference_client import YoloInferenceClient

logger = logging.getLogger("controllers.camera")


class SingleCameraPipeline:
    """Pipeline xử lý độc lập cho từng camera: Capture -> AI -> Logic -> RTSP.
    
    Tối ưu hiệu năng:
    - Dùng chung inference service; mỗi camera có ByteTrack state riêng.
    - Hỗ trợ skip_frames: suy luận AI xen kẽ frame và giữ lại box cho frame sau -> video mượt mà không quá tải CPU.
    - Điều tiết đầu ra không vượt quá target_fps; thông lượng thực tế phụ thuộc model và CPU.
    """

    def __init__(
        self,
        config: CameraConfig,
        inference_client: YoloInferenceClient,
        db: EventDatabase,
        model_config: ModelConfig,
        enable_rtsp: bool = True,
    ) -> None:
        self.config = config
        self.camera_id = config.id
        self.name = config.name
        self.inference_client = inference_client
        self.tracker = CameraTracker()
        self.db = db
        self.model_config = model_config

        self.reader = CameraReader(config.id, config.source, config.name)
        self.output_reader = (
            CameraReader(config.id, config.rtsp_out, f"{config.name} RTSP output")
            if enable_rtsp
            else None
        )
        self.skip_frames = model_config.skip_frames
        self.target_fps = model_config.target_fps
        self.streamer = FFmpegRTSPStreamer(
            config.rtsp_out, fps=self.target_fps, enabled=enable_rtsp
        )

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
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._inference_wakeup = threading.Event()
        self._inference_frame_lock = threading.Lock()
        self._latest_inference_frame: Optional[np.ndarray] = None
        self._detections_lock = threading.Lock()

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
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            inference_thread = self._inference_thread
            output_alive = thread is not None and thread.is_alive()
            inference_alive = (
                inference_thread is not None and inference_thread.is_alive()
            )
            if output_alive or inference_alive:
                if self._stop_event.is_set() or not (
                    output_alive and inference_alive
                ):
                    raise RuntimeError(
                        f"Camera pipeline {self.camera_id} still has a worker running."
                    )
                return
            self._stop_event.clear()
            self._inference_wakeup.clear()
            with self._inference_frame_lock:
                self._latest_inference_frame = None
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
            if self.output_reader:
                self.output_reader.start()
            self.is_running = True
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                name=f"infer-{self.camera_id}",
                daemon=True,
            )
            self._inference_thread.start()
            self._thread = threading.Thread(
                target=self._process_loop, name=f"pipe-{self.camera_id}", daemon=True
            )
            self._thread.start()
        logger.info("[%s] Khởi động camera pipeline.", self.camera_id)

    def stop(self) -> None:
        self.request_stop()
        with self._lifecycle_lock:
            thread = self._thread
            inference_thread = self._inference_thread
            if (
                (thread is None or not thread.is_alive())
                and (inference_thread is None or not inference_thread.is_alive())
            ):
                self._thread = None
                self._inference_thread = None
                self.is_running = False
                return
        self.reader.stop()
        if self.output_reader:
            self.output_reader.stop()
        for worker in (thread, inference_thread):
            if worker is None or worker is threading.current_thread():
                continue
            worker.join(timeout=15.0)
            if worker.is_alive():
                logger.warning(
                    "[%s] Thread %s chưa kết thúc sau 15 giây; "
                    "sẽ không tạo lại pipeline cho đến khi thread hiện tại dừng.",
                    self.camera_id,
                    worker.name,
                )
        with self._lifecycle_lock:
            if thread is not None and not thread.is_alive() and self._thread is thread:
                self._thread = None
            if (
                inference_thread is not None
                and not inference_thread.is_alive()
                and self._inference_thread is inference_thread
            ):
                self._inference_thread = None
            if self._thread is None and self._inference_thread is None:
                self.is_running = False
        if not self.is_running:
            logger.info("[%s] Dừng camera pipeline an toàn.", self.camera_id)

    def request_stop(self) -> None:
        self._stop_event.set()
        self._inference_wakeup.set()

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        return (
            self.output_reader.read_latest_frame()
            if self.output_reader is not None
            else None
        )

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
                    with self._inference_frame_lock:
                        self._latest_inference_frame = raw_frame
                    self._inference_wakeup.set()

                with self._detections_lock:
                    detections = list(self._last_detections)

                annotated = self._draw(raw_frame.copy(), detections)
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

                elapsed = time.perf_counter() - t_start
                if elapsed < min_frame_time:
                    self._stop_event.wait(min_frame_time - elapsed)
        except Exception:
            logger.exception(
                "[%s] Pipeline bị dừng do lỗi không được xử lý.", self.camera_id
            )
        finally:
            self._stop_event.set()
            self._inference_wakeup.set()
            self.reader.stop()
            if self.output_reader:
                self.output_reader.stop()
            self.streamer.close()
            with self._lifecycle_lock:
                self.is_running = False

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            self._inference_wakeup.wait()
            self._inference_wakeup.clear()
            if self._stop_event.is_set():
                return
            with self._inference_frame_lock:
                frame = self._latest_inference_frame
                self._latest_inference_frame = None
            if frame is None:
                continue

            inference_started_at = time.perf_counter()
            try:
                raw_detections = self.inference_client.infer(frame)
            except (InferenceQueueFullError, InferenceUnavailableError, TimeoutError) as exc:
                logger.warning(
                    "[%s] Bỏ frame inference do API bận hoặc chưa sẵn sàng: %s",
                    self.camera_id,
                    exc,
                )
                continue
            except Exception:
                logger.exception(
                    "[%s] Inference/tracking lỗi; bỏ detections của frame này.",
                    self.camera_id,
                )
                continue

            if self._stop_event.is_set():
                return
            detections = self.tracker.track(raw_detections, frame.shape)

            inference_elapsed = time.perf_counter() - inference_started_at
            self._inference_latency_total_ms += inference_elapsed * 1000
            self._inference_count += 1
            self.inference_latency_ms = round(
                self._inference_latency_total_ms / self._inference_count, 1
            )
            with self._detections_lock:
                self._last_detections = detections
            active_ids = {d.track_id for d in detections}
            try:
                self._process_events(frame, detections, active_ids)
            except Exception:
                logger.exception(
                    "[%s] Event analytics failed for the inferred frame.",
                    self.camera_id,
                )

            now = time.perf_counter()
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
            "rtsp_state": (
                self.output_reader.state if self.output_reader else STOPPED
            ),
            "rtsp_enabled": self.output_reader is not None,
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

    def __init__(self, config: AppConfig, enable_rtsp: bool = True) -> None:
        self.config = config
        self.enable_rtsp = enable_rtsp
        self._inference_shutdown_timeout = (
            config.model.inference_timeout_seconds + 10
        )
        self.db = EventDatabase(config.database.path, config.database.snapshot_dir)
        self.pipelines: Dict[str, SingleCameraPipeline] = {}
        self.camera_order: List[str] = []
        self._lifecycle_lock = threading.RLock()
        self._stopping = False

        self.inference_service = YoloInferenceService(
            weights=config.model.weights,
            conf=config.model.conf,
            iou=config.model.iou,
            imgsz=config.model.imgsz,
            classes=config.model.classes,
            device=config.model.device,
            queue_size=max(1, len(config.cameras)),
            request_timeout_seconds=config.model.inference_timeout_seconds,
        )
        self.inference_api = YoloInferenceAPIServer(
            self.inference_service,
            config.model.inference_api_host,
            config.model.inference_api_port,
        )
        inference_url = (
            f"http://{config.model.inference_api_host}:"
            f"{config.model.inference_api_port}"
        )

        for c_cfg in config.cameras:
            self.db.ensure_camera(c_cfg.id)
            self.pipelines[c_cfg.id] = SingleCameraPipeline(
                c_cfg,
                YoloInferenceClient(
                    inference_url,
                    c_cfg.id,
                    timeout_seconds=config.model.inference_timeout_seconds + 5,
                ),
                self.db,
                config.model,
                enable_rtsp=enable_rtsp,
            )
            self.camera_order.append(c_cfg.id)

    def start_all(self, camera_ids: Optional[List[str]] = None) -> None:
        with self._lifecycle_lock:
            selected_ids = set(camera_ids) if camera_ids is not None else None
            unknown_ids = (
                selected_ids.difference(self.pipelines)
                if selected_ids is not None
                else set()
            )
            if unknown_ids:
                raise ValueError(
                    "Không tìm thấy camera được chọn: "
                    + ", ".join(sorted(unknown_ids))
                )
            disabled_ids = (
                {
                    camera_id
                    for camera_id in selected_ids
                    if not self.pipelines[camera_id].config.enabled
                }
                if selected_ids is not None
                else set()
            )
            if disabled_ids:
                raise ValueError(
                    "Camera được chọn đang bị tắt trong cấu hình: "
                    + ", ".join(sorted(disabled_ids))
                )
            self.inference_api.start()
            try:
                for camera_id, pipe in self.pipelines.items():
                    if pipe.config.enabled and (
                        selected_ids is None or camera_id in selected_ids
                    ):
                        pipe.start()
            except Exception:
                logger.exception("Camera startup failed; stopping started pipelines.")
                for pipe in self.pipelines.values():
                    pipe.request_stop()
                for pipe in self.pipelines.values():
                    pipe.stop()
                self.inference_api.stop(timeout=self._inference_shutdown_timeout)
                raise

    def stop_all(self) -> None:
        with self._lifecycle_lock:
            if self._stopping:
                return
            self._stopping = True
            try:
                for pipe in self.pipelines.values():
                    pipe.request_stop()
                for pipe in self.pipelines.values():
                    pipe.stop()
                self.inference_api.stop(timeout=self._inference_shutdown_timeout)
            finally:
                self._stopping = False

    def start_camera(self, camera_id: str) -> bool:
        pipe = self.pipelines.get(camera_id)
        if pipe:
            with self._lifecycle_lock:
                self.inference_api.start()
                pipe.start()
            return True
        return False

    def stop_camera(self, camera_id: str) -> bool:
        pipe = self.pipelines.get(camera_id)
        if pipe:
            with self._lifecycle_lock:
                pipe.stop()
                if not any(
                    running_pipe.is_running
                    for running_pipe in self.pipelines.values()
                ):
                    self.inference_api.stop(
                        timeout=self._inference_shutdown_timeout
                    )
            return True
        return False

    def get_all_status(self) -> List[dict]:
        return [self.pipelines[cid].get_status() for cid in self.camera_order]

    def get_inference_status(self) -> dict:
        status = self.inference_service.get_status()
        status["api_url"] = (
            f"http://{self.config.model.inference_api_host}:"
            f"{self.config.model.inference_api_port}"
        )
        status["api_running"] = self.inference_api.is_running
        return status

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
            if not pipe.is_running or pipe.output_reader is None:
                return STOPPED
            return pipe.output_reader.state
        return STOPPED

    def get_reconnect_count_by_index(self, index: int) -> int:
        if 0 <= index < len(self.camera_order):
            return self.pipelines[self.camera_order[index]].reader.reconnect_count
        return 0
