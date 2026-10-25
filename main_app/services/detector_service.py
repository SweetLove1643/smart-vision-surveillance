from concurrent.futures import Future
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

logger = logging.getLogger("services.detector")

torch.set_num_threads(2)


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    class_name: str
    conf: float
    box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection(RawDetection):
    track_id: int
    bottom_center: Tuple[float, float]


@dataclass
class _InferenceRequest:
    camera_id: str
    frame: np.ndarray
    result: Future[List[RawDetection]]


class YoloInferenceService:
    """One shared YOLO model behind a bounded, serialized inference queue."""

    def __init__(
        self,
        weights: str = "resources/models/yolov8s_openvino_model",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 480,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        queue_size: int = 4,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes or [0, 1, 2, 3, 5, 7]
        self.device = device
        self.request_timeout_seconds = request_timeout_seconds
        if conf > 0.25:
            logger.warning(
                "model.conf=%.2f loại detection confidence thấp trước ByteTrack; "
                "có thể làm giảm khả năng giữ track qua frame bị che khuất.",
                conf,
            )
        self.weights = self._resolve_model_weights(weights)
        logger.info(
            "Khởi tạo shared YOLO inference service: %s "
            "(imgsz=%d, device=%s, conf=%.2f)",
            self.weights,
            imgsz,
            device,
            conf,
        )
        self.model = YOLO(self.weights)
        self.names = self.model.names

        self._requests: queue.Queue[Optional[_InferenceRequest]] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._processed_requests = 0
        self._total_latency_ms = 0.0
        self._last_latency_ms = 0.0
        self._last_error: Optional[str] = None

    def _resolve_model_weights(self, weights: str) -> str:
        """Reuse a matching OpenVINO export or export the configured source model."""
        path = Path(weights).expanduser()
        if path.is_dir():
            if self._openvino_imgsz_matches(path):
                logger.info("Sử dụng model OpenVINO có sẵn: %s", path)
                return str(path)
            if not path.name.lower().endswith("_openvino_model"):
                raise ValueError(f"Thư mục model OpenVINO không hợp lệ: {path}")

        if path.suffix.lower() in {".pt", ".onnx"}:
            source_path = path if path.is_file() else None
            openvino_dir = path.parent / f"{path.stem}_openvino_model"
        elif path.name.lower().endswith("_openvino_model"):
            openvino_dir = path
            source_stem = path.name[: -len("_openvino_model")]
            source_path = next(
                (
                    candidate
                    for candidate in (
                        path.with_name(f"{source_stem}.pt"),
                        path.with_name(f"{source_stem}.onnx"),
                    )
                    if candidate.is_file()
                ),
                None,
            )
        else:
            raise FileNotFoundError(
                f"Không tìm thấy model '{weights}'. Hãy chỉ định file .pt/.onnx "
                "hoặc thư mục OpenVINO hợp lệ."
            )

        if openvino_dir.is_dir() and self._openvino_imgsz_matches(openvino_dir):
            logger.info("Sử dụng model OpenVINO đã convert: %s", openvino_dir)
            return str(openvino_dir)
        if source_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy model nguồn '{path}' hoặc OpenVINO model phù hợp "
                f"cho imgsz={self.imgsz}."
            )

        logger.info(
            "Đang export %s sang OpenVINO (imgsz=%d, FP16)...",
            source_path,
            self.imgsz,
        )
        try:
            exported_dir = YOLO(str(source_path)).export(
                format="openvino", imgsz=self.imgsz, half=True
            )
            exported_path = Path(exported_dir)
            if not exported_path.is_dir():
                raise RuntimeError(f"Exporter không tạo thư mục OpenVINO: {exported_dir}")
            logger.info("Export OpenVINO thành công: %s", exported_path)
            return str(exported_path)
        except Exception as exc:
            logger.warning(
                "Export OpenVINO thất bại cho %s: %s. "
                "Fallback sang model nguồn; inference sẽ không chạy bằng OpenVINO.",
                source_path,
                exc,
            )
            return str(source_path)

    def _openvino_imgsz_matches(self, model_dir: Path) -> bool:
        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.is_file():
            return False
        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = yaml.safe_load(metadata_file) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Không đọc được metadata OpenVINO %s: %s", metadata_path, exc)
            return False

        exported_imgsz = metadata.get("imgsz")
        if isinstance(exported_imgsz, int):
            return exported_imgsz == self.imgsz
        if isinstance(exported_imgsz, (list, tuple)):
            return bool(exported_imgsz) and all(
                size == self.imgsz for size in exported_imgsz
            )
        return False

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._run,
                name="yolo-inference",
                daemon=True,
            )
            self._worker.start()
            logger.info("Shared YOLO inference queue đã khởi động.")

    def submit(
        self, frame: np.ndarray, camera_id: str = "unknown"
    ) -> Future[List[RawDetection]]:
        future: Future[List[RawDetection]] = Future()
        request = _InferenceRequest(
            camera_id=camera_id, frame=frame, result=future
        )
        with self._lifecycle_lock:
            if self._worker is None or not self._worker.is_alive():
                raise InferenceUnavailableError(
                    "Shared YOLO inference service chưa được khởi động."
                )
            if self._stop_event.is_set():
                raise InferenceUnavailableError(
                    "Shared YOLO inference service đang dừng."
                )
            try:
                self._requests.put_nowait(request)
            except queue.Full as exc:
                raise InferenceQueueFullError(
                    "Inference queue đang quá tải."
                ) from exc
        return future

    def infer(
        self, frame: np.ndarray, camera_id: str = "unknown"
    ) -> List[RawDetection]:
        future = self.submit(frame, camera_id)
        try:
            return future.result(timeout=self.request_timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Inference của camera {camera_id} quá thời gian chờ."
            ) from exc

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            try:
                if request is None:
                    return
                if not request.result.set_running_or_notify_cancel():
                    continue
                started_at = time.perf_counter()
                try:
                    logger.debug(
                        "Inference request camera=%s", request.camera_id
                    )
                    detections = self._predict(request.frame)
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    self._processed_requests += 1
                    self._last_latency_ms = elapsed_ms
                    self._total_latency_ms += elapsed_ms
                    self._last_error = None
                    request.result.set_result(detections)
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.exception("Shared YOLO inference request thất bại.")
                    request.result.set_exception(exc)
            finally:
                self._requests.task_done()

    def _predict(self, frame: np.ndarray) -> List[RawDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: List[RawDetection] = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections.append(
                RawDetection(
                    class_id=class_id,
                    class_name=self.names.get(class_id, str(class_id)),
                    conf=round(float(box.conf[0].item()), 4),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    def get_status(self) -> dict:
        return {
            "model": self.weights,
            "worker_running": self._worker is not None and self._worker.is_alive(),
            "queue_size": self._requests.qsize(),
            "queue_capacity": self._requests.maxsize,
            "processed_requests": self._processed_requests,
            "last_latency_ms": round(self._last_latency_ms, 1),
            "average_latency_ms": round(
                self._total_latency_ms / self._processed_requests, 1
            )
            if self._processed_requests
            else 0.0,
            "last_error": self._last_error,
        }

    def stop(self, timeout: float = 15.0) -> None:
        with self._lifecycle_lock:
            worker = self._worker
            if worker is None or not worker.is_alive():
                self._worker = None
                return
            self._stop_event.set()
            while True:
                try:
                    pending_request = self._requests.get_nowait()
                except queue.Empty:
                    break
                try:
                    if pending_request is not None and not pending_request.result.done():
                        pending_request.result.set_exception(
                            InferenceUnavailableError(
                                "Inference service is shutting down."
                            )
                        )
                finally:
                    self._requests.task_done()
            try:
                self._requests.put(None, timeout=timeout)
            except queue.Full as exc:
                raise RuntimeError(
                    "Không thể dừng inference service vì queue đang đầy."
                ) from exc
            worker.join(timeout=timeout)
            if worker.is_alive():
                raise TimeoutError("Shared YOLO inference worker chưa dừng đúng hạn.")
            self._worker = None
        logger.info("Shared YOLO inference queue đã dừng.")


class CameraTracker:
    """Lightweight per-camera ByteTrack state over shared-model detections."""

    def __init__(self, tracker_config: Optional[str] = None) -> None:
        if tracker_config is None:
            tracker_config = str(
                Path(ultralytics.__file__).parent
                / "cfg"
                / "trackers"
                / "bytetrack.yaml"
            )
        self._tracker = BYTETracker(get_cfg(tracker_config))

    def track(
        self,
        detections: List[RawDetection],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        height, width = frame_shape[:2]
        boxes_array = np.asarray(
            [
                [*detection.box, detection.conf, detection.class_id]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 6))
        boxes = Boxes(boxes_array, orig_shape=(height, width)).numpy()
        tracks = self._tracker.update(boxes)

        tracked_detections: List[Detection] = []
        for x1, y1, x2, y2, track_id, conf, class_id, detection_index in tracks:
            index = int(detection_index)
            detection = detections[index]
            left, top, right, bottom = map(int, (x1, y1, x2, y2))
            tracked_detections.append(
                Detection(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=detection.class_name,
                    conf=round(float(conf), 4),
                    box=(left, top, right, bottom),
                    bottom_center=((left + right) / 2.0, float(bottom)),
                )
            )
        return tracked_detections

    def reset(self) -> None:
        self._tracker.reset()


class InferenceUnavailableError(RuntimeError):
    """The inference worker cannot currently accept work."""


class InferenceQueueFullError(InferenceUnavailableError):
    """Inference requests could not be accepted within the queue timeout."""
