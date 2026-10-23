from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
import yaml

logger = logging.getLogger("services.detector")

# Giới hạn số thread cho PyTorch trên CPU để tránh cạnh tranh tài nguyên giữa các camera
torch.set_num_threads(2)


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    conf: float
    box: Tuple[int, int, int, int]
    bottom_center: Tuple[float, float]


class YoloTracker:
    """Nhận diện & bám vết đối tượng đa luồng dùng YOLOv8 (F03, F04).
    
    Model .pt/.onnx được tự động export và dùng dưới dạng OpenVINO trên CPU.
    """

    def __init__(
        self,
        weights: str = "resources/models/yolov8s_openvino_model",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 480,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes or [0, 1, 2, 3, 5, 7]
        self.device = device

        resolved_weights = self._resolve_model_weights(weights)
        logger.info(
            "Khởi tạo YOLO tracker với: %s (imgsz=%d, device=%s)",
            resolved_weights,
            imgsz,
            device,
        )
        self.model = YOLO(resolved_weights)

    def _resolve_model_weights(self, weights: str) -> str:
        """Resolve an explicit OpenVINO model or export a source model for CPU use."""
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
                "Đang fallback sang model nguồn; inference sẽ không chạy bằng OpenVINO.",
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

    def track(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0].item())
            class_id = int(box.cls[0].item())
            class_name = self.model.names.get(class_id, str(class_id))
            conf = float(box.conf[0].item())

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            bottom_center = ((x1 + x2) / 2.0, float(y2))

            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    conf=round(conf, 2),
                    box=(x1, y1, x2, y2),
                    bottom_center=bottom_center,
                )
            )

        return detections
