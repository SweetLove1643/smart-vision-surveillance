from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

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
    
    Hỗ trợ:
    - Model gốc PyTorch (.pt)
    - Model OpenVINO (folder `resources/models/yolov8n_openvino_model`) tối ưu CPU
    - Model ONNX (.onnx)
    """

    def __init__(
        self,
        weights: str = "resources/models/yolov8n_openvino_model",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes or [0, 1, 2, 3, 5, 7]
        self.device = device

        resolved_weights = self._resolve_model_weights(weights)
        logger.info("Khởi tạo YOLO tracker với: %s (imgsz=%d, device=%s)", resolved_weights, imgsz, device)
        self.model = YOLO(resolved_weights)

    def _resolve_model_weights(self, weights: str) -> str:
        """Kiểm tra và tự động chuyển đổi sang OpenVINO nếu cần."""
        path = Path(weights)

        # 1. Nếu đường dẫn chỉ định đã tồn tại (folder OpenVINO hoặc file .pt / .onnx)
        if path.exists():
            return str(path)

        # 2. Nếu đang chỉ định OpenVINO nhưng chưa có, thử tìm .pt để convert
        pt_path = Path("resources/models/yolov8n.pt")
        if not pt_path.exists():
            pt_path = Path("yolov8n.pt")

        if pt_path.exists() and "openvino" in weights.lower():
            logger.info("Chưa có OpenVINO model, đang tự động export từ: %s ...", pt_path)
            try:
                base_model = YOLO(str(pt_path))
                exported_dir = base_model.export(format="openvino", imgsz=self.imgsz, half=True)
                logger.info("Export OpenVINO thành công: %s", exported_dir)
                return str(exported_dir)
            except Exception as exc:
                logger.warning("Không thể export sang OpenVINO (%s), dùng tạm file .pt gốc.", exc)
                return str(pt_path)

        # 3. Fallback về .pt
        if pt_path.exists():
            return str(pt_path)

        return weights

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
