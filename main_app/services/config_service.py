from pathlib import Path
from typing import List, Tuple, Union
import yaml
from pydantic import BaseModel, Field, ValidationError


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class ModelConfig(BaseModel):
    # Đường dẫn model: hỗ trợ .pt, OpenVINO folder (resources/models/yolov8n_openvino_model) hoặc .onnx
    weights: str = "resources/models/yolov8n_openvino_model"
    conf: float = Field(0.35, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    imgsz: int = 320  # Kích thước 320 hoặc 416 tối ưu CPU, giảm 60% tải tính toán so với 640
    skip_frames: int = 1  # Bỏ qua n frame giữa các lần infer AI (1: infer xen kẽ frame, video vẫn mượt 30fps)
    target_fps: int = 15  # Giới hạn FPS xử lý tối đa cho mỗi camera để tránh quá tải CPU
    classes: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    device: str = "cpu"


class DatabaseConfig(BaseModel):
    path: str = "resources/database/events.db"
    snapshot_dir: str = "resources/database/snapshots"


class LineCrossingConfig(BaseModel):
    enabled: bool = False
    point1: Tuple[int, int] = (0, 0)
    point2: Tuple[int, int] = (0, 0)


class RestrictedZoneConfig(BaseModel):
    enabled: bool = False
    polygon: List[Tuple[int, int]] = Field(default_factory=list)


class CameraConfig(BaseModel):
    id: str
    name: str
    source: Union[int, str]
    rtsp_out: str
    enabled: bool = True
    line_crossing: LineCrossingConfig = Field(default_factory=LineCrossingConfig)
    restricted_zone: RestrictedZoneConfig = Field(default_factory=RestrictedZoneConfig)


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cameras: List[CameraConfig] = Field(default_factory=list)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Đọc và kiểm tra tính hợp lệ của file cấu hình YAML (F09)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"File YAML cú pháp không hợp lệ ({config_path}): {exc}") from exc

    try:
        return AppConfig(**raw_data)
    except ValidationError as exc:
        raise ValueError(f"Lỗi cấu hình trong {config_path}:\n{exc}") from exc
