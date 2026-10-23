import logging

import yaml

from main_app.services import detector_service
from main_app.services.detector_service import YoloTracker


def make_tracker(imgsz=480):
    tracker = YoloTracker.__new__(YoloTracker)
    tracker.imgsz = imgsz
    return tracker


def test_existing_openvino_export_is_reused_for_matching_imgsz(tmp_path):
    model_dir = tmp_path / "yolov8s_openvino_model"
    model_dir.mkdir()
    (model_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"imgsz": [480, 480]}), encoding="utf-8"
    )
    weights = tmp_path / "yolov8s.pt"
    weights.write_bytes(b"model")

    resolved = make_tracker()._resolve_model_weights(str(weights))

    assert resolved == str(model_dir)


def test_pt_model_is_exported_to_openvino_when_needed(tmp_path, monkeypatch):
    weights = tmp_path / "yolov8s.pt"
    weights.write_bytes(b"model")
    exported_dir = tmp_path / "yolov8s_openvino_model"
    export_calls = []

    class FakeYOLO:
        def __init__(self, path):
            assert path == str(weights)

        def export(self, **kwargs):
            export_calls.append(kwargs)
            exported_dir.mkdir()
            return str(exported_dir)

    monkeypatch.setattr(detector_service, "YOLO", FakeYOLO)

    resolved = make_tracker()._resolve_model_weights(str(weights))

    assert resolved == str(exported_dir)
    assert export_calls == [
        {"format": "openvino", "imgsz": 480, "half": True}
    ]


def test_openvino_export_failure_warns_before_pytorch_fallback(
    tmp_path, monkeypatch, caplog
):
    weights = tmp_path / "yolov8s.pt"
    weights.write_bytes(b"model")

    class FakeYOLO:
        def __init__(self, path):
            pass

        def export(self, **kwargs):
            raise RuntimeError("conversion unavailable")

    monkeypatch.setattr(detector_service, "YOLO", FakeYOLO)

    with caplog.at_level(logging.WARNING, logger="services.detector"):
        resolved = make_tracker()._resolve_model_weights(str(weights))

    assert resolved == str(weights)
    assert "fallback sang model nguồn" in caplog.text
