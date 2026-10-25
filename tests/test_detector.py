import logging
import threading
import time

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

from main_app.services import detector_service
from main_app.services.detector_service import (
    CameraTracker,
    InferenceQueueFullError,
    RawDetection,
    YoloInferenceService,
)
from main_app.services.inference_api import create_inference_app


def make_service(imgsz=480):
    service = YoloInferenceService.__new__(YoloInferenceService)
    service.imgsz = imgsz
    return service


def test_existing_openvino_export_is_reused_for_matching_imgsz(tmp_path):
    model_dir = tmp_path / "yolov8s_openvino_model"
    model_dir.mkdir()
    (model_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"imgsz": [480, 480]}), encoding="utf-8"
    )
    weights = tmp_path / "yolov8s.pt"
    weights.write_bytes(b"model")

    resolved = make_service()._resolve_model_weights(str(weights))

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

    resolved = make_service()._resolve_model_weights(str(weights))

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
        resolved = make_service()._resolve_model_weights(str(weights))

    assert resolved == str(weights)
    assert "Fallback sang model nguồn" in caplog.text


def test_shared_inference_service_serializes_concurrent_requests(monkeypatch):
    model_instances = []

    class FakeYOLO:
        names = {0: "person"}

        def __init__(self, path):
            model_instances.append(self)

    monkeypatch.setattr(detector_service, "YOLO", FakeYOLO)
    monkeypatch.setattr(
        YoloInferenceService, "_resolve_model_weights", lambda self, weights: weights
    )
    service = YoloInferenceService(weights="fake-openvino")
    active_predictions = 0
    max_active_predictions = 0
    result_lock = threading.Lock()
    results = []
    errors = []
    start_gate = threading.Barrier(5)

    def fake_predict(frame):
        nonlocal active_predictions, max_active_predictions
        with result_lock:
            active_predictions += 1
            max_active_predictions = max(max_active_predictions, active_predictions)
        time.sleep(0.02)
        with result_lock:
            active_predictions -= 1
        return [RawDetection(0, "person", 0.9, (0, 0, 10, 10))]

    service._predict = fake_predict

    def request_inference():
        try:
            start_gate.wait()
            results.append(service.infer(np.zeros((8, 8, 3), dtype=np.uint8)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=request_inference) for _ in range(4)]
    service.start()
    try:
        for thread in threads:
            thread.start()
        start_gate.wait()
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert len(results) == 4
        assert max_active_predictions == 1
        assert len(model_instances) == 1
        assert service.get_status()["processed_requests"] == 4
    finally:
        service.stop()


def test_trackers_keep_separate_state_per_camera():
    first_camera_tracker = CameraTracker()
    second_camera_tracker = CameraTracker()
    detection = RawDetection(2, "car", 0.9, (20, 20, 80, 100))

    first_tracks = first_camera_tracker.track([detection], (120, 120, 3))
    second_tracks = second_camera_tracker.track([detection], (120, 120, 3))

    assert first_camera_tracker._tracker is not second_camera_tracker._tracker
    assert len(first_tracks) == 1
    assert len(second_tracks) == 1
    assert first_tracks[0].track_id == second_tracks[0].track_id
    assert first_tracks[0].bottom_center == (50.0, 100.0)


def test_inference_api_accepts_jpeg_and_returns_camera_scoped_detections():
    from concurrent.futures import Future

    class FakeInferenceService:
        request_timeout_seconds = 1

        def get_status(self):
            return {"worker_running": True}

        def submit(self, frame, camera_id):
            assert camera_id == "cam_7"
            assert frame.shape == (24, 32, 3)
            future = Future()
            future.set_result([RawDetection(2, "car", 0.91, (1, 2, 20, 22))])
            return future

    success, encoded = cv2.imencode(
        ".jpg", np.zeros((24, 32, 3), dtype=np.uint8)
    )
    assert success

    client = TestClient(create_inference_app(FakeInferenceService()))
    response = client.post(
        "/infer/cam_7",
        content=encoded.tobytes(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "camera_id": "cam_7",
        "detections": [
            {
                "class_id": 2,
                "class_name": "car",
                "conf": 0.91,
                "box": [1, 2, 20, 22],
            }
        ],
    }


def test_inference_api_rejects_invalid_frames_and_media_types():
    class FakeInferenceService:
        def get_status(self):
            return {"worker_running": True}

    client = TestClient(create_inference_app(FakeInferenceService()))

    invalid_frame = client.post(
        "/infer/cam_1",
        content=b"not-jpeg",
        headers={"Content-Type": "image/jpeg"},
    )
    invalid_type = client.post(
        "/infer/cam_1",
        content=b"frame",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert invalid_frame.status_code == 400
    assert invalid_type.status_code == 415


def test_inference_api_returns_rate_limit_when_shared_queue_is_full():
    class FullInferenceService:
        def submit(self, frame, camera_id):
            raise InferenceQueueFullError("Inference queue đang quá tải.")

    success, encoded = cv2.imencode(
        ".jpg", np.zeros((24, 32, 3), dtype=np.uint8)
    )
    assert success
    client = TestClient(create_inference_app(FullInferenceService()))

    response = client.post(
        "/infer/cam_1",
        content=encoded.tobytes(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Inference queue đang quá tải."
