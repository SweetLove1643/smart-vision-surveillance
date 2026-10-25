from concurrent.futures import Future
import socket

import cv2
import numpy as np

from main_app.services.detector_service import RawDetection
from main_app.services.inference_api import YoloInferenceAPIServer
from main_app.services.inference_client import YoloInferenceClient


class FakeInferenceService:
    def __init__(self):
        self.worker_running = False
        self.inferred_camera_ids = []
        self.request_timeout_seconds = 1

    def start(self):
        self.worker_running = True

    def stop(self, timeout=15.0):
        self.worker_running = False

    def get_status(self):
        return {"worker_running": self.worker_running}

    def infer(self, frame, camera_id):
        return self._infer(frame, camera_id)

    def submit(self, frame, camera_id):
        future = Future()
        future.set_result(self._infer(frame, camera_id))
        return future

    def _infer(self, frame, camera_id):
        self.inferred_camera_ids.append(camera_id)
        assert frame.shape == (24, 32, 3)
        return [RawDetection(0, "person", 0.88, (1, 2, 20, 22))]


def test_camera_client_calls_dedicated_http_inference_api():
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]

    inference_service = FakeInferenceService()
    api_server = YoloInferenceAPIServer(
        inference_service, "127.0.0.1", port
    )
    try:
        success, frame = cv2.imencode(
            ".jpg", np.zeros((24, 32, 3), dtype=np.uint8)
        )
        assert success
        client = YoloInferenceClient(
            f"http://127.0.0.1:{port}", "cam_1", timeout_seconds=5
        )
        for _ in range(2):
            api_server.start()
            detections = client.infer(cv2.imdecode(frame, cv2.IMREAD_COLOR))

            assert detections == [
                RawDetection(0, "person", 0.88, (1, 2, 20, 22))
            ]
            assert api_server.is_running
            api_server.stop()
    finally:
        if api_server.is_running or inference_service.worker_running:
            api_server.stop()

    assert not api_server.is_running
    assert not inference_service.worker_running
    assert inference_service.inferred_camera_ids == ["cam_1", "cam_1"]
