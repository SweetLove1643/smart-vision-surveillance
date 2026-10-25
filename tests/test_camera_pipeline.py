from types import SimpleNamespace
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from main_app.controllers import camera_controller
from main_app.controllers.api_controller import create_api_app
from main_app.controllers.camera_controller import CameraController, SingleCameraPipeline
from main_app.services.detector_service import InferenceQueueFullError


def test_video_output_continues_while_inference_request_is_pending():
    pipeline = SingleCameraPipeline.__new__(SingleCameraPipeline)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    inference_started = threading.Event()
    finish_inference = threading.Event()
    streamed_frames = []
    stream_lock = threading.Lock()

    class FakeReader:
        fps = 25.0

        def read_latest_frame(self):
            return frame.copy()

        def stop(self):
            pass

    class FakeStreamer:
        def push_frame(self, image):
            with stream_lock:
                streamed_frames.append(image)

        def close(self):
            pass

    class FakeInferenceClient:
        def infer(self, image):
            inference_started.set()
            finish_inference.wait(timeout=2)
            return []

    class FakeTracker:
        def track(self, detections, frame_shape):
            return []

    pipeline.target_fps = 20
    pipeline.skip_frames = 0
    pipeline._stop_event = threading.Event()
    pipeline._lifecycle_lock = threading.Lock()
    pipeline._inference_wakeup = threading.Event()
    pipeline._inference_frame_lock = threading.Lock()
    pipeline._latest_inference_frame = None
    pipeline._detections_lock = threading.Lock()
    pipeline._last_detections = []
    pipeline._frame_count = 0
    pipeline._fps_count = 0
    pipeline._fps_time = time.perf_counter()
    pipeline.output_fps = 0.0
    pipeline.latency_ms = 0.0
    pipeline._metrics_started_at = time.perf_counter()
    pipeline._inference_count = 0
    pipeline._inference_latency_total_ms = 0.0
    pipeline.inference_latency_ms = 0.0
    pipeline.inference_fps = 0.0
    pipeline.is_running = True
    pipeline.model_config = SimpleNamespace(metrics_log_interval_seconds=30)
    pipeline.reader = FakeReader()
    pipeline.output_reader = FakeReader()
    pipeline.streamer = FakeStreamer()
    pipeline.inference_client = FakeInferenceClient()
    pipeline.tracker = FakeTracker()
    pipeline._draw = lambda image, detections: image
    pipeline._process_events = lambda frame, detections, active_ids: None

    pipeline._inference_thread = threading.Thread(target=pipeline._inference_loop)
    pipeline._thread = threading.Thread(target=pipeline._process_loop)
    pipeline._inference_thread.start()
    pipeline._thread.start()

    assert inference_started.wait(timeout=2)
    time.sleep(0.3)
    with stream_lock:
        frames_during_inference = len(streamed_frames)

    pipeline._stop_event.set()
    pipeline._inference_wakeup.set()
    finish_inference.set()
    pipeline._thread.join(timeout=2)
    pipeline._inference_thread.join(timeout=2)

    assert frames_during_inference >= 3
    assert not pipeline._thread.is_alive()
    assert not pipeline._inference_thread.is_alive()


def test_overloaded_inference_does_not_age_or_reset_camera_tracks(caplog):
    pipeline = SingleCameraPipeline.__new__(SingleCameraPipeline)
    pipeline.camera_id = "cam_1"
    pipeline._stop_event = threading.Event()
    pipeline._inference_wakeup = threading.Event()
    pipeline._inference_frame_lock = threading.Lock()
    pipeline._latest_inference_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    tracker_calls = []

    class FakeInferenceClient:
        def infer(self, image):
            raise InferenceQueueFullError("queue full")

    class FakeTracker:
        def track(self, detections, frame_shape):
            tracker_calls.append(detections)
            return []

    pipeline.inference_client = FakeInferenceClient()
    pipeline.tracker = FakeTracker()
    pipeline._inference_wakeup.set()
    worker = threading.Thread(target=pipeline._inference_loop)
    worker.start()

    deadline = time.monotonic() + 2
    while not caplog.records and time.monotonic() < deadline:
        time.sleep(0.01)
    pipeline._stop_event.set()
    pipeline._inference_wakeup.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert not tracker_calls
    assert "queue full" in caplog.text


def test_no_rtsp_pipeline_does_not_create_output_capture_or_publisher(monkeypatch):
    readers = []
    streamers = []

    class FakeReader:
        def __init__(self, camera_id, source, name):
            readers.append(source)

    class FakeStreamer:
        def __init__(self, rtsp_url, fps, enabled):
            streamers.append(enabled)

    monkeypatch.setattr(camera_controller, "CameraReader", FakeReader)
    monkeypatch.setattr(camera_controller, "FFmpegRTSPStreamer", FakeStreamer)
    camera = SimpleNamespace(
        id="cam_1",
        name="Camera 1",
        source="input.mp4",
        rtsp_out="rtsp://localhost:8554/live/cam_1",
        line_crossing=SimpleNamespace(enabled=False),
        restricted_zone=SimpleNamespace(enabled=False, polygon=[]),
    )

    pipeline = SingleCameraPipeline(
        camera,
        inference_client=object(),
        db=SimpleNamespace(get_counts=lambda: {}),
        model_config=SimpleNamespace(skip_frames=0, target_fps=12),
        enable_rtsp=False,
    )

    assert readers == ["input.mp4"]
    assert streamers == [False]
    assert pipeline.output_reader is None


def test_stop_all_signals_every_pipeline_before_joining_any():
    controller = CameraController.__new__(CameraController)
    controller._lifecycle_lock = threading.RLock()
    controller._stopping = False
    controller._inference_shutdown_timeout = 1

    class FakePipeline:
        def __init__(self):
            self.stop_requested = False
            self.stopped = False

        def request_stop(self):
            self.stop_requested = True

        def stop(self):
            assert all(pipeline.stop_requested for pipeline in controller.pipelines.values())
            self.stopped = True

    class FakeInferenceAPI:
        stopped = False

        def stop(self, timeout):
            self.stopped = True

    controller.pipelines = {"cam_1": FakePipeline(), "cam_2": FakePipeline()}
    controller.inference_api = FakeInferenceAPI()

    controller.stop_all()

    assert all(pipeline.stopped for pipeline in controller.pipelines.values())
    assert controller.inference_api.stopped


def test_start_all_can_limit_work_to_selected_cameras():
    controller = CameraController.__new__(CameraController)
    controller._lifecycle_lock = threading.RLock()

    class FakePipeline:
        def __init__(self):
            self.config = SimpleNamespace(enabled=True)
            self.started = False

        def start(self):
            self.started = True

    class FakeInferenceAPI:
        def start(self):
            pass

    controller.pipelines = {"cam_1": FakePipeline(), "cam_2": FakePipeline()}
    controller.inference_api = FakeInferenceAPI()

    controller.start_all(camera_ids=["cam_2"])

    assert not controller.pipelines["cam_1"].started
    assert controller.pipelines["cam_2"].started


def test_live_endpoint_reports_disabled_rtsp_in_cli_mode(tmp_path):
    controller = SimpleNamespace(
        enable_rtsp=False,
        config=SimpleNamespace(
            database=SimpleNamespace(snapshot_dir=str(tmp_path / "snapshots"))
        ),
        pipelines={},
    )
    client = TestClient(create_api_app(controller))

    response = client.get("/cameras/cam_1/live")

    assert response.status_code == 503
    assert "--no-rtsp" in response.json()["detail"]
