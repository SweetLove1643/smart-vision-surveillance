import logging
from io import BytesIO
import threading

import pytest

from main_app.services import capture_service
from main_app.services.streamer_service import FFmpegRTSPStreamer
from main_app.services import streamer_service


def test_ffmpeg_uses_pipeline_rate_and_wall_clock_timestamps(monkeypatch):
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 123
            self.stderr = BytesIO()
            commands.append(command)

    monkeypatch.setattr(streamer_service.subprocess, "Popen", FakeProcess)
    streamer = FFmpegRTSPStreamer(
        "rtsp://localhost:8554/test",
        fps=12,
        enabled=False,
    )

    process = streamer._start_process(1920, 1080)

    assert isinstance(process, FakeProcess)
    command = commands[0]
    assert command[command.index("-framerate") + 1] == "30"
    assert command[command.index("-use_wallclock_as_timestamps") + 1] == "1"
    assert command[command.index("-fps_mode") + 1] == "vfr"
    assert command[command.index("-g") + 1] == "12"
    assert command[command.index("-keyint_min") + 1] == "12"
    assert command[command.index("-loglevel") + 1] == "warning"


def test_rtsp_url_without_port_uses_standard_port():
    streamer = FFmpegRTSPStreamer("rtsp://camera/live", enabled=False)

    assert (streamer._host, streamer._port) == ("camera", 554)


def test_invalid_rtsp_url_fails_fast():
    with pytest.raises(ValueError, match="RTSP output URL"):
        FFmpegRTSPStreamer("http://localhost:8554/live", enabled=False)


def test_ffmpeg_stderr_is_logged_with_credentials_redacted(caplog):
    streamer = FFmpegRTSPStreamer(
        "rtsp://user:secret@localhost:8554/live",
        enabled=False,
    )
    process = type(
        "FakeProcess",
        (),
        {"stderr": BytesIO(b"failed rtsp://user:secret@localhost:8554/live\n")},
    )()

    with caplog.at_level(logging.WARNING, logger="services.streamer"):
        streamer._log_process_errors(process)

    assert "secret" not in caplog.text
    assert "rtsp://***@localhost" in caplog.text


def test_capture_logs_redact_rtsp_credentials(caplog, monkeypatch):
    class FailedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(
        capture_service.cv2, "VideoCapture", lambda *args: FailedCapture()
    )
    reader = capture_service.CameraReader(
        "cam_1", "rtsp://user:secret@camera.local/live"
    )

    with caplog.at_level(logging.WARNING, logger="services.capture"):
        reader._open_capture()

    assert "secret" not in caplog.text
    assert "rtsp://***@camera.local" in caplog.text


def test_rtsp_capture_uses_bounded_open_and_read_timeouts(monkeypatch):
    captures = []

    class OpenCapture:
        def isOpened(self):
            return True

    def fake_capture(*args):
        captures.append(args)
        return OpenCapture()

    monkeypatch.setattr(capture_service.cv2, "VideoCapture", fake_capture)
    reader = capture_service.CameraReader("cam_1", "rtsp://camera.local/live")

    assert reader._open_capture() is not None
    assert captures[0][:2] == (
        "rtsp://camera.local/live",
        capture_service.cv2.CAP_FFMPEG,
    )
    assert captures[0][2] == [
        capture_service.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        capture_service.RTSP_OPEN_TIMEOUT_MS,
        capture_service.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        capture_service.RTSP_READ_TIMEOUT_MS,
    ]


def test_capture_stop_is_safe_when_called_concurrently():
    reader = capture_service.CameraReader("cam_1", "unused.mp4")
    finish_capture = threading.Event()

    def capture_thread():
        finish_capture.wait(timeout=2)

    thread = threading.Thread(target=capture_thread, daemon=True)
    reader._thread = thread
    reader.state = capture_service.LIVE
    thread.start()
    errors = []
    barrier = threading.Barrier(3)

    def stop_reader():
        try:
            barrier.wait()
            reader.stop()
        except Exception as exc:
            errors.append(exc)

    callers = [threading.Thread(target=stop_reader) for _ in range(2)]
    for caller in callers:
        caller.start()
    barrier.wait()
    finish_capture.set()
    for caller in callers:
        caller.join(timeout=2)

    assert all(not caller.is_alive() for caller in callers)
    assert not errors
    assert reader._thread is None
    assert reader.state == capture_service.STOPPED
