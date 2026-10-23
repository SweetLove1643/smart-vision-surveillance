from main_app.services.streamer_service import FFmpegRTSPStreamer
from main_app.services import streamer_service


def test_ffmpeg_uses_pipeline_rate_and_wall_clock_timestamps(monkeypatch):
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
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
    assert command[command.index("-framerate") + 1] == "12"
    assert command[command.index("-use_wallclock_as_timestamps") + 1] == "1"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
