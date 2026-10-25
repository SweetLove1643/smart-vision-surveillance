from main_app.services.config_service import load_config


def test_default_camera_regions_are_configured_independently():
    config = load_config("config.yaml")

    assert config.model.inference_api_host == "127.0.0.1"
    assert config.model.inference_api_port == 8001
    assert config.model.inference_timeout_seconds == 120
    assert config.model.skip_frames == 0
    assert config.model.metrics_log_interval_seconds >= 1
    lines = {
        (camera.line_crossing.point1, camera.line_crossing.point2)
        for camera in config.cameras
    }
    zones = {
        tuple(camera.restricted_zone.polygon)
        for camera in config.cameras
    }

    assert len(config.cameras) == len(lines) == len(zones)
    assert all(len(camera.restricted_zone.polygon) >= 3 for camera in config.cameras)
