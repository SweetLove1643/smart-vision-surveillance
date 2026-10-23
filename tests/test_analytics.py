import pytest
import main_app.services.analytics_service as analytics_service
from main_app.services.analytics_service import (
    LineCrossingCounter,
    ZoneIntrusionDetector,
    segments_intersect,
)


def test_segments_intersect():
    # Hai đoạn thẳng cắt nhau hình chữ thập
    p1, p2 = (0.0, 5.0), (10.0, 5.0)
    q1, q2 = (5.0, 0.0), (5.0, 10.0)
    assert segments_intersect(p1, p2, q1, q2) is True

    # Hai đoạn thẳng song song
    q3, q4 = (0.0, 2.0), (10.0, 2.0)
    assert segments_intersect(p1, p2, q3, q4) is False

    # Không cắt nhau
    q5, q6 = (15.0, 0.0), (15.0, 10.0)
    assert segments_intersect(p1, p2, q5, q6) is False

    # Chạm đầu mút và cắt tại đầu cuối của vạch vẫn được tính là giao nhau.
    assert segments_intersect((0.0, 5.0), (10.0, 5.0), (10.0, 5.0), (10.0, 9.0))


def test_line_crossing_counter():
    counter = LineCrossingCounter((0, 100), (200, 100))

    # Track 1 đi từ trên xuống dưới (y: 50 -> 150)
    res1 = counter.update(track_id=1, current_pos=(100.0, 50.0))
    assert res1 is None
    assert counter.count_in == 0
    assert counter.count_out == 0

    res2 = counter.update(track_id=1, current_pos=(100.0, 150.0))
    assert res2 in ["in", "out"]
    assert counter.count_in + counter.count_out == 1

    # Cùng track 1 di chuyển tiếp ở dưới, không đếm trùng
    res3 = counter.update(track_id=1, current_pos=(100.0, 180.0))
    assert res3 is None
    assert counter.count_in + counter.count_out == 1

    # Track 2 đi từ dưới lên trên (chiều ngược lại)
    counter.update(track_id=2, current_pos=(120.0, 160.0))
    res4 = counter.update(track_id=2, current_pos=(120.0, 40.0))
    assert res4 in ["in", "out"]
    assert res4 != res2
    assert counter.count_in == 1
    assert counter.count_out == 1


def test_line_counter_counts_crossing_after_short_track_miss(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(analytics_service.time, "monotonic", lambda: clock[0])
    counter = LineCrossingCounter(
        (0, 10), (10, 10), max_missing_seconds=2.0
    )
    assert counter.update(1, (5.0, 0.0)) is None
    clock[0] = 1.0
    counter.clean_old_tracks(set())
    clock[0] = 1.8

    assert counter.update(1, (5.0, 20.0)) == "in"
    assert counter.count_in == 1


def test_line_counter_expires_stale_track_by_elapsed_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(analytics_service.time, "monotonic", lambda: clock[0])
    counter = LineCrossingCounter(
        (0, 10), (10, 10), max_missing_seconds=2.0
    )
    counter.update(1, (5.0, 0.0))

    clock[0] = 2.1
    counter.clean_old_tracks(set())
    clock[0] = 2.2

    assert counter.update(1, (5.0, 20.0)) is None
    assert counter.count_in == 0


def test_line_counter_counts_when_track_lands_on_line_first():
    counter = LineCrossingCounter((0, 10), (10, 10))
    counter.update(1, (5.0, 0.0))

    assert counter.update(1, (5.0, 10.0)) is None
    assert counter.update(1, (5.0, 20.0)) == "in"


def test_zone_intrusion_detector():
    polygon = [(10, 10), (100, 10), (100, 100), (10, 100)]
    detector = ZoneIntrusionDetector(polygon)

    # 1. Điểm ở ngoài vùng cấm
    assert detector.is_inside((5.0, 5.0)) is False
    assert detector.update(track_id=1, point=(5.0, 5.0)) is False

    # 2. Bước vào trong vùng cấm lần đầu -> Kích hoạt sự kiện (True)
    assert detector.is_inside((50.0, 50.0)) is True
    assert detector.update(track_id=1, point=(50.0, 50.0)) is True

    # 3. Vẫn ở trong vùng cấm ở frame tiếp theo -> Không được spam sự kiện (False) (F07)
    assert detector.update(track_id=1, point=(60.0, 60.0)) is False

    # 4. Bước ra ngoài vùng cấm -> Không kích hoạt sự kiện
    assert detector.update(track_id=1, point=(150.0, 150.0)) is False

    # 5. Bước vào lại -> Được tính là lần vi phạm mới (True)
    assert detector.update(track_id=1, point=(50.0, 50.0)) is True


def test_zone_intrusion_does_not_repeat_after_short_track_miss(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(analytics_service.time, "monotonic", lambda: clock[0])
    detector = ZoneIntrusionDetector(
        [(10, 10), (100, 10), (100, 100), (10, 100)],
        max_missing_seconds=2.0,
    )
    assert detector.update(1, (50.0, 50.0)) is True
    clock[0] = 1.0
    detector.clean_old_tracks(set())
    clock[0] = 1.8
    assert detector.update(1, (60.0, 60.0)) is False
