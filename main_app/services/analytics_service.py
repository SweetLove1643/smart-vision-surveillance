from typing import Dict, List, Optional, Set, Tuple
import cv2
import math
import numpy as np
import time


def _point_on_segment(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    epsilon: float = 1e-9,
) -> bool:
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
        and abs(ccw(start, end, point)) <= epsilon
    )


def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Tích có hướng 2D xác định hướng quay của 3 điểm."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """Kiểm tra hai đoạn thẳng p1-p2 và q1-q2 có cắt nhau không."""
    cp1 = ccw(p1, p2, q1)
    cp2 = ccw(p1, p2, q2)
    cp3 = ccw(q1, q2, p1)
    cp4 = ccw(q1, q2, p2)
    if cp1 * cp2 < 0 and cp3 * cp4 < 0:
        return True
    return (
        (abs(cp1) <= 1e-9 and _point_on_segment(q1, p1, p2))
        or (abs(cp2) <= 1e-9 and _point_on_segment(q2, p1, p2))
        or (abs(cp3) <= 1e-9 and _point_on_segment(p1, q1, q2))
        or (abs(cp4) <= 1e-9 and _point_on_segment(p2, q1, q2))
    )


class LineCrossingCounter:
    """Đếm đối tượng cắt qua vạch hai chiều (F05)."""

    def __init__(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        count_in: int = 0,
        count_out: int = 0,
        max_missing_seconds: float = 2.0,
        hysteresis_px: float = 5.0,
    ) -> None:
        self.p1 = (float(point1[0]), float(point1[1]))
        self.p2 = (float(point2[0]), float(point2[1]))
        if self.p1 == self.p2:
            raise ValueError("Hai đầu mút của vạch đếm không được trùng nhau.")
        self.count_in = count_in
        self.count_out = count_out
        self.max_missing_seconds = max_missing_seconds
        self.hysteresis_px = max(0.0, hysteresis_px)
        self._line_length = math.hypot(
            self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        )
        self._previous_positions: Dict[int, Tuple[float, float]] = {}
        self._last_sides: Dict[int, int] = {}
        self._last_seen_at: Dict[int, float] = {}

    def update(self, track_id: int, current_pos: Tuple[float, float]) -> Optional[str]:
        self._last_seen_at[track_id] = time.monotonic()
        signed_distance = ccw(self.p1, self.p2, current_pos) / self._line_length
        if abs(signed_distance) <= self.hysteresis_px:
            return None

        prev_pos = self._previous_positions.get(track_id)
        current_side = 1 if signed_distance > 0 else -1
        previous_side = self._last_sides.get(track_id)

        if (
            previous_side is not None
            and previous_side != current_side
            and segments_intersect(self.p1, self.p2, prev_pos, current_pos)
        ):
            direction = "in" if current_side > 0 else "out"
            if direction == "in":
                self.count_in += 1
            else:
                self.count_out += 1
        else:
            direction = None

        self._previous_positions[track_id] = current_pos
        self._last_sides[track_id] = current_side
        return direction

    def clean_old_tracks(self, active_track_ids: Set[int]) -> None:
        self._expire_missing_tracks(active_track_ids)

    def reset_tracks(self) -> None:
        self._previous_positions.clear()
        self._last_sides.clear()
        self._last_seen_at.clear()

    def _expire_missing_tracks(self, active_track_ids: Set[int]) -> None:
        now = time.monotonic()
        for track_id in list(self._previous_positions):
            if track_id in active_track_ids:
                self._last_seen_at[track_id] = now
                continue

            if now - self._last_seen_at[track_id] > self.max_missing_seconds:
                self._previous_positions.pop(track_id, None)
                self._last_sides.pop(track_id, None)
                self._last_seen_at.pop(track_id, None)


class ZoneIntrusionDetector:
    """Phát hiện đối tượng xâm nhập vùng cấm đa giác (F06, F07)."""

    def __init__(
        self,
        polygon: List[Tuple[int, int]],
        max_missing_seconds: float = 2.0,
        hysteresis_px: float = 5.0,
    ) -> None:
        self.polygon = polygon
        self.np_polygon = np.array(polygon, dtype=np.int32) if len(polygon) >= 3 else None
        self._active_intruders: Set[int] = set()
        self.max_missing_seconds = max_missing_seconds
        self.hysteresis_px = max(0.0, hysteresis_px)
        self._last_seen_at: Dict[int, float] = {}

    def is_inside(self, point: Tuple[float, float]) -> bool:
        if self.np_polygon is None or len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.np_polygon, (float(point[0]), float(point[1])), False) >= 0

    def update(self, track_id: int, point: Tuple[float, float]) -> bool:
        self._last_seen_at[track_id] = time.monotonic()
        if self.np_polygon is None:
            return False
        signed_distance = cv2.pointPolygonTest(
            self.np_polygon, (float(point[0]), float(point[1])), True
        )
        if track_id in self._active_intruders:
            inside = signed_distance >= -self.hysteresis_px
        else:
            inside = signed_distance >= self.hysteresis_px
        if inside:
            if track_id not in self._active_intruders:
                self._active_intruders.add(track_id)
                return True
            return False
        else:
            self._active_intruders.discard(track_id)
            return False

    def clean_old_tracks(self, active_track_ids: Set[int]) -> None:
        now = time.monotonic()
        for track_id in list(self._last_seen_at):
            if track_id in active_track_ids:
                self._last_seen_at[track_id] = now
                continue

            if now - self._last_seen_at[track_id] > self.max_missing_seconds:
                self._active_intruders.discard(track_id)
                self._last_seen_at.pop(track_id, None)

    def reset_tracks(self) -> None:
        self._active_intruders.clear()
        self._last_seen_at.clear()
