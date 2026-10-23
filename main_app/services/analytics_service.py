from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


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
    return (cp1 * cp2 < 0) and (cp3 * cp4 < 0)


class LineCrossingCounter:
    """Đếm đối tượng cắt qua vạch hai chiều (F05)."""

    def __init__(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> None:
        self.p1 = (float(point1[0]), float(point1[1]))
        self.p2 = (float(point2[0]), float(point2[1]))
        self.count_in = 0
        self.count_out = 0
        self._previous_positions: Dict[int, Tuple[float, float]] = {}
        self._counted_tracks: Dict[int, str] = {}

    def update(self, track_id: int, current_pos: Tuple[float, float]) -> Optional[str]:
        prev_pos = self._previous_positions.get(track_id)
        self._previous_positions[track_id] = current_pos

        if prev_pos is None:
            return None

        if segments_intersect(self.p1, self.p2, prev_pos, current_pos):
            side = ccw(self.p1, self.p2, current_pos)
            direction = "in" if side > 0 else "out"

            if self._counted_tracks.get(track_id) != direction:
                self._counted_tracks[track_id] = direction
                if direction == "in":
                    self.count_in += 1
                else:
                    self.count_out += 1
                return direction

        return None

    def clean_old_tracks(self, active_track_ids: Set[int]) -> None:
        tracked_keys = list(self._previous_positions.keys())
        for tid in tracked_keys:
            if tid not in active_track_ids:
                self._previous_positions.pop(tid, None)
                self._counted_tracks.pop(tid, None)


class ZoneIntrusionDetector:
    """Phát hiện đối tượng xâm nhập vùng cấm đa giác (F06, F07)."""

    def __init__(self, polygon: List[Tuple[int, int]]) -> None:
        self.polygon = polygon
        self.np_polygon = np.array(polygon, dtype=np.int32) if len(polygon) >= 3 else None
        self._active_intruders: Set[int] = set()

    def is_inside(self, point: Tuple[float, float]) -> bool:
        if self.np_polygon is None or len(self.polygon) < 3:
            return False
        return cv2.pointPolygonTest(self.np_polygon, (float(point[0]), float(point[1])), False) >= 0

    def update(self, track_id: int, point: Tuple[float, float]) -> bool:
        inside = self.is_inside(point)
        if inside:
            if track_id not in self._active_intruders:
                self._active_intruders.add(track_id)
                return True
            return False
        else:
            self._active_intruders.discard(track_id)
            return False

    def clean_old_tracks(self, active_track_ids: Set[int]) -> None:
        self._active_intruders = self._active_intruders.intersection(active_track_ids)
