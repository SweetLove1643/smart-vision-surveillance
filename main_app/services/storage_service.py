from datetime import datetime
from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("services.storage")


class EventDatabase:
    """Quản lý lưu sự kiện vi phạm vào SQLite & lưu ảnh snapshot trong resources/database (F07)."""

    def __init__(
        self,
        db_path: str = "resources/database/events.db",
        snapshot_dir: str = "resources/database/snapshots",
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshot_dir = Path(snapshot_dir)
        self._lock = threading.Lock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    snapshot_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_counts (
                    camera_id TEXT PRIMARY KEY,
                    count_in INTEGER NOT NULL DEFAULT 0,
                    count_out INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO camera_counts (camera_id, count_in, count_out)
                SELECT camera_id,
                       SUM(CASE WHEN details LIKE '%Direction: in%' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN details LIKE '%Direction: out%' THEN 1 ELSE 0 END)
                FROM events
                WHERE event_type = 'line_crossing'
                GROUP BY camera_id
                """
            )
            conn.commit()
            logger.info("Khởi tạo SQLite database tại: %s", self.db_path)

    def save_event(
        self,
        camera_id: str,
        track_id: int,
        event_type: str,
        details: str = "",
        frame: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_rel_path = ""

        if frame is not None:
            filename = f"{camera_id}_t{track_id}_{now_str}.jpg"
            snapshot_full_path = self.snapshot_dir / filename
            cv2.imwrite(str(snapshot_full_path), frame)
            snapshot_rel_path = str(snapshot_full_path)

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (camera_id, track_id, event_type, details, snapshot_path, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
                """,
                (camera_id, track_id, event_type, details, snapshot_rel_path),
            )
            conn.commit()
            event_id = cursor.lastrowid

        logger.info(
            "[%s] Ghi sự kiện #%s: %s (Track: %s, Ảnh: %s)",
            camera_id,
            event_id,
            event_type,
            track_id,
            snapshot_rel_path,
        )
        return event_id

    def save_line_crossing(
        self,
        camera_id: str,
        track_id: int,
        direction: str,
        details: str = "",
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int]:
        if direction not in {"in", "out"}:
            raise ValueError(f"Hướng đếm không hợp lệ: {direction}")

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_rel_path = ""
        if frame is not None:
            filename = f"{camera_id}_t{track_id}_{now_str}.jpg"
            snapshot_full_path = self.snapshot_dir / filename
            cv2.imwrite(str(snapshot_full_path), frame)
            snapshot_rel_path = str(snapshot_full_path)

        increment_in = int(direction == "in")
        increment_out = int(direction == "out")
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (camera_id, track_id, event_type, details, snapshot_path, created_at)
                VALUES (?, ?, 'line_crossing', ?, ?, datetime('now', 'localtime'))
                """,
                (camera_id, track_id, details, snapshot_rel_path),
            )
            conn.execute(
                """
                INSERT INTO camera_counts (camera_id, count_in, count_out)
                VALUES (?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    count_in = count_in + excluded.count_in,
                    count_out = count_out + excluded.count_out,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (camera_id, increment_in, increment_out),
            )
            counts = conn.execute(
                "SELECT count_in, count_out FROM camera_counts WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
            event_id = int(cursor.lastrowid)

        logger.info(
            "[%s] Ghi sự kiện #%s: line_crossing (%s, Track: %s, Count: %s/%s)",
            camera_id,
            event_id,
            direction,
            track_id,
            counts["count_in"],
            counts["count_out"],
        )
        return event_id, int(counts["count_in"]), int(counts["count_out"])

    def get_counts(self) -> Dict[str, Dict[str, int]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT camera_id, count_in, count_out FROM camera_counts"
            ).fetchall()
            return {
                row["camera_id"]: {
                    "count_in": int(row["count_in"]),
                    "count_out": int(row["count_out"]),
                }
                for row in rows
            }

    def ensure_camera(self, camera_id: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO camera_counts (camera_id) VALUES (?)",
                (camera_id,),
            )

    def get_events(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if start_time:
            query += " AND created_at >= ?"
            params.append(start_time)
        if end_time:
            query += " AND created_at <= ?"
            params.append(end_time)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
