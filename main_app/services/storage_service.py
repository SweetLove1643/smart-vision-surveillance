from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional
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

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
