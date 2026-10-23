import sqlite3

from main_app.services.storage_service import EventDatabase


def test_line_crossing_counts_are_persisted_with_events(tmp_path):
    database_path = tmp_path / "events.db"
    snapshot_dir = tmp_path / "snapshots"

    database = EventDatabase(str(database_path), str(snapshot_dir))
    database.ensure_camera("cam_1")
    assert database.get_counts() == {
        "cam_1": {"count_in": 0, "count_out": 0}
    }

    event_id, count_in, count_out = database.save_line_crossing(
        camera_id="cam_1",
        track_id=7,
        direction="in",
        details="Direction: in, Class: car",
    )
    assert event_id > 0
    assert (count_in, count_out) == (1, 0)

    _, count_in, count_out = database.save_line_crossing(
        camera_id="cam_1",
        track_id=8,
        direction="out",
        details="Direction: out, Class: person",
    )
    assert (count_in, count_out) == (1, 1)

    reopened_database = EventDatabase(str(database_path), str(snapshot_dir))
    assert reopened_database.get_counts() == {
        "cam_1": {"count_in": 1, "count_out": 1}
    }
    assert len(reopened_database.get_events(camera_id="cam_1")) == 2


def test_existing_line_events_are_migrated_into_persisted_counts(tmp_path):
    database_path = tmp_path / "events.db"
    snapshot_dir = tmp_path / "snapshots"
    EventDatabase(str(database_path), str(snapshot_dir))

    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO events (camera_id, track_id, event_type, details)
            VALUES (?, ?, 'line_crossing', ?)
            """,
            [
                ("cam_2", 1, "Direction: in, Class: car"),
                ("cam_2", 2, "Direction: in, Class: person"),
                ("cam_2", 3, "Direction: out, Class: truck"),
            ],
        )

    database = EventDatabase(str(database_path), str(snapshot_dir))

    assert database.get_counts() == {
        "cam_2": {"count_in": 2, "count_out": 1}
    }
