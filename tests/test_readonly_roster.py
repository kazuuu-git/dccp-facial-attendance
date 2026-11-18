"""
P5 tests for src/database/readonly_roster.py using real temporary SQLite files.

Verifies that the roster never creates or modifies database files, folders or
tables, refuses WAL databases, distinguishes "unavailable" from "zero
employees", retains a cached roster only for transient read errors, and never
opens a writable connection.
"""

import gc
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.database import readonly_roster
from src.database.db_manager import AttendanceDatabase
from src.database.readonly_roster import ReadOnlyEmployeeRoster


def synthetic_embedding(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def snapshot(directory: Path):
    """Names, sizes, mtimes and bytes of every file in a folder."""
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


class RosterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def make_db(self, n_employees):
        path = self.root / "roster.db"
        db = AttendanceDatabase(db_path=path)
        for i in range(n_employees):
            db.enroll_employee(f"SYN-{i:04d}", f"Synthetic {i}", synthetic_embedding(i))
        del db
        gc.collect()
        return path


class TestNoWrites(RosterTestCase):
    def test_missing_database_and_folder_are_not_created(self):
        path = self.root / "missing-folder" / "attendance.db"
        roster = ReadOnlyEmployeeRoster(db_path=path)
        self.assertEqual(roster.state, readonly_roster.UNAVAILABLE)
        self.assertIn("not found", roster.reason)
        self.assertEqual(roster.employee_count, 0)
        self.assertFalse(path.exists())
        self.assertFalse(path.parent.exists())

    def test_existing_database_is_not_modified(self):
        path = self.make_db(2)
        before = snapshot(self.root)
        roster = ReadOnlyEmployeeRoster(db_path=path)
        roster.refresh()
        roster.load_enrolled_embeddings()
        self.assertEqual(roster.state, readonly_roster.AVAILABLE)
        self.assertEqual(roster.employee_count, 2)
        self.assertEqual(snapshot(self.root), before, "no file created or changed")

    def test_database_without_tables_is_unavailable_and_untouched(self):
        path = self.root / "other.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE something_else (x INTEGER)")
        conn.commit()
        conn.close()
        before = snapshot(self.root)

        roster = ReadOnlyEmployeeRoster(db_path=path)
        self.assertEqual(roster.state, readonly_roster.UNAVAILABLE)
        self.assertIn("no employees table", roster.reason)
        self.assertEqual(snapshot(self.root), before)
        conn = sqlite3.connect(path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        conn.close()
        self.assertEqual(tables, ["something_else"])

    def test_empty_and_non_sqlite_files_are_unavailable(self):
        empty = self.root / "empty.db"
        empty.write_bytes(b"")
        text = self.root / "text.db"
        text.write_text("not a database")
        before = snapshot(self.root)
        self.assertEqual(ReadOnlyEmployeeRoster(db_path=empty).state, readonly_roster.UNAVAILABLE)
        self.assertEqual(ReadOnlyEmployeeRoster(db_path=text).state, readonly_roster.UNAVAILABLE)
        self.assertEqual(snapshot(self.root), before)

    def test_wal_database_is_refused_without_creating_side_files(self):
        path = self.make_db(1)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        gc.collect()
        before = snapshot(self.root)

        roster = ReadOnlyEmployeeRoster(db_path=path)
        self.assertEqual(roster.state, readonly_roster.UNAVAILABLE)
        self.assertIn("WAL", roster.reason)
        self.assertEqual(snapshot(self.root), before)

    def test_connection_is_read_only_and_never_writable(self):
        path = self.make_db(1)
        calls = []
        real_connect = sqlite3.connect

        def recording_connect(*args, **kwargs):
            calls.append((args, kwargs))
            return real_connect(*args, **kwargs)

        with mock.patch.object(readonly_roster.sqlite3, "connect", side_effect=recording_connect):
            roster = ReadOnlyEmployeeRoster(db_path=path)
            roster.refresh()
        self.assertTrue(calls)
        for args, kwargs in calls:
            self.assertTrue(kwargs.get("uri"))
            self.assertTrue(args[0].endswith("?mode=ro"))
            self.assertNotIn("immutable", args[0])

        conn = roster._connect_readonly()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM employees")
        finally:
            conn.close()

    def test_roster_has_no_write_methods(self):
        for name in ("log_attendance", "enroll_employee", "reset_database"):
            self.assertFalse(hasattr(ReadOnlyEmployeeRoster, name))


class TestRosterStates(RosterTestCase):
    def test_zero_employees_is_available_not_unavailable(self):
        roster = ReadOnlyEmployeeRoster(db_path=self.make_db(0))
        self.assertEqual(roster.state, readonly_roster.AVAILABLE)
        self.assertEqual(roster.employee_count, 0)

    def test_transient_error_keeps_cached_roster(self):
        roster = ReadOnlyEmployeeRoster(db_path=self.make_db(2))
        with mock.patch.object(readonly_roster.sqlite3, "connect",
                               side_effect=sqlite3.OperationalError("database is locked")):
            roster.refresh()
        self.assertEqual(roster.state, readonly_roster.CACHED)
        self.assertIn("locked", roster.reason)
        self.assertEqual(roster.employee_count, 2)

        roster.refresh()  # recovers
        self.assertEqual(roster.state, readonly_roster.AVAILABLE)

    def test_transient_error_without_previous_roster_is_unavailable(self):
        path = self.make_db(2)
        with mock.patch.object(readonly_roster.sqlite3, "connect",
                               side_effect=sqlite3.OperationalError("database is locked")):
            roster = ReadOnlyEmployeeRoster(db_path=path)
        self.assertEqual(roster.state, readonly_roster.UNAVAILABLE)
        self.assertEqual(roster.employee_count, 0)

    def test_successful_empty_refresh_clears_old_identities(self):
        path = self.make_db(2)
        roster = ReadOnlyEmployeeRoster(db_path=path)
        self.assertEqual(roster.employee_count, 2)
        conn = sqlite3.connect(path)  # test setup: another writer removes employees
        conn.execute("DELETE FROM employees")
        conn.commit()
        conn.close()
        roster.refresh()
        self.assertEqual(roster.state, readonly_roster.AVAILABLE)
        self.assertEqual(roster.employee_count, 0)

    def test_deleted_database_clears_roster(self):
        path = self.make_db(1)
        roster = ReadOnlyEmployeeRoster(db_path=path)
        gc.collect()
        os.remove(path)
        roster.refresh()
        self.assertEqual(roster.state, readonly_roster.UNAVAILABLE)
        self.assertEqual(roster.employee_count, 0)

    def test_env_path_is_used(self):
        path = self.make_db(1)
        with mock.patch.dict(os.environ, {"DCCP_DB_PATH": str(path)}):
            roster = ReadOnlyEmployeeRoster()
        self.assertEqual(roster.db_path, path)
        self.assertEqual(roster.employee_count, 1)


if __name__ == "__main__":
    unittest.main()
