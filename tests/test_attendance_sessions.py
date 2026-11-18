"""
Regression tests for attendance-session selection in AttendanceDatabase.log_attendance.

Blocked audit rows (SPOOF_BLOCKED) are stored with time_in='--' and time_out='--'.
They must never be treated as an open TIME_IN session that a later genuine scan
can close with a TIME_OUT.

All tests use a temporary SQLite database and synthetic employee IDs.
"""

import gc
import tempfile
import unittest
from pathlib import Path

from src.database.db_manager import AttendanceDatabase

EMP = "SYN-0001"


class TestBlockedRowsAreNotOpenSessions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = AttendanceDatabase(db_path=Path(self.temp_dir.name) / "sessions.db")

    def tearDown(self):
        del self.db
        gc.collect()
        self.temp_dir.cleanup()

    def _rows(self):
        conn = self.db._get_connection()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT id, employee_id, time_in, time_out, spoof_status FROM logs ORDER BY id"
                ).fetchall()
            ]
        finally:
            conn.close()

    def _blocked(self):
        return self.db.log_attendance(EMP, spoof_detected=True, liveness_score=0.10, similarity_score=0.90)

    def _genuine(self):
        return self.db.log_attendance(EMP, spoof_detected=False, liveness_score=0.95, similarity_score=0.90)

    def assert_blocked_row_untouched(self, row):
        self.assertEqual(row["time_in"], "--")
        self.assertEqual(row["time_out"], "--")
        self.assertTrue(row["spoof_status"].startswith("Failed"))

    def test_blocked_then_genuine_creates_time_in(self):
        self.assertEqual(self._blocked(), "SPOOF_BLOCKED")
        self.assertEqual(self._genuine(), "TIME_IN")

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assert_blocked_row_untouched(rows[0])
        self.assertEqual(rows[1]["spoof_status"], "Passed")
        self.assertNotEqual(rows[1]["time_in"], "--")
        self.assertEqual(rows[1]["time_out"], "--")

    def test_blocked_between_in_and_out_closes_the_genuine_session(self):
        self.assertEqual(self._genuine(), "TIME_IN")
        self.assertEqual(self._blocked(), "SPOOF_BLOCKED")
        self.assertEqual(self._genuine(), "TIME_OUT")

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        session, blocked = rows
        self.assertEqual(session["spoof_status"], "Passed")
        self.assertNotEqual(session["time_in"], "--")
        self.assertNotEqual(session["time_out"], "--")
        self.assert_blocked_row_untouched(blocked)

    def test_multiple_blocked_rows_never_receive_time_out(self):
        for _ in range(3):
            self._blocked()
        self.assertEqual(self._genuine(), "TIME_IN")
        self.assertEqual(self._genuine(), "TIME_OUT")

        rows = self._rows()
        self.assertEqual(len(rows), 4)
        for row in rows[:3]:
            self.assert_blocked_row_untouched(row)
        self.assertNotEqual(rows[3]["time_out"], "--")

    def test_normal_in_out_cycle_is_preserved(self):
        self.assertEqual(self._genuine(), "TIME_IN")
        self.assertEqual(self._genuine(), "TIME_OUT")
        # After a closed session, the next genuine scan opens a new session.
        self.assertEqual(self._genuine(), "TIME_IN")

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["time_out"], "--")
        self.assertEqual(rows[1]["time_out"], "--")
        self.assertTrue(all(r["spoof_status"] == "Passed" for r in rows))

    def test_other_employees_blocked_rows_are_ignored(self):
        self.db.log_attendance("SYN-0002", spoof_detected=True, liveness_score=0.10)
        self.assertEqual(self._genuine(), "TIME_IN")
        self.assert_blocked_row_untouched(self._rows()[0])


if __name__ == "__main__":
    unittest.main()
