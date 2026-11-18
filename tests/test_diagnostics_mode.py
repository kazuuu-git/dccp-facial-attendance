"""
P5 scanner tests: diagnostics mode writes nothing, normal mode keeps its
attendance/enrollment semantics, and the liveness overlay never shows stale or
invented results.

Camera, OpenCV window calls, recognition and liveness are test doubles.
Diagnostics tests that touch SQLite use real temporary files through the
read-only roster.
"""

import gc
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import app_scanner
from src.database import readonly_roster
from src.database.db_manager import AttendanceDatabase

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FACE = [100, 100, 300, 300]
EMB = np.ones(512, dtype=np.float32) / np.sqrt(512)
PASS = {"status": "PASS", "passed": True, "live_score": 0.91, "average_scores": [0.05, 0.91, 0.04],
        "threshold": 0.5, "reason": "ok"}
REJECT = {"status": "REJECT", "passed": False, "live_score": 0.31, "average_scores": [0.6, 0.31, 0.09],
          "threshold": 0.5, "reason": "below"}
INVALID = {"status": "INVALID", "passed": False, "live_score": None, "average_scores": None,
           "threshold": 0.5, "reason": "crop failed"}


class FakeCapture:
    def __init__(self, frames=50):
        self.remaining = frames
        self.released = False

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def getBackendName(self):
        return "FAKE"

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, FRAME.copy()

    def release(self):
        self.released = True


class FakeDatabase:
    def __init__(self):
        self.db_path = Path("nonexistent-test-only.db")
        self.enroll_employee = mock.Mock(return_value=True)
        self.log_attendance = mock.Mock(return_value="TIME_IN")

    def load_enrolled_embeddings(self):
        return {"SYN-0001": EMB}


class FakeRoster:
    """Roster double with no write methods, like the real one."""

    def __init__(self, state=readonly_roster.AVAILABLE, count=1, reason=None):
        self.db_path = Path("nonexistent-roster.db")
        self.state = state
        self.reason = reason
        self._emb = {f"SYN-{i:04d}": EMB for i in range(count)}

    @property
    def employee_count(self):
        return len(self._emb)

    @property
    def embeddings(self):
        return dict(self._emb)

    def refresh(self):
        pass


class Harness:
    def __init__(self, keys, decisions=(PASS,), recognize=("SYN-0001", 0.82, FACE),
                 enroll_liveness=(True, 0.9)):
        self.capture = FakeCapture()
        self.db = FakeDatabase()
        self.liveness = mock.Mock()
        self.liveness.evaluate.side_effect = list(decisions) + [decisions[-1]] * 50
        self.liveness.check_passive_liveness.return_value = enroll_liveness
        self.recognition = mock.Mock()
        if callable(recognize):
            self.recognition.recognize_employee.side_effect = recognize
        else:
            self.recognition.recognize_employee.return_value = recognize
        self.recognition.extract_embedding.return_value = (EMB, FACE)
        self.keys = list(keys)
        self.texts_per_frame = []
        self._current = []
        self.titles = []

    def _wait_key(self, _delay):
        self.texts_per_frame.append(self._current)
        self._current = []
        return self.keys.pop(0) if self.keys else ord("q")

    def _put_text(self, img, text, *args, **kwargs):
        self._current.append(text)
        return img

    def _imshow(self, title, _img):
        self.titles.append(title)

    def run(self, diagnostics, roster=None, attendance_db_cls=None, **kwargs):
        kwargs.setdefault("process_interval", 0.0)
        kwargs.setdefault("read_retry_delay", 0.0)
        db_cls = attendance_db_cls or mock.Mock(return_value=self.db)
        patches = [
            mock.patch.object(app_scanner, "AttendanceDatabase", db_cls),
            mock.patch.object(app_scanner, "PassiveLivenessEngine", return_value=self.liveness),
            mock.patch.object(app_scanner, "FaceRecognitionEngine", return_value=self.recognition),
            mock.patch.object(app_scanner.cv2, "VideoCapture", return_value=self.capture),
            mock.patch.object(app_scanner.cv2, "imshow", side_effect=self._imshow),
            mock.patch.object(app_scanner.cv2, "putText", side_effect=self._put_text),
            mock.patch.object(app_scanner.cv2, "destroyAllWindows"),
            mock.patch.object(app_scanner.cv2, "waitKey", side_effect=self._wait_key),
        ]
        if roster is not None:
            patches.append(mock.patch.object(app_scanner, "ReadOnlyEmployeeRoster", return_value=roster))
        with redirect_stdout(io.StringIO()):
            for p in patches:
                p.start()
            try:
                app_scanner.run_attendance_scanner(diagnostics=diagnostics, **kwargs)
            finally:
                for p in reversed(patches):
                    p.stop()
        return db_cls

    def all_texts(self):
        return [t for frame in self.texts_per_frame for t in frame]


def must_not_construct(*args, **kwargs):
    raise AssertionError("AttendanceDatabase must not be created in diagnostics mode")


class TestDiagnosticsWritesNothing(unittest.TestCase):
    def test_no_database_object_no_enrollment(self):
        h = Harness(keys=[-1, ord("e"), ord("e"), ord("q")])
        db_cls = mock.Mock(side_effect=must_not_construct)
        h.run(diagnostics=True, roster=FakeRoster(), attendance_db_cls=db_cls)

        db_cls.assert_not_called()
        h.recognition.extract_embedding.assert_not_called()
        h.liveness.check_passive_liveness.assert_not_called()
        texts = h.all_texts()
        self.assertIn(app_scanner.DIAG_HEADER, texts)
        self.assertIn(app_scanner.DIAG_FOOTER, texts)
        self.assertIn(app_scanner.DIAG_ENROLL_DISABLED, texts)
        self.assertIn("Identity: SYN-0001 (similarity 0.82)", texts)
        self.assertIn("Liveness: PASS | model live score 0.91 | threshold 0.50", texts)
        self.assertTrue(all(t == app_scanner.DIAG_WINDOW_TITLE for t in h.titles))

    def test_header_is_on_every_frame(self):
        h = Harness(keys=[-1, -1, ord("q")])
        h.run(diagnostics=True, roster=FakeRoster(), attendance_db_cls=mock.Mock(side_effect=must_not_construct))
        for frame_texts in h.texts_per_frame:
            self.assertIn(app_scanner.DIAG_HEADER, frame_texts)

    def test_reject_and_invalid_are_display_only(self):
        h = Harness(keys=[-1, -1, ord("q")], decisions=(REJECT, INVALID, INVALID))
        h.run(diagnostics=True, roster=FakeRoster(), attendance_db_cls=mock.Mock(side_effect=must_not_construct))
        texts = h.all_texts()
        self.assertIn("Liveness: REJECT | model live score 0.31 | threshold 0.50", texts)
        self.assertIn("Liveness: INVALID - check could not be completed | threshold 0.50", texts)
        self.assertFalse(any("spoof" in t.lower() for t in texts))

    def test_real_database_file_is_unchanged(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "diag.db"
            db = AttendanceDatabase(db_path=path)
            db.enroll_employee("SYN-0001", "Synthetic", EMB)
            del db
            gc.collect()
            before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in Path(tmp).iterdir()}

            h = Harness(keys=[-1, ord("e"), ord("q")])
            with mock.patch.dict("os.environ", {"DCCP_DB_PATH": str(path)}):
                h.run(diagnostics=True, attendance_db_cls=mock.Mock(side_effect=must_not_construct))
            gc.collect()
            after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in Path(tmp).iterdir()}
            self.assertEqual(after, before)
            self.assertIn("Identity: SYN-0001 (similarity 0.82)", h.all_texts())

    def test_missing_database_runs_liveness_only_without_creating_it(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "absent" / "attendance.db"
            h = Harness(keys=[-1, ord("q")], recognize=(None, 0.0, FACE))
            with mock.patch.dict("os.environ", {"DCCP_DB_PATH": str(path)}):
                h.run(diagnostics=True, attendance_db_cls=mock.Mock(side_effect=must_not_construct))
            self.assertFalse(path.parent.exists())
            texts = h.all_texts()
            self.assertIn("Identity matching off: database file not found", texts)
            self.assertIn("Liveness: PASS | model live score 0.91 | threshold 0.50", texts)


class TestDiagnosticsIdentityText(unittest.TestCase):
    def identity_texts(self, roster, recognize):
        h = Harness(keys=[ord("q")], recognize=recognize)
        h.run(diagnostics=True, roster=roster, attendance_db_cls=mock.Mock(side_effect=must_not_construct))
        return [t for t in h.all_texts() if t.startswith("Identity")]

    def test_cached_roster_is_labelled(self):
        roster = FakeRoster(state=readonly_roster.CACHED, reason="read error: database is locked")
        texts = self.identity_texts(roster, ("SYN-0000", 0.8, FACE))
        self.assertEqual(texts, ["Identity: SYN-0000 (similarity 0.80) [cached roster: latest refresh failed]"])

    def test_zero_employees_differs_from_unavailable(self):
        self.assertEqual(self.identity_texts(FakeRoster(count=0), (None, 0.0, FACE)),
                         ["Identity: no employees enrolled"])
        self.assertEqual(
            self.identity_texts(FakeRoster(state=readonly_roster.UNAVAILABLE, count=0, reason="database file not found"),
                                (None, 0.0, FACE)),
            ["Identity matching off: database file not found"])

    def test_no_match_is_not_called_unknown(self):
        texts = self.identity_texts(FakeRoster(count=2), (None, 0.21, FACE))
        self.assertEqual(texts, ["Identity: no enrolled match (best similarity 0.21)"])


class TestNormalModeParity(unittest.TestCase):
    """Normal attendance and enrollment semantics are unchanged by P5."""

    def test_pass_logs_attendance_once_with_same_arguments(self):
        h = Harness(keys=[ord("q")], decisions=(PASS,))
        h.run(diagnostics=False)
        h.db.log_attendance.assert_called_once_with(
            employee_id="SYN-0001", spoof_detected=False, liveness_score=0.91, similarity_score=0.82)
        h.liveness.evaluate.assert_called_once()
        h.liveness.check_passive_liveness.assert_not_called()
        self.assertIn("Liveness: PASS | model live score 0.91 | threshold 0.50", h.all_texts())

    def test_reject_logs_blocked_row_with_same_arguments(self):
        h = Harness(keys=[ord("q")], decisions=(REJECT,))
        h.run(diagnostics=False)
        h.db.log_attendance.assert_called_once_with(
            employee_id="SYN-0001", spoof_detected=True, liveness_score=0.31, similarity_score=0.82)
        self.assertIn(app_scanner.BLOCKED_BANNER, h.all_texts())
        self.assertFalse(any("SPOOF BLOCKED" in t for t in h.all_texts()))

    def test_invalid_logs_blocked_row_with_legacy_zero_score(self):
        h = Harness(keys=[ord("q")], decisions=(INVALID,))
        h.run(diagnostics=False)
        h.db.log_attendance.assert_called_once_with(
            employee_id="SYN-0001", spoof_detected=True, liveness_score=0.0, similarity_score=0.82)
        self.assertIn("Liveness: INVALID - check could not be completed | threshold 0.50", h.all_texts())

    def test_unknown_live_face_is_not_logged(self):
        h = Harness(keys=[ord("q")], decisions=(PASS,), recognize=(None, 0.2, FACE))
        h.run(diagnostics=False)
        h.db.log_attendance.assert_not_called()

    def test_quick_enroll_still_requires_liveness(self):
        h = Harness(keys=[ord("e"), ord("q")], enroll_liveness=(True, 0.9))
        h.run(diagnostics=False, process_interval=float("inf"))
        h.db.enroll_employee.assert_called_once_with("DCCP-KARL-001", "Karl Stephen V. Zubiri", EMB)

        h = Harness(keys=[ord("e"), ord("q")], enroll_liveness=(False, 0.1))
        h.run(diagnostics=False, process_interval=float("inf"))
        h.db.enroll_employee.assert_not_called()
        self.assertIn("Enrollment BLOCKED: liveness check failed", h.all_texts())


class TestStaleResults(unittest.TestCase):
    def test_result_cleared_when_face_disappears(self):
        faces = iter([("SYN-0001", 0.82, FACE), (None, 0.0, None), (None, 0.0, None)])
        h = Harness(keys=[-1, -1, ord("q")], recognize=lambda *a: next(faces))
        h.run(diagnostics=False)
        first, second, _ = h.texts_per_frame
        self.assertTrue(any(t.startswith("Liveness:") for t in first))
        self.assertFalse(any(t.startswith("Liveness:") for t in second))

    def test_freshness_window(self):
        fresh = app_scanner.liveness_result_is_fresh
        self.assertTrue(fresh(100.0, 100.9, 0.25))
        self.assertFalse(fresh(100.0, 101.1, 0.25))
        self.assertTrue(fresh(100.0, 102.9, 1.0))
        self.assertFalse(fresh(None, 100.0, 0.25))


class TestOverlayFitsFrame(unittest.TestCase):
    def test_long_text_is_scaled_inside_frame(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        long_text = "Identity matching off: " + "x" * 200
        calls = []
        real_put = app_scanner.cv2.putText

        def record(im, text, org, font, scale, color, thickness, *rest):
            calls.append((text, org, scale, thickness))
            return real_put(im, text, org, font, scale, color, thickness, *rest)

        with mock.patch.object(app_scanner.cv2, "putText", side_effect=record):
            app_scanner.draw_label(img, long_text, 470, (255, 255, 255))
        text, (x, y), scale, thickness = calls[0]
        (tw, th), base = app_scanner.cv2.getTextSize(text, app_scanner.cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(y + base, 480)
        self.assertGreaterEqual(y - th, 0)
        # Very long text may exceed the width at the minimum scale; typical lines fit.
        typical = "Liveness: INVALID - check could not be completed | threshold 0.50"
        calls.clear()
        with mock.patch.object(app_scanner.cv2, "putText", side_effect=record):
            app_scanner.draw_label(img, typical, 40, (255, 255, 255), 0.75, 2)
        text, (x, y), scale, thickness = calls[0]
        (tw, _), _ = app_scanner.cv2.getTextSize(text, app_scanner.cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        self.assertLessEqual(x + tw, 640)


class TestCommandLine(unittest.TestCase):
    def test_diagnostics_flag(self):
        with mock.patch.object(app_scanner, "run_attendance_scanner") as run:
            self.assertEqual(app_scanner.main(["--diagnostics"]), 0)
            run.assert_called_once_with(diagnostics=True)
        with mock.patch.object(app_scanner, "run_attendance_scanner") as run:
            self.assertEqual(app_scanner.main([]), 0)
            run.assert_called_once_with(diagnostics=False)
        with mock.patch.object(app_scanner, "run_attendance_scanner") as run:
            app_scanner.main()
            run.assert_called_once_with(diagnostics=False)


if __name__ == "__main__":
    unittest.main()
