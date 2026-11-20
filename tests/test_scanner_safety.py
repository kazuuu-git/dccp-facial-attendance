"""
Scanner safety tests (no camera, no models, no real database).

P2: keyboard handling — 'q' quits, 'e' never exits the loop, and quick-enroll
    still requires a detected face and a passing liveness check.
P3: required components — the scanner must not fall back to an always-pass
    liveness stub or an in-memory database, and must fail before opening the
    camera if a required component cannot import or initialize.

Every collaborator (camera, OpenCV window calls, recognition, liveness, database)
is replaced with an explicit test double inside each test.
"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app_scanner

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
SYNTHETIC_EMBEDDING = np.ones(512, dtype=np.float32) / np.sqrt(512)
FACE_BBOX = [100, 100, 300, 300]
NO_KEY = -1
MAX_READS = 50  # safety stop so a broken quit key cannot hang the test


class FakeCapture:
    def __init__(self, *args, **kwargs):
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        self.reads += 1
        if self.reads > MAX_READS:
            return False, None
        return True, FRAME.copy()

    def release(self):
        self.released = True


class FakeDatabase:
    def __init__(self, *args, **kwargs):
        self.db_path = Path("nonexistent-test-only.db")
        self.enroll_employee = mock.Mock(return_value=True)
        self.log_attendance = mock.Mock(return_value="TIME_IN")

    def load_enrolled_embeddings(self):
        return {}

    def load_enrolled_profiles(self):
        return {}, {}


class ScannerHarness:
    """Runs run_attendance_scanner with every external dependency replaced."""

    def __init__(self, keys, liveness=(True, 0.9), face_found=True):
        self.capture = FakeCapture()
        self.db = FakeDatabase()
        self.liveness = mock.Mock()
        self.liveness.check_passive_liveness.return_value = liveness
        self.recognition = mock.Mock()
        self.recognition.extract_embedding.return_value = (
            (SYNTHETIC_EMBEDDING, FACE_BBOX) if face_found else None
        )
        self.keys = list(keys)

    def _wait_key(self, _delay):
        return self.keys.pop(0) if self.keys else NO_KEY

    def run(self):
        with mock.patch.object(app_scanner, "AttendanceDatabase", return_value=self.db), \
             mock.patch.object(app_scanner, "PassiveLivenessEngine", return_value=self.liveness), \
             mock.patch.object(app_scanner, "FaceRecognitionEngine", return_value=self.recognition), \
             mock.patch.object(app_scanner.cv2, "VideoCapture", return_value=self.capture), \
             mock.patch.object(app_scanner.cv2, "imshow"), \
             mock.patch.object(app_scanner.cv2, "destroyAllWindows"), \
             mock.patch.object(app_scanner.cv2, "waitKey", side_effect=self._wait_key):
            # An infinite interval skips the timed recognition branch so only
            # the keyboard handling is exercised.
            app_scanner.run_attendance_scanner(process_interval=float("inf"))


class TestScannerKeyboard(unittest.TestCase):
    def test_q_quits_the_loop(self):
        h = ScannerHarness([NO_KEY, NO_KEY, ord("q")])
        h.run()
        self.assertEqual(h.capture.reads, 3)
        self.assertTrue(h.capture.released)

    def test_e_with_live_face_enrolls_and_keeps_running(self):
        h = ScannerHarness([NO_KEY, ord("e"), NO_KEY, NO_KEY, ord("q")], liveness=(True, 0.9))
        h.run()
        self.assertEqual(h.capture.reads, 5, "'e' must not exit the scanner; 'q' must")
        h.liveness.check_passive_liveness.assert_called_once()
        h.db.enroll_employee.assert_called_once()
        h.db.log_attendance.assert_not_called()

    def test_e_with_failed_liveness_does_not_enroll_and_keeps_running(self):
        h = ScannerHarness([ord("e"), NO_KEY, ord("q")], liveness=(False, 0.12))
        h.run()
        self.assertEqual(h.capture.reads, 3)
        h.db.enroll_employee.assert_not_called()
        h.db.log_attendance.assert_not_called()

    def test_e_without_face_does_not_enroll_and_keeps_running(self):
        h = ScannerHarness([ord("e"), NO_KEY, ord("q")], face_found=False)
        h.run()
        self.assertEqual(h.capture.reads, 3)
        h.liveness.check_passive_liveness.assert_not_called()
        h.db.enroll_employee.assert_not_called()


class TestNoUnsafeFallbacks(unittest.TestCase):
    """Import failures of required modules must stop the scanner, not be mocked."""

    def _import_scanner_with_broken(self, module_name):
        # Mapping a module to None in sys.modules makes importing it raise ImportError.
        with mock.patch.dict(sys.modules, {module_name: None}):
            sys.modules.pop("app_scanner", None)
            try:
                importlib.import_module("app_scanner")
            finally:
                sys.modules.pop("app_scanner", None)

    def tearDown(self):
        # Restore the normally imported module for other tests.
        sys.modules["app_scanner"] = app_scanner

    def test_liveness_import_failure_is_fatal_and_keeps_cause(self):
        with self.assertRaises(ImportError) as ctx:
            self._import_scanner_with_broken("src.liveness.anti_spoof")
        self.assertIn("src.liveness.anti_spoof", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__, "original error must be chained")

    def test_database_import_failure_is_fatal_and_keeps_cause(self):
        with self.assertRaises(ImportError) as ctx:
            self._import_scanner_with_broken("src.database.db_manager")
        self.assertIn("src.database.db_manager", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__, "original error must be chained")

    def test_real_components_are_used(self):
        from src.database.db_manager import AttendanceDatabase
        from src.liveness.anti_spoof import PassiveLivenessEngine

        self.assertIs(app_scanner.PassiveLivenessEngine, PassiveLivenessEngine)
        self.assertIs(app_scanner.AttendanceDatabase, AttendanceDatabase)


class TestInitFailuresStopBeforeCamera(unittest.TestCase):
    def _run_with_failing(self, attr):
        error = RuntimeError(f"synthetic {attr} init failure")
        video_capture = mock.Mock()
        patches = {
            "AttendanceDatabase": mock.Mock(return_value=FakeDatabase()),
            "PassiveLivenessEngine": mock.Mock(),
            "FaceRecognitionEngine": mock.Mock(),
        }
        patches[attr] = mock.Mock(side_effect=error)
        with mock.patch.multiple(app_scanner, **patches), \
             mock.patch.object(app_scanner.cv2, "VideoCapture", video_capture):
            with self.assertRaises(RuntimeError) as ctx:
                app_scanner.run_attendance_scanner()
        self.assertIs(ctx.exception, error, "original error must propagate unchanged")
        video_capture.assert_not_called()

    def test_liveness_init_failure_never_opens_camera(self):
        self._run_with_failing("PassiveLivenessEngine")

    def test_database_init_failure_never_opens_camera(self):
        self._run_with_failing("AttendanceDatabase")


if __name__ == "__main__":
    unittest.main()
