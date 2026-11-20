"""
P4 scanner tests: bounded camera-read retries, configuration validation,
stale-result clearing, resource cleanup and exit codes.

The camera, OpenCV window calls, recognition, liveness and database are all
test doubles; no camera is opened and no database is touched.
"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import app_scanner

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
OK, FAIL, EMPTY, NONE_FRAME = "ok", "fail", "empty", "none"


class ScriptedCapture:
    """Returns reads according to a script, then good frames."""

    def __init__(self, script, backend="FAKE_BACKEND", backend_error=None):
        self.script = list(script)
        self.reads = 0
        self.released = False
        self.backend = backend
        self.backend_error = backend_error

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def getBackendName(self):
        if self.backend_error is not None:
            raise self.backend_error
        return self.backend

    def read(self):
        self.reads += 1
        step = self.script.pop(0) if self.script else OK
        if step == FAIL:
            return False, None
        if step == EMPTY:
            return True, np.empty((0, 0, 3), dtype=np.uint8)
        if step == NONE_FRAME:
            return True, None
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


class Harness:
    def __init__(self, script, keys=None, **capture_kwargs):
        self.capture = ScriptedCapture(script, **capture_kwargs)
        self.db = FakeDatabase()
        self.liveness = mock.Mock()
        self.liveness.check_passive_liveness.return_value = (True, 0.9)
        self.recognition = mock.Mock()
        self.recognition.recognize_employee.return_value = (None, 0.0, None)
        self.recognition.extract_embedding.return_value = None
        self.keys = list(keys or [])
        self.put_texts = []
        self.stdout = io.StringIO()
        self.destroy = mock.Mock()

    def _wait_key(self, _delay):
        return self.keys.pop(0) if self.keys else -1

    def _put_text(self, img, text, *args, **kwargs):
        self.put_texts.append(text)
        return img

    def run(self, **kwargs):
        kwargs.setdefault("read_retry_delay", 0.0)
        with mock.patch.object(app_scanner, "AttendanceDatabase", return_value=self.db), \
             mock.patch.object(app_scanner, "PassiveLivenessEngine", return_value=self.liveness), \
             mock.patch.object(app_scanner, "FaceRecognitionEngine", return_value=self.recognition), \
             mock.patch.object(app_scanner.cv2, "VideoCapture", return_value=self.capture), \
             mock.patch.object(app_scanner.cv2, "imshow"), \
             mock.patch.object(app_scanner.cv2, "putText", side_effect=self._put_text), \
             mock.patch.object(app_scanner.cv2, "destroyAllWindows", self.destroy), \
             mock.patch.object(app_scanner.cv2, "waitKey", side_effect=self._wait_key), \
             redirect_stdout(self.stdout):
            app_scanner.run_attendance_scanner(**kwargs)

    def frames_given_to_recognition(self):
        return [c.args[0] for c in self.recognition.recognize_employee.call_args_list]


def keys_quit_after(valid_frames):
    """waitKey is called once per loop iteration; quit on the given iteration."""
    return [-1] * (valid_frames - 1) + [ord("q")]


class TestReadRetries(unittest.TestCase):
    def test_recovers_after_transient_failures(self):
        h = Harness([OK, FAIL, FAIL, OK, OK], keys=keys_quit_after(5))
        h.run(max_read_failures=3, process_interval=0.0)
        self.assertEqual(h.capture.reads, 5)
        self.assertTrue(h.capture.released)
        frames = h.frames_given_to_recognition()
        self.assertEqual(len(frames), 3, "only the 3 valid frames reach recognition")
        for f in frames:
            self.assertIsInstance(f, np.ndarray)
            self.assertGreater(f.size, 0)

    def test_failure_counter_resets_after_valid_frame(self):
        # 4 failures in total but never 3 in a row: must not stop with limit 3.
        h = Harness([FAIL, FAIL, OK, FAIL, FAIL, OK], keys=keys_quit_after(6))
        h.run(max_read_failures=3, process_interval=0.0)
        self.assertEqual(h.capture.reads, 6)

    def test_empty_and_none_frames_count_as_failures(self):
        with self.assertRaises(app_scanner.CameraReadError):
            Harness([EMPTY, NONE_FRAME, FAIL]).run(max_read_failures=3, process_interval=0.0)

        h = Harness([EMPTY, NONE_FRAME, OK], keys=keys_quit_after(3))
        h.run(max_read_failures=3, process_interval=0.0)
        self.assertEqual(len(h.frames_given_to_recognition()), 1)

    def test_retry_exhaustion_raises_and_cleans_up(self):
        h = Harness([OK, FAIL, FAIL, FAIL, OK])
        with self.assertRaises(app_scanner.CameraReadError) as ctx:
            h.run(max_read_failures=3, process_interval=0.0)
        message = str(ctx.exception)
        self.assertIn("3 times in a row", message)
        self.assertIn("camera index 0", message)
        self.assertIn("backend FAKE_BACKEND", message)
        self.assertIn("Possible causes", message)
        self.assertIn("has not been diagnosed", message)
        self.assertEqual(h.capture.reads, 4, "no reads after the limit")
        self.assertTrue(h.capture.released)
        h.destroy.assert_called()
        h.db.log_attendance.assert_not_called()
        h.db.enroll_employee.assert_not_called()

    def test_failed_frames_never_reach_enrollment(self):
        # 'e' pressed while frames are failing must not trigger quick-enroll.
        h = Harness([FAIL, FAIL, OK], keys=[ord("e"), ord("e"), ord("q")])
        h.run(max_read_failures=5, process_interval=float("inf"))
        h.recognition.extract_embedding.assert_not_called()
        h.db.enroll_employee.assert_not_called()

    def test_stale_result_is_cleared_and_status_shown(self):
        # A large interval means recognition runs on the first frame only,
        # unless a read failure forces fresh processing.
        h = Harness([OK, FAIL, OK], keys=keys_quit_after(3))
        h.run(max_read_failures=5, process_interval=1e6)
        self.assertEqual(h.recognition.recognize_employee.call_count, 2,
                         "recognition must run again on the first frame after a failure")
        self.assertTrue(any("Camera frame unavailable" in t for t in h.put_texts))
        self.assertTrue(any("no faces are being checked" in t for t in h.put_texts))

    def test_q_during_failures_quits_normally(self):
        h = Harness([FAIL, FAIL], keys=[-1, ord("q")])
        h.run(max_read_failures=10, process_interval=0.0)
        self.assertEqual(h.capture.reads, 2)
        self.assertTrue(h.capture.released)


class TestCameraInfoAndCleanup(unittest.TestCase):
    def test_prints_camera_index_and_backend(self):
        h = Harness([OK], keys=[ord("q")])
        h.run(process_interval=0.0)
        self.assertIn("Camera opened: index 0, backend FAKE_BACKEND", h.stdout.getvalue())

    def test_backend_name_failure_does_not_stop_scanner(self):
        h = Harness([OK, OK], keys=keys_quit_after(2), backend_error=RuntimeError("no backend info"))
        h.run(process_interval=0.0)
        self.assertEqual(h.capture.reads, 2)
        self.assertIn("backend unknown", h.stdout.getvalue())

    def test_ctrl_c_releases_camera_and_closes_windows(self):
        h = Harness([OK, OK])
        h.keys = []
        with mock.patch.object(h, "_wait_key", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                h.run(process_interval=0.0)
        self.assertTrue(h.capture.released)
        h.destroy.assert_called()

    def test_cleanup_continues_if_release_fails(self):
        h = Harness([OK], keys=[ord("q")])
        h.capture.release = mock.Mock(side_effect=RuntimeError("release failed"))
        h.run(process_interval=0.0)
        h.destroy.assert_called()


class TestConfiguration(unittest.TestCase):
    def test_parse_valid_and_default(self):
        p = app_scanner.parse_max_read_failures
        self.assertEqual(p(None), app_scanner.DEFAULT_MAX_READ_FAILURES)
        self.assertEqual(p("  "), app_scanner.DEFAULT_MAX_READ_FAILURES)
        self.assertEqual(p("5"), 5)
        self.assertEqual(p(" 12 "), 12)

    def test_parse_rejects_invalid_values(self):
        for raw in ("0", "-1", "abc", "1.5", "3x"):
            with self.subTest(raw=raw), self.assertRaises(app_scanner.ScannerConfigError):
                app_scanner.parse_max_read_failures(raw)

    def test_invalid_env_fails_before_models_or_camera(self):
        h = Harness([OK])
        with mock.patch.dict("os.environ", {"DCCP_MAX_READ_FAILURES": "0"}), \
             mock.patch.object(app_scanner, "PassiveLivenessEngine") as liveness_cls, \
             mock.patch.object(app_scanner.cv2, "VideoCapture") as video_capture:
            with self.assertRaises(app_scanner.ScannerConfigError):
                app_scanner.run_attendance_scanner()
        liveness_cls.assert_not_called()
        video_capture.assert_not_called()

    def test_invalid_argument_rejected(self):
        for bad in (0, -3, True, 2.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(app_scanner.ScannerConfigError):
                app_scanner.run_attendance_scanner(max_read_failures=bad)


class TestMainExitCodes(unittest.TestCase):
    def _main_with(self, side_effect=None):
        with mock.patch.object(app_scanner, "run_attendance_scanner", side_effect=side_effect), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return app_scanner.main()

    def test_exit_codes(self):
        self.assertEqual(self._main_with(None), 0)
        self.assertEqual(self._main_with(app_scanner.CameraReadError("x")), 2)
        self.assertEqual(self._main_with(app_scanner.ScannerConfigError("x")), 3)
        self.assertEqual(self._main_with(KeyboardInterrupt()), 130)


if __name__ == "__main__":
    unittest.main()
