"""
Regression tests for the dashboard enrollment path (src/dashboard/ui_views.py).

Verifies that the merged UI cannot enroll a face unless passive liveness passes,
and that success feedback (toast, success message) only appears after the
embedding has actually been saved.

Streamlit, the camera image, face recognition and liveness are replaced with
test doubles. Enrollment writes go to a temporary SQLite database.
"""

import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.dashboard import ui_views
from src.database.db_manager import AttendanceDatabase

EMP_ID = "SYN-0001"
EMP_NAME = "Synthetic Person"
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
EMBEDDING = np.ones(512, dtype=np.float32) / np.sqrt(512)
BBOX = [100, 100, 300, 300]


class EnrollmentHarness:
    def __init__(self, db_path, *, face_found=True, liveness=(True, 0.9),
                 liveness_error=None, enroll_error=None):
        self.events = []
        self.db = AttendanceDatabase(db_path=db_path)
        real_enroll = self.db.enroll_employee

        def enroll(*args, **kwargs):
            if enroll_error is not None:
                raise enroll_error
            result = real_enroll(*args, **kwargs)
            self.events.append("enroll")
            return result

        self.db.enroll_employee = mock.Mock(side_effect=enroll)

        self.recognition = mock.Mock()
        self.recognition.extract_embedding.return_value = (EMBEDDING, BBOX) if face_found else None

        self.liveness = mock.Mock()
        if liveness_error is not None:
            self.liveness.check_passive_liveness.side_effect = liveness_error
        else:
            self.liveness.check_passive_liveness.return_value = liveness

        self.st = self._fake_streamlit()

    def _fake_streamlit(self):
        st = mock.MagicMock()
        st.columns.side_effect = lambda spec, **kw: [
            mock.MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        st.text_input.side_effect = lambda label, **kw: {"Employee ID": EMP_ID, "Full Name": EMP_NAME}[label]
        st.form_submit_button.return_value = True
        camera_image = mock.Mock()
        camera_image.read.return_value = b"synthetic-image-bytes"
        st.camera_input.return_value = camera_image
        for name in ("toast", "success", "balloons", "error"):
            getattr(st, name).side_effect = (lambda n: lambda *a, **k: self.events.append(n))(name)
        return st

    def run(self):
        with mock.patch.object(ui_views, "st", self.st), \
             mock.patch.object(ui_views, "AttendanceDatabase", return_value=self.db), \
             mock.patch.object(ui_views, "FaceRecognitionEngine", return_value=self.recognition), \
             mock.patch.object(ui_views, "PassiveLivenessEngine", return_value=self.liveness), \
             mock.patch.object(ui_views.cv2, "imdecode", return_value=FRAME):
            ui_views.render_enrollment_view()

    def enrolled_ids(self):
        return set(self.db.load_enrolled_embeddings())


class TestEnrollmentRequiresLiveness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "enroll.db"

    def tearDown(self):
        gc.collect()
        self.temp_dir.cleanup()

    def _run(self, **kwargs):
        h = EnrollmentHarness(self.db_path, **kwargs)
        h.run()
        return h

    def test_failed_liveness_blocks_enrollment(self):
        h = self._run(liveness=(False, 0.12))
        h.liveness.check_passive_liveness.assert_called_once()
        _, bbox_arg = h.liveness.check_passive_liveness.call_args.args
        self.assertEqual(bbox_arg, BBOX, "liveness must be checked on the detected face")
        h.db.enroll_employee.assert_not_called()
        self.assertEqual(h.enrolled_ids(), set())
        self.assertEqual(h.events, ["error"])

    def test_passing_liveness_enrolls_then_shows_feedback(self):
        h = self._run(liveness=(True, 0.9))
        self.assertEqual(h.enrolled_ids(), {EMP_ID})
        self.assertEqual(h.events, ["enroll", "toast", "success"])
        h.st.balloons.assert_not_called()

    def test_no_face_skips_liveness_and_enrollment(self):
        h = self._run(face_found=False)
        h.liveness.check_passive_liveness.assert_not_called()
        h.db.enroll_employee.assert_not_called()
        self.assertEqual(h.events, ["error"])

    def test_liveness_exception_blocks_enrollment(self):
        h = self._run(liveness_error=RuntimeError("synthetic inference failure"))
        h.db.enroll_employee.assert_not_called()
        self.assertEqual(h.enrolled_ids(), set())
        self.assertEqual(h.events, ["error"])

    def test_failed_save_shows_no_success_feedback(self):
        h = self._run(enroll_error=ValueError("synthetic save failure"))
        self.assertEqual(h.enrolled_ids(), set())
        self.assertEqual(h.events, ["error"])


if __name__ == "__main__":
    unittest.main()
