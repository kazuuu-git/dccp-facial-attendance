"""
Dashboard view tests for the logs and export pages (src/dashboard/ui_views.py).

Verifies:
- Metric counts come only from real stored rows; sample preview rows never
  produce operational counts.
- Sample preview rows are clearly labelled and never exported.
- The status filter matches the values the database actually stores.
- Export is disabled when no real records exist and the inactive date control
  is disabled.
- No unsupported "Secure"/alert deltas are shown.

Streamlit is replaced with a mock. Data comes from a temporary SQLite database
selected through DCCP_DB_PATH, populated via AttendanceDatabase.log_attendance
with synthetic employee IDs.
"""

import gc
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.dashboard import ui_views
from src.database.db_manager import AttendanceDatabase


def fake_streamlit(selectbox_choice=ui_views.STATUS_ALL):
    st = mock.MagicMock()
    st.columns.side_effect = lambda spec, **kw: [
        mock.MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    st.button.return_value = False
    st.selectbox.return_value = selectbox_choice
    return st


class DashboardTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "dashboard.db"
        self.db = AttendanceDatabase(db_path=self.db_path)  # creates empty tables
        self.env = mock.patch.dict(os.environ, {"DCCP_DB_PATH": str(self.db_path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        del self.db
        gc.collect()
        self.temp_dir.cleanup()

    def populate(self):
        # SYN-A: TIME_IN then TIME_OUT on one row; SYN-B: TIME_IN;
        # SYN-A blocked attempt; UNKNOWN blocked attempt.
        self.db.log_attendance("SYN-A", spoof_detected=False, liveness_score=0.9, similarity_score=0.8)
        self.db.log_attendance("SYN-A", spoof_detected=False, liveness_score=0.9, similarity_score=0.8)
        self.db.log_attendance("SYN-B", spoof_detected=False, liveness_score=0.9, similarity_score=0.8)
        self.db.log_attendance("SYN-A", spoof_detected=True, liveness_score=0.1, similarity_score=0.8)
        self.db.log_attendance(None, spoof_detected=True, liveness_score=0.0, similarity_score=0.0)

    def render(self, view, **kwargs):
        st = fake_streamlit(**kwargs)
        with mock.patch.object(ui_views, "st", st):
            view()
        return st


class TestLogsView(DashboardTestCase):
    def metric_calls(self, st):
        return {c.kwargs["label"]: c.kwargs for c in st.metric.call_args_list}

    def test_empty_db_shows_labelled_sample_without_counts(self):
        st = self.render(ui_views.render_logs_view)

        notice = st.info.call_args.args[0]
        self.assertIn("sample data", notice.lower())
        self.assertIn("not from the database", notice)

        metrics = self.metric_calls(st)
        self.assertEqual(set(metrics), {"Log entries", "Passed entries", "Blocked attempts"})
        for kwargs in metrics.values():
            self.assertEqual(kwargs["value"], "—", "no counts may be derived from sample rows")

        shown = st.dataframe.call_args.args[0]
        self.assertTrue(shown["employee_id"].str.startswith("SAMPLE-").all())
        self.assertTrue(any("Sample data" in c.args[0] for c in st.caption.call_args_list))

    def test_populated_db_counts_real_rows(self):
        self.populate()
        st = self.render(ui_views.render_logs_view)

        st.info.assert_not_called()
        metrics = self.metric_calls(st)
        self.assertEqual(metrics["Log entries"]["value"], 4)
        self.assertEqual(metrics["Passed entries"]["value"], 2)
        self.assertEqual(metrics["Blocked attempts"]["value"], 2)

        shown = st.dataframe.call_args.args[0]
        self.assertFalse(shown["employee_id"].str.startswith("SAMPLE-").any())

    def test_no_unsupported_deltas_and_native_cards(self):
        self.populate()
        st = self.render(ui_views.render_logs_view)
        for call in st.metric.call_args_list:
            self.assertNotIn("delta", call.kwargs)
            self.assertTrue(call.kwargs.get("border"))
            self.assertTrue(call.kwargs.get("help"))

    def test_filter_options_and_blocked_selection(self):
        self.populate()
        st = self.render(ui_views.render_logs_view, selectbox_choice=ui_views.STATUS_BLOCKED)
        options = st.selectbox.call_args.args[1]
        self.assertEqual(options, ["All", "Passed", "Blocked (liveness check did not pass)"])
        shown = st.dataframe.call_args.args[0]
        self.assertEqual(sorted(shown["employee_id"]), ["SYN-A", "UNKNOWN"])
        self.assertTrue(shown["spoof_status"].str.startswith("Failed").all())


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "employee_id": ["A", "B", "UNKNOWN", "C"],
            "spoof_status": [
                "Passed",
                "Failed (Spoof Detected, Score: 0.10)",
                "Failed (Spoof Detected, Score: 0.00)",
                "Passed",
            ],
        })

    def test_summarize_logs(self):
        self.assertEqual(ui_views.summarize_logs(self.df), (4, 2, 2))

    def test_filter_logs_by_status(self):
        f = ui_views.filter_logs_by_status
        self.assertEqual(list(f(self.df, "All")["employee_id"]), ["A", "B", "UNKNOWN", "C"])
        self.assertEqual(list(f(self.df, "Passed")["employee_id"]), ["A", "C"])
        self.assertEqual(list(f(self.df, ui_views.STATUS_BLOCKED)["employee_id"]), ["B", "UNKNOWN"])


class TestExportView(DashboardTestCase):
    def test_empty_db_disables_export_and_has_no_sample(self):
        st = self.render(ui_views.render_export_view)

        download = st.download_button.call_args.kwargs
        self.assertTrue(download["disabled"])
        self.assertEqual(download["data"], b"")
        st.dataframe.assert_not_called()
        st.info.assert_called_with("No attendance records to export yet.")

    def test_controls_disabled_when_empty_db(self):
        st = self.render(ui_views.render_export_view)
        self.assertTrue(st.date_input.call_args.kwargs["disabled"])
        self.assertTrue(st.selectbox.call_args.kwargs["disabled"])
        captions = " ".join(c.args[0] for c in st.caption.call_args_list)
        self.assertIn("Export contains", captions)

    def test_populated_db_exports_real_rows_including_blocked(self):
        self.populate()
        st = self.render(ui_views.render_export_view)

        download = st.download_button.call_args.kwargs
        self.assertFalse(download["disabled"])
        csv_text = download["data"].decode("utf-8")
        self.assertIn("SYN-A", csv_text)
        self.assertIn("UNKNOWN", csv_text)
        self.assertNotIn("SAMPLE-", csv_text)
        self.assertNotIn("DCCP-100", csv_text)
        self.assertEqual(len(csv_text.strip().splitlines()), 1 + 4)


class TestManagePersonnelView(DashboardTestCase):
    def test_empty_db_shows_info(self):
        st = self.render(ui_views.render_manage_personnel_view)
        st.info.assert_called()
        self.assertIn("No personnel currently enrolled", st.info.call_args.args[0])

    def test_populated_db_lists_employees(self):
        import numpy as np
        v = np.ones(512, dtype=np.float32)
        self.db.enroll_employee("EMP-001", "Test User", v)
        st = self.render(ui_views.render_manage_personnel_view)
        st.dataframe.assert_called()
        df = st.dataframe.call_args.args[0]
        self.assertEqual(list(df["employee_id"]), ["EMP-001"])
        self.assertEqual(list(df["full_name"]), ["Test User"])


if __name__ == "__main__":
    unittest.main()
