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
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.dashboard import ui_views
from src.database.db_manager import AttendanceDatabase


def fake_streamlit(selectbox_choice=ui_views.STATUS_ALL, text_value="", checkbox_value=False,
                   button_value=False):
    """Mocked Streamlit whose widgets return the same defaults as the real UI.

    The logs-view status filter returns `selectbox_choice`; every other selectbox
    returns its first option. Date inputs return their default value.
    """
    st = mock.MagicMock()
    st.columns.side_effect = lambda spec, **kw: [
        mock.MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    st.button.return_value = button_value
    st.text_input.return_value = text_value
    st.checkbox.return_value = checkbox_value
    st.selectbox.side_effect = lambda label, options=None, *a, **k: (
        selectbox_choice if label.startswith("Filter Display") else options[0]
    )
    st.date_input.side_effect = lambda label, value=None, *a, **k: value
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
        self.assertIn("both passed and blocked log entries", captions)
        self.assertIn("not a calculated payroll report", captions)

    def test_payroll_checkbox_wording_does_not_claim_spoofs(self):
        st = self.render(ui_views.render_export_view)
        label = st.checkbox.call_args.args[0]
        self.assertEqual(label, "Payroll-ready only (exclude blocked attempts)")
        self.assertNotIn("spoof", label.lower())

    def test_default_date_range_covers_all_recorded_dates(self):
        self.populate()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO logs (employee_id, date, time_in, time_out, spoof_status) "
            "VALUES ('SYN-OLD', '2026-01-05', '08:00:00 AM', '05:00:00 PM', 'Passed')"
        )
        conn.commit()
        conn.close()
        st = self.render(ui_views.render_export_view)

        start, end = st.date_input.call_args.kwargs["value"]
        self.assertEqual(start.isoformat(), "2026-01-05")
        csv_text = st.download_button.call_args.kwargs["data"].decode("utf-8")
        self.assertIn("SYN-OLD", csv_text, "older dates are exported by default")
        self.assertEqual(len(csv_text.strip().splitlines()), 1 + 5)

    def test_payroll_only_excludes_blocked_rows(self):
        self.populate()
        st = self.render(ui_views.render_export_view, checkbox_value=True)
        csv_text = st.download_button.call_args.kwargs["data"].decode("utf-8")
        self.assertNotIn("UNKNOWN", csv_text)
        self.assertNotIn("Failed", csv_text)
        self.assertEqual(len(csv_text.strip().splitlines()), 1 + 2)

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
        self.assertEqual(st.dataframe.call_args.kwargs["column_config"]["created_at"], "Enrolled On (UTC)")

    def _enroll_two(self):
        import numpy as np
        v = np.ones(512, dtype=np.float32)
        self.db.enroll_employee("EMP-001", "Alpha User", v)
        self.db.enroll_employee("EMP-002", "Bravo User", v)

    def test_delete_confirmation_is_per_employee(self):
        self._enroll_two()
        st = self.render(ui_views.render_manage_personnel_view)
        self.assertEqual(st.checkbox.call_args.kwargs["key"], "confirm_emp_del_EMP-001")
        self.assertTrue(st.button.call_args.kwargs["disabled"], "delete stays disabled until confirmed")

    def test_confirmed_delete_removes_employee_and_clears_its_confirmation(self):
        self._enroll_two()
        st = self.render(ui_views.render_manage_personnel_view, checkbox_value=True, button_value=True)
        remaining = [e["employee_id"] for e in self.db.list_employees()]
        self.assertEqual(remaining, ["EMP-002"])
        st.session_state.pop.assert_called_with("confirm_emp_del_EMP-001", None)

    def test_unconfirmed_delete_does_nothing(self):
        self._enroll_two()
        self.render(ui_views.render_manage_personnel_view, checkbox_value=False, button_value=False)
        self.assertEqual(len(self.db.list_employees()), 2)

    def test_search_with_special_characters_does_not_crash(self):
        self._enroll_two()
        for text in ["(", "*", "[a", "user"]:
            with self.subTest(text=text):
                st = self.render(ui_views.render_manage_personnel_view, text_value=text)
                df = st.dataframe.call_args.args[0]
                expected = 2 if text == "user" else 0
                self.assertEqual(len(df), expected)


if __name__ == "__main__":
    unittest.main()
