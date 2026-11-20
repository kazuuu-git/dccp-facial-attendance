from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import export_utils
from src.database.db_manager import AttendanceDatabase
from src.liveness.anti_spoof import PassiveLivenessEngine
from src.recognition.matcher import FaceRecognitionEngine

def inject_custom_css():
    """Injects custom CSS to style the Streamlit interface cleanly."""
    st.markdown("""
        <style>
        /* Custom Header Banners: colours derive from the active theme's text
           colour (currentColor), so they follow light/dark switching. */
        .dashboard-header {
            background: rgba(128, 128, 128, 0.06);
            background: color-mix(in srgb, currentColor 4%, transparent);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
            border-left: 4px solid #2563eb;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .dashboard-header h2 { margin: 0; padding: 0; color: inherit; font-size: 1.4rem; font-weight: 600; }
        .dashboard-header p { margin: 4px 0 0 0; color: inherit; opacity: 0.72; font-size: 0.9rem; }
        </style>
    """, unsafe_allow_html=True)


# Status filter options, matched against the values the database actually stores:
# "Passed" for verified attendance, "Failed (...)" for blocked attempts.
STATUS_ALL = "All"
STATUS_PASSED = "Passed"
STATUS_BLOCKED = "Blocked (liveness check did not pass)"
STATUS_FILTER_OPTIONS = [STATUS_ALL, STATUS_PASSED, STATUS_BLOCKED]

METRIC_HELP_ENTRIES = (
    "Stored attendance and blocked-attempt rows. A time-in and its time-out "
    "share one row. This is not a count of every camera scan."
)
METRIC_HELP_PASSED = (
    "Rows where the face matched an enrolled employee and the liveness check "
    "passed. The time-out, if any, is stored on the same row."
)
METRIC_HELP_BLOCKED = (
    "Rows stored when the liveness check did not pass, for recognized or "
    "unrecognized (UNKNOWN) faces. A blocked attempt is not necessarily a "
    "confirmed spoof; failed or invalid checks are included. Repeated blocks "
    "within the scanner's cooldown are stored once."
)


def summarize_logs(df):
    """Returns (log entries, passed entries, blocked entries) for real log rows."""
    total_logs = len(df)
    passed_logs = len(df[df["spoof_status"] == "Passed"]) if "spoof_status" in df.columns else 0
    return total_logs, passed_logs, total_logs - passed_logs


def filter_logs_by_status(df, choice):
    """Filters log rows by a STATUS_FILTER_OPTIONS choice using stored status values."""
    if choice == STATUS_PASSED:
        return df[df["spoof_status"] == "Passed"]
    if choice == STATUS_BLOCKED:
        return df[df["spoof_status"].astype(str).str.startswith("Failed")]
    return df


def _sample_preview_logs():
    """Display-only layout preview rows. Never written to the database or exported."""
    return pd.DataFrame({
        "employee_id": ["SAMPLE-01", "SAMPLE-02", "SAMPLE-03", "SAMPLE-04"],
        "full_name": ["Juan Dela Cruz", "Maria Santos", "Pedro Penduko", "Unknown / Not Enrolled"],
        "date": ["2026-10-28", "2026-10-28", "2026-10-28", "2026-10-28"],
        "time_in": ["07:45 AM", "07:52 AM", "08:05 AM", "--"],
        "time_out": ["--", "--", "--", "--"],
        "spoof_status": ["Passed", "Passed", "Passed", "Failed (Spoof Detected, Score: 0.12)"],
    })


def render_logs_view():
    inject_custom_css()
    
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("""
            <div class="dashboard-header">
                <h2>Live Attendance Logs</h2>
                <p>Monitor real-time employee clock-in records and passive anti-spoofing outcomes.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.write("")
        if st.button("🔄 Refresh Logs", use_container_width=True, type="primary"):
            st.rerun()

    # Fetch real data from SQLite
    df = export_utils.fetch_attendance_logs()

    # Empty log: show a display-only sample table, but never derive counts from it.
    is_sample = df.empty
    if is_sample:
        st.info(
            "No attendance records found in the configured database. The table below is "
            "**sample data for layout preview only**; it is not from the database."
        )
        df = _sample_preview_logs()
        metric_values = ("—", "—", "—")
    else:
        metric_values = summarize_logs(df)

    met1, met2, met3 = st.columns(3)
    with met1:
        st.metric(label="Log entries", value=metric_values[0], help=METRIC_HELP_ENTRIES, border=True)
    with met2:
        st.metric(label="Passed entries", value=metric_values[1], help=METRIC_HELP_PASSED, border=True)
    with met3:
        st.metric(label="Blocked attempts", value=metric_values[2], help=METRIC_HELP_BLOCKED, border=True)

    st.divider()

    # Interactive Filter
    status_filter = st.selectbox("Filter Display by Verification Status", STATUS_FILTER_OPTIONS)
    display_df = filter_logs_by_status(df, status_filter)

    if is_sample:
        st.caption("Sample data: display only, not stored or exported.")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_enrollment_view():
    inject_custom_css()
    
    st.markdown("""
        <div class="dashboard-header green">
            <h2>Biometric Enrollment Module</h2>
            <p>Capture and store mathematical 512-dimensional facial embeddings for new personnel.</p>
        </div>
    """, unsafe_allow_html=True)

    col_form, col_cam = st.columns([1, 1], gap="large")
    
    with col_form:
        st.subheader("1. Employee Information")
        with st.form("enrollment_form", clear_on_submit=False):
            emp_id = st.text_input("Employee ID", placeholder="e.g. DCCP-1001")
            emp_name = st.text_input("Full Name", placeholder="e.g. Juan Cruz")
            
            st.info("💡 Ensure the face is well-lit and directly facing the camera for optimal embedding extraction.")
            
            st.subheader("3. Verification")
            submit_btn = st.form_submit_button("Extract & Save Embedding", type="primary", use_container_width=True)

    with col_cam:
        st.subheader("2. Face Capture")
        image_data = st.camera_input("Position face within the frame", label_visibility="collapsed")

    # Handle Submission (Outside the columns to use full width for alerts)
    if submit_btn:
        if not emp_id or not emp_name:
            st.error("⚠️ Please provide both an Employee ID and Full Name.")
        elif image_data is None:
            st.error("📸 Please capture a photo using the camera input above.")
        else:
            with st.spinner("Extracting 512-D embedding via InsightFace..."):
                try:
                    # Convert uploaded camera bytes to OpenCV BGR image
                    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    rec_engine = FaceRecognitionEngine()
                    result = rec_engine.extract_embedding(frame)

                    if result is None:
                        st.error("❌ No face detected in the image. Please position your face clearly in the camera and try again.")
                    else:
                        embedding, bbox = result
                        # Strict passive liveness verification
                        liveness_engine = PassiveLivenessEngine()
                        is_real, live_score = liveness_engine.check_passive_liveness(frame, bbox)

                        if not is_real:
                            st.error(
                                f"❌ Biometric Enrollment Blocked: Anti-spoofing verification failed "
                                f"(Live score: {live_score:.2f}). Please present a live, physical face in good lighting."
                            )
                        else:
                            db = AttendanceDatabase()
                            db.enroll_employee(emp_id.strip(), emp_name.strip(), embedding)

                            st.toast(f"Embedding generated for {emp_id}!", icon="🧬")
                            st.success(f"✅ Successfully enrolled **{emp_name}** ({emp_id})! 512-D embedding saved to SQLite.")
                except Exception as e:
                    st.error(f"Enrollment failed: {e}")


def render_manage_personnel_view():
    inject_custom_css()

    st.markdown("""
        <div class="dashboard-header blue">
            <h2>Personnel Directory & Management</h2>
            <p>Inspect enrolled company personnel, verify biometric registration, and manage active records.</p>
        </div>
    """, unsafe_allow_html=True)

    db = AttendanceDatabase()
    emp_df = export_utils.fetch_enrolled_employees()
    total_enrolled = len(emp_df)

    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        st.metric(label="Total Enrolled Personnel", value=total_enrolled, border=True)
    with col_m2:
        search_query = st.text_input("🔍 Search Personnel", placeholder="Filter by Employee ID or Full Name...")

    st.divider()

    if emp_df.empty:
        st.info("No personnel currently enrolled in the database. Use the **Enroll New Employee** module to register personnel.")
    else:
        filtered_df = emp_df.copy()
        if search_query:
            q = search_query.lower()
            # regex=False: characters such as "(" or "*" are matched literally.
            filtered_df = filtered_df[
                filtered_df["employee_id"].astype(str).str.lower().str.contains(q, regex=False)
                | filtered_df["full_name"].astype(str).str.lower().str.contains(q, regex=False)
            ]

        col_table, col_del = st.columns([2, 1], gap="medium")

        with col_table:
            st.subheader("Active Roster")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "employee_id": "Employee ID",
                    "full_name": "Full Name",
                    "created_at": "Enrolled On (UTC)",
                },
            )

        with col_del:
            st.subheader("Deregister Employee")
            st.caption("Permanently remove an employee profile and facial embedding from the system.")

            emp_choices = [
                f"{row['employee_id']} — {row['full_name']}"
                for _, row in emp_df.iterrows()
            ]
            selected_emp_del = st.selectbox("Select Employee to Deregister", options=emp_choices)

            if selected_emp_del:
                del_id = selected_emp_del.split(" — ")[0]
                # One confirmation per employee: a tick never carries over to the
                # next person selected after a deletion.
                confirm_key = f"confirm_emp_del_{del_id}"
                confirm_del = st.checkbox(f"Confirm deletion of {del_id}", key=confirm_key)

                if st.button("🗑️ Delete Employee", type="primary", disabled=not confirm_del,
                             use_container_width=True, key="delete_employee"):
                    success = db.delete_employee(del_id)
                    st.session_state.pop(confirm_key, None)
                    if success:
                        st.toast(f"Deleted {del_id} from database!", icon="🗑️")
                        st.success(f"Employee {del_id} removed. Active roster will auto-sync with the kiosk.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {del_id}. Record may have already been removed.")


def render_export_view():
    inject_custom_css()

    st.markdown("""
        <div class="dashboard-header purple">
            <h2>Data Export Hub</h2>
            <p>Automate CSV data exports for payroll processing and system auditing.</p>
        </div>
    """, unsafe_allow_html=True)

    df = export_utils.fetch_attendance_logs()
    has_records = not df.empty

    col_filter, col_preview = st.columns([1, 2], gap="medium")

    with col_filter:
        st.subheader("Filter Parameters")

        # 1. Employee filter
        emp_options = ["All Personnel"]
        if has_records and "employee_id" in df.columns:
            unique_emps = df[["employee_id", "full_name"]].drop_duplicates()
            for _, r in unique_emps.iterrows():
                eid = str(r["employee_id"])
                fname = str(r.get("full_name", ""))
                label = f"{eid} — {fname}" if fname and fname != "Unknown / Not Enrolled" else eid
                emp_options.append(label)
        selected_emp = st.selectbox("Select Employee", options=emp_options, disabled=not has_records)

        # 2. Date Range filter: defaults to every recorded date, so an export
        #    never silently covers only today.
        today = datetime.now().date()
        default_range = (today, today)
        if has_records and "date" in df.columns:
            recorded = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce").dropna()
            if not recorded.empty:
                default_range = (recorded.min().date(), recorded.max().date())
        date_selection = st.date_input("Select Date Range", value=default_range, disabled=not has_records)

        # 3. Status filter (Payroll vs Audit)
        payroll_only = st.checkbox(
            "Payroll-ready only (exclude blocked attempts)",
            value=False,
            disabled=not has_records,
        )

        st.write("---")

        # Apply filtering
        filtered_export_df = df.copy() if has_records else pd.DataFrame()

        if has_records:
            # Employee filter
            if selected_emp != "All Personnel":
                target_id = selected_emp.split(" — ")[0]
                filtered_export_df = filtered_export_df[filtered_export_df["employee_id"] == target_id]

            # Date filter (handles single date or range tuple)
            if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
                start_date, end_date = date_selection
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                filtered_export_df = filtered_export_df[
                    (filtered_export_df["date"] >= start_str) & (filtered_export_df["date"] <= end_str)
                ]
            elif isinstance(date_selection, (list, tuple)) and len(date_selection) == 1:
                start_str = date_selection[0].strftime("%Y-%m-%d")
                filtered_export_df = filtered_export_df[filtered_export_df["date"] >= start_str]

            # Status filter
            if payroll_only:
                filtered_export_df = filtered_export_df[filtered_export_df["spoof_status"] == "Passed"]

        has_filtered_records = not filtered_export_df.empty

        st.caption(
            f"Export contains **{len(filtered_export_df)}** record(s) matching current filters."
        )
        st.caption(
            "Unless 'Payroll-ready only' is ticked, the download includes both passed and "
            "blocked log entries. It is a raw attendance-log export, not a calculated payroll report."
        )

        st.download_button(
            label="Download Attendance Log (CSV)",
            data=export_utils.convert_df_to_csv(filtered_export_df) if has_filtered_records else b"",
            file_name="attendance_log_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            disabled=not has_filtered_records,
        )

    with col_preview:
        st.subheader("Live Data Preview")
        if has_filtered_records:
            st.dataframe(filtered_export_df, use_container_width=True, hide_index=True)
        elif has_records:
            st.info("No records match the selected filter criteria.")
        else:
            st.info("No attendance records to export yet.")