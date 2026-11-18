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
        st.date_input("Select Date Range", disabled=True)
        st.caption("Date-range filtering is not implemented. Export includes all recorded dates.")

        st.write("---")
        st.caption(
            "The download includes both passed and blocked log entries. "
            "It is a raw attendance-log export, not a calculated payroll report."
        )
        st.download_button(
            label="Download Attendance Log (CSV)",
            data=export_utils.convert_df_to_csv(df) if has_records else b"",
            file_name="attendance_log_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            disabled=not has_records,
        )

    with col_preview:
        st.subheader("Live Data Preview")
        if has_records:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance records to export yet.")