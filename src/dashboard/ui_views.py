import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import export_utils
from src.database.db_manager import AttendanceDatabase
from src.recognition.matcher import FaceRecognitionEngine


def render_logs_view():
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.header("Live Attendance Logs")
        st.write("Monitor real-time employee clock-in and clock-out attempts.")
    with col_btn:
        st.write("")
        if st.button("🔄 Refresh Logs", use_container_width=True):
            st.rerun()

    # Fetch real data from SQLite
    df = export_utils.fetch_attendance_logs()

    # If the database or table isn't created yet, show a fallback preview
    if df.empty:
        st.info("Notice: `attendance.db` logs table is currently empty. Displaying layout preview:")
        df = pd.DataFrame(
            {
                "employee_id": ["DCCP-1001", "DCCP-1002", "DCCP-1003"],
                "date": ["2026-10-28", "2026-10-28", "2026-10-28"],
                "time_in": ["07:45 AM", "07:52 AM", "08:05 AM"],
                "time_out": ["--", "--", "--"],
                "spoof_status": ["Passed", "Passed", "Failed (2D Photo)"],
            }
        )

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_enrollment_view():
    st.header("Biometric Enrollment Module")
    st.write("Capture and store mathematical 512-dimensional facial embeddings for new personnel.")

    with st.form("enrollment_form"):
        col1, col2 = st.columns(2)
        with col1:
            emp_id = st.text_input("Employee ID (e.g. DCCP-1001)")
        with col2:
            emp_name = st.text_input("Full Name")

        st.write("Capture Face Data")
        image_data = st.camera_input("Take an enrollment photo")

        submit_btn = st.form_submit_button("Extract & Save Embedding")

        if submit_btn:
            if not emp_id or not emp_name:
                st.error("Please provide both an Employee ID and Full Name.")
            elif image_data is None:
                st.error("Please capture a photo using the camera input above.")
            else:
                try:
                    # Convert uploaded camera bytes to OpenCV BGR image
                    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    rec_engine = FaceRecognitionEngine()
                    result = rec_engine.extract_embedding(frame)

                    if result is None:
                        st.error("No face detected in the image. Please position your face clearly in the camera and try again.")
                    else:
                        embedding, _ = result
                        db = AttendanceDatabase()
                        db.enroll_employee(emp_id.strip(), emp_name.strip(), embedding)
                        st.success(f"Successfully enrolled {emp_name} ({emp_id})! 512-D embedding saved to SQLite.")
                except Exception as e:
                    st.error(f"Enrollment failed: {e}")


def render_export_view():
    st.header("Data Export")
    st.write("Automate CSV data exports for payroll processing.")

    st.date_input("Select Date Range for Export")

    df = export_utils.fetch_attendance_logs()

    # Fallback preview if SQLite database is empty
    if df.empty:
        df = pd.DataFrame(
            {
                "employee_id": ["DCCP-1001", "DCCP-1002"],
                "date": ["2026-10-28", "2026-10-28"],
                "time_in": ["07:45 AM", "07:52 AM"],
                "time_out": ["05:00 PM", "05:00 PM"],
                "spoof_status": ["Passed", "Passed"],
            }
        )

    csv_data = export_utils.convert_df_to_csv(df)
    st.download_button(
        label="Download CSV Payroll Report",
        data=csv_data,
        file_name="payroll_export.csv",
        mime="text/csv",
        type="primary",
    )
