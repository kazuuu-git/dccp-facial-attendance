import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import export_utils
from src.database.db_manager import AttendanceDatabase
from src.recognition.matcher import FaceRecognitionEngine

def inject_custom_css():
    """Injects custom CSS to style the Streamlit interface cleanly."""
    st.markdown("""
        <style>
        /* Custom Header Banners */
        .dashboard-header {
            background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
            padding: 18px 24px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 25px;
        }
        .dashboard-header.green { border-left-color: #10b981; }
        .dashboard-header.purple { border-left-color: #8b5cf6; }
        
        .dashboard-header h2 { margin: 0; color: #ffffff; font-size: 1.5rem; font-weight: 600; }
        .dashboard-header p { margin: 4px 0 0 0; color: #9ca3af; font-size: 0.9rem; }
        
        /* Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: #1e222d;
            border: 1px solid #2e3440;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

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

    # Fallback Data if empty
    if df.empty:
        st.info("Notice: `attendance.db` logs table is currently empty. Displaying layout preview:")
        df = pd.DataFrame({
            "employee_id": ["DCCP-1001", "DCCP-1002", "DCCP-1003", "DCCP-1004"],
            "date": ["2026-10-28", "2026-10-28", "2026-10-28", "2026-10-28"],
            "time_in": ["07:45 AM", "07:52 AM", "08:05 AM", "08:12 AM"],
            "time_out": ["--", "--", "--", "--"],
            "spoof_status": ["Passed", "Passed", "Passed", "Failed (2D Photo)"],
        })

    # Summary Metrics Row
    total_logs = len(df)
    passed_logs = len(df[df["spoof_status"] == "Passed"]) if "spoof_status" in df.columns else 0
    failed_logs = total_logs - passed_logs
    
    met1, met2, met3 = st.columns(3)
    with met1:
        st.metric(label="Total Scans", value=total_logs)
    with met2:
        st.metric(label="Live Faces Verified", value=passed_logs, delta="Secure")
    with met3:
        st.metric(label="Spoof Attacks Blocked", value=failed_logs, delta=f"-{failed_logs} Alert" if failed_logs > 0 else "0", delta_color="inverse")

    st.divider()

    # Interactive Filter
    status_filter = st.selectbox("Filter Display by Verification Status", ["All", "Passed", "Failed (2D Photo)", "Failed (3D Mask/Screen)"])
    display_df = df if status_filter == "All" else df[df["spoof_status"] == status_filter]

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
                        embedding, _ = result
                        db = AttendanceDatabase()
                        db.enroll_employee(emp_id.strip(), emp_name.strip(), embedding)
                        
                        st.toast(f"Embedding generated for {emp_id}!", icon="🧬")
                        st.success(f"✅ Successfully enrolled **{emp_name}** ({emp_id})! 512-D embedding saved to SQLite.")
                        st.balloons()
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

    if df.empty:
        df = pd.DataFrame({
            "employee_id": ["DCCP-1001", "DCCP-1002"],
            "date": ["2026-10-28", "2026-10-28"],
            "time_in": ["07:45 AM", "07:52 AM"],
            "time_out": ["05:00 PM", "05:00 PM"],
            "spoof_status": ["Passed", "Passed"],
        })

    col_filter, col_preview = st.columns([1, 2], gap="medium")
    
    with col_filter:
        st.subheader("Filter Parameters")
        st.date_input("Select Date Range")
        
        st.write("---")
        csv_data = export_utils.convert_df_to_csv(df)
        st.download_button(
            label="Download CSV Payroll Report",
            data=csv_data,
            file_name="payroll_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

    with col_preview:
        st.subheader("Live Data Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)