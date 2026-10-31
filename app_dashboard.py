import streamlit as st
from src.dashboard import ui_views
from src.database.db_manager import AttendanceDatabase

st.set_page_config(
    page_title="HR Admin Dashboard | DCCP",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.sidebar.title("HR Admin Navigation")
    st.sidebar.write("DCCP Attendance Logging System")

    menu_options = {
        "Live Attendance Logs": ui_views.render_logs_view,
        "Enroll New Employee": ui_views.render_enrollment_view,
        "Export Payroll Data": ui_views.render_export_view,
    }

    selection = st.sidebar.radio("Go to:", list(menu_options.keys()))

    # Sidebar utility for testing and resetting
    st.sidebar.markdown("---")
    st.sidebar.subheader("Testing Controls")
    with st.sidebar.expander("Reset Database / Logs"):
        st.caption("Convenient shortcuts for capstone testing cycles:")
        if st.button("Clear Logs Only", use_container_width=True):
            db = AttendanceDatabase()
            db.reset_database(clear_employees=False, clear_logs=True)
            st.success("Attendance logs cleared!")
            st.rerun()

        if st.button("Reset Everything (Logs + Faces)", type="primary", use_container_width=True):
            db = AttendanceDatabase()
            db.reset_database(clear_employees=True, clear_logs=True)
            st.warning("Database completely reset!")
            st.rerun()

    # Execute selected view
    menu_options[selection]()


if __name__ == "__main__":
    main()
