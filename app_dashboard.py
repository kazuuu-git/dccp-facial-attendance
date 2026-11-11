import streamlit as st
from src.dashboard import ui_views
from src.database.db_manager import AttendanceDatabase

st.set_page_config(
    page_title="HR Admin Dashboard | DCCP",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.markdown("""
        <style>
        button[kind="primary"] {
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
            color: white !important;
        }
        button[kind="primary"]:hover, button[kind="primary"]:active, button[kind="primary"]:focus {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="padding-bottom: 10px;">
            <h2 style="margin-bottom: 0px; color: #3b82f6;">🏢 DCCP Console</h2>
            <p style="margin-top: 5px; font-size: 0.95rem; color: #9ca3af;">AI Facial Attendance System</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Map views to professional Google Material Icons instead of text emojis
    menu_options = {
        "Live Attendance Logs": {
            "func": ui_views.render_logs_view, 
            "icon": ":material/monitoring:"
        },
        "Enroll New Employee": {
            "func": ui_views.render_enrollment_view, 
            "icon": ":material/person_add:"
        },
        "Export Payroll Data": {
            "func": ui_views.render_export_view, 
            "icon": ":material/folder_zip:"
        },
    }

    if "current_view" not in st.session_state:
        st.session_state.current_view = "Live Attendance Logs"

    st.sidebar.markdown("<p style='color: #9ca3af; font-size: 0.9rem; margin-bottom: 5px;'>Navigation</p>", unsafe_allow_html=True)

    # 2. Apply the 'icon' parameter directly to the buttons
    for view_name, config in menu_options.items():
        is_active = (st.session_state.current_view == view_name)
        button_type = "primary" if is_active else "secondary"
        
        if st.sidebar.button(view_name, icon=config["icon"], type=button_type, use_container_width=True):
            st.session_state.current_view = view_name
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Testing Controls")
    
    with st.sidebar.expander("Reset Database / Logs"):
        st.caption("Convenient shortcuts for capstone testing cycles:")
        # You can add Material Icons to the testing buttons as well!
        if st.button("Clear Logs Only", icon=":material/delete_sweep:", use_container_width=True):
            db = AttendanceDatabase()
            db.reset_database(clear_employees=False, clear_logs=True)
            st.success("Attendance logs cleared!")
            st.rerun()

        if st.button("Reset Everything", type="primary", icon=":material/warning:", use_container_width=True):
            db = AttendanceDatabase()
            db.reset_database(clear_employees=True, clear_logs=True)
            st.warning("Database completely reset!")
            st.rerun()

    # 3. Execute the function stored inside the dictionary
    menu_options[st.session_state.current_view]["func"]()


if __name__ == "__main__":
    main()