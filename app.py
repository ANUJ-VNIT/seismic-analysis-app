import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from streamlit_lottie import st_lottie
import json
import io
import matplotlib.pyplot as plt



def fig_to_png_bytes(x, y, title, xlabel, ylabel):
    buf = io.BytesIO()
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

from PIL import Image

from pathlib import Path

from solver.central_difference_RSL import cd_response_spectrum_solver
from solver.central_difference_THL import central_difference_solver
from solver.newmark_method_THL import newmark_solver
from solver.newmark_method_RSL import newmark_response_spectrum_solver
from solver.Interpolation_Excitation_THL import interpolation_excitation_solver
from solver.Interpolation_Excitation_RSL import interpolation_response_spectrum_solver
from solver.KR_aplha_THL import kr_alpha_linear_solver
from solver.KR_alpha_RSL import kr_alpha_response_spectrum_solver
from solver.EPP_CDM_THL import epp_time_history_solver
from solver.EPP_Newmark_THL import epp_newmark_solver
from solver.EPP_KR_THL import epp_kr_alpha_solver

# === PAGE SETUP ===
st.set_page_config(layout="wide", page_title="Dynamic Analysis")

# === SESSION STATE ===
if "started" not in st.session_state:
    st.session_state.started = False
if "page" not in st.session_state:
    st.session_state.page = "home"

# === LOAD LOTTIE ===


def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_eq = load_lottie_file("assets/loading_animation.json")

# === CUSTOM CSS ===
st.markdown("""
<style>
:root {
    --bg-color: #ffffff;
    --text-color: #222222;
    --header-bg: #0a2c58;
    --button-bg: #0a2c58;
    --button-text: white;
    --button-hover-bg: white;
    --button-hover-text: #0a2c58;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-color: #0f1117;
        --text-color: #f0f0f0;
        --header-bg: #1a1f2e;
        --button-bg: #1a1f2e;
        --button-text: white;
        --button-hover-bg: white;
        --button-hover-text: #1a1f2e;
    }
}

html, body, [data-testid="stApp"] {
    height: 100%;
    margin: 0;
    background: linear-gradient(to bottom, var(--bg-color), var(--bg-color));
    background-attachment: fixed;
    color: var(--text-color);
}

/* Essential layout padding restored */
.block-container {
    padding: 2rem 7rem 14rem 7rem !important;
    background-color: transparent !important;
}

/* Remove default shadows & boxes */
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
.element-container,
[data-testid="column"] {
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Header */
.custom-header {
    width: 100%;
    background-color: var(--header-bg);
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: center;
    font-family: 'Segoe UI', sans-serif;
    color: var(--button-text);
}

/* Nav buttons */
.nav-buttons {
    display: flex;
    gap: 2rem;
}
.nav-buttons button,
.start-analysis-btn {
    background-color: var(--button-bg);
    border: 2px solid var(--button-bg);
    color: var(--button-text);
    padding: 0.5rem 1.5rem;
    font-size: 1rem;
    font-weight: 500;
    border-radius: 8px;
    cursor: pointer;
    transition: 0.3s ease-in-out;
}
.nav-buttons button:hover,
.start-analysis-btn:hover {
    background-color: var(--button-hover-bg);
    color: var(--button-hover-text);
}

/* Title */
h1 {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-top: 2.5rem;
    color: var(--text-color);
}

/* Center section */
.center-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 3rem;
    width: 100%;
}

/* Center button container */
.button-wrapper {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# === HEADER NAVIGATION BAR ===
st.markdown("""
<div class="custom-header">
    <form method="get">
        <div class="nav-buttons">
            <button name="nav" value="home" type="submit">Home</button>
            <button name="nav" value="analyze" type="submit">Analyze</button>
            <button name="nav" value="about" type="submit">About</button>
            <button name="nav" value="help" type="submit">Help</button>
        </div>
    </form>
</div>
""", unsafe_allow_html=True)

# === HANDLE NAVIGATION ===
nav = st.query_params.get("nav")
if nav:
    st.session_state.page = nav
    if nav == "analyze":
        st.session_state.started = True
    elif nav == "home":
        st.session_state.started = False

# === ROUTING ===
if st.session_state.page == "home":
    st.markdown("<h1>Dynamic Response Analyzer</h1>", unsafe_allow_html=True)
    st.markdown('<div class="center-section">', unsafe_allow_html=True)
    st_lottie(lottie_eq, height=300, key="intro_anim")

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.markdown("""
            <form method="get">
                <button name="nav" value="analyze" type="submit" class="start-analysis-btn">
                    Start Dynamic Analysis
                </button>
            </form>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

elif st.session_state.page == "about":
    st.markdown("<h1>About the App</h1>", unsafe_allow_html=True)
    st.markdown("""
    ###  Dynamic Response Analyzer

    This educational tool enables the simulation of seismic responses of **Single Degree of Freedom (SDOF)** systems under ground motion.  
    The app supports both linear and nonlinear structural behavior.

    It implements classical time integration methods for dynamic analysis:

    -  Central Difference Method      - -  Newmark's Method     - -  KR-alpha Method

    **Key Features:**
    - Time history analysis     - - Response spectrum       - - Upload custom ground motion files
    - Predefined ground motion datasets    - - Residual deformation & ductility demand estimation   - - Clean and interactive UI for better interpretation

    **Built With:** Python • Streamlit • NumPy • Matplotlib  
    **Intended Users:** Civil engineering students, researchers, earthquake engineers, and educators.

    ---
    ## 👨‍💻 About the Developer

    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        # Replace with your actual image path
        image = Image.open("assets/anuj_photo.jpg")
        st.image(image, width=280, caption="Anuj Sharma", output_format="auto")
    with col2:
        st.markdown("""
        ### Anuj Sharma  
        **Final Year B.Tech Student, Civil Engineering**  
        Passionate about **structural dynamics**, **seismic analysis**, and software development in engineering tools.

        - 🌍 **Institution:** Visvesvaraya National Institute of Technology Nagpur
        - 📚 SURGE Intern @ IIT Kanpur         
        - 💻 Python • Streamlit

        📧 **Email:** anuj2708sharma@gmail.com  
        🌐 **LinkedIn:** www.linkedin.com/in/anuj-sharma2708
        """)

    st.markdown("<h2>Credits & Acknowledgements</h2>", unsafe_allow_html=True)
    st.markdown("""
    I express my sincere gratitude to:

    - **Dr. Chinmoy Kolay** – *Assistant Professor, IIT Kanpur*  
    - **Mr. Hironmoy Kakoti** – *PhD Scholar,IIT Kanpur*  

    for their guidance, mentorship, and support in developing this app. Their expertise in structural dynamics and seismic analysis has been invaluable in shaping this tool.

    ---
    """)


    st.stop()

elif st.session_state.page == "help":
    st.markdown("<h1>Help</h1>", unsafe_allow_html=True)

    # ✅ Add your YouTube video link here
    st.video("https://youtu.be/HPiTfnTjvnM")

    st.markdown("""
    ### Need Assistance?

    - Navigate to **Analyze** to begin seismic analysis using various numerical methods.
    - Use the **Home** page to restart or return to the intro screen.
    - The **About** section gives details about this app's purpose.
    
    **Troubleshooting:**
    - If uploads fail, ensure the file format is `.txt` or `.csv`.
    - Use clear ground motion data in proper format (time, acceleration).
    - Acceleration data should be in `g` units, which will be converted to `m/s²`.                           
    - For bugs, please contact the developer.

    **Contact:** anuj2708sharma@gmail.com
    """)
    st.stop()


# --- MAIN APP ---
if st.session_state.started:

    def load_raw_ground_motion():
        st.subheader("Select Ground Motion")
        motion_choice = selected_option = st.selectbox(
            "Choose a ground motion file or upload your own:",
            ["-- Select --", "Upload your own", "El Centro", "Beverli Hill 009", "Beverli Hill 279", "Delta 262",
             "Delta 352", "Hollywood 90", "Hollywood 180", "Poe Road 270", "Poe Road 360", "Tolmezzo 000", "Tolmezzo 270"],
            index=2  # "El Centro" is at index 2
        )

        st.markdown(
            "**Note:** File must contain two columns: `time (s)` and `acceleration (g)`.")

        uploaded_file = None
        time = accel = None

        if motion_choice == "Upload your own":
            uploaded_file = st.file_uploader(
                "Upload Ground Motion File", type=["txt", "csv"])

        if motion_choice != "-- Select --":
            try:
                if motion_choice == "Upload your own":
                    if uploaded_file is not None:
                        data = pd.read_csv(
                            uploaded_file, sep=r'\s+', header=None, names=["time", "accel"])

                    else:
                        st.warning("Please upload a file.")
                        return None, None
                else:
                    file_map = {
                        "El Centro": "ElCentro.txt",
                        "Beverli Hill 009": "Beverli_Hill_009.txt",
                        "Beverli Hill 279": "Beverli_Hill_279.txt",
                        "Delta 262": "Delta_262.txt",
                        "Delta 352": "Delta_352.txt",
                        "Hollywood 90": "Hollywood_090.txt",
                        "Hollywood 180": "Hollywood_180.txt",
                        "Poe Road 270": "Poe Road_270.txt",
                        "Poe Road 360": "Poe Road_360.txt",
                        "Tolmezzo 000": "Tolmezzo_000.txt",
                        "Tolmezzo 270": "Tolmezzo_270.txt"
                    }
                    filepath = os.path.join("GM_data", file_map[motion_choice])
                    data = pd.read_csv(
                        filepath, sep=r'\s+', header=None, names=["time", "accel"])

                time = data['time'].to_numpy()
                accel = data['accel'].to_numpy() * 9.81
                st.success("Ground motion data loaded.")

            except Exception as e:
                st.error(f"Failed to load file: {e}")

        return time, accel

    analysis_type = st.selectbox("Choose Analysis Type:", [
                                 "-- Select --", "Linear", "Non-Linear"])

    if analysis_type == "Linear":
        lin_type = st.selectbox("Select Response Type:", [
                                "-- Select --", "Time History", "Response Spectrum"])

        if lin_type == "Time History":
            st.success("You selected Time History method.")
            time_history_method = st.selectbox("Choose Numerical Method:", [
                                               "-- Select --", "Interpolation of Excitation", "K R-Alpha Method ", "Central Difference", "Newmark's Method"])

            if time_history_method == "Central Difference":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")

                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Central Difference Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            u, v, a, t = central_difference_solver(
                                m, ζ, Tn, accel_new, time_new)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")

                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=t, y=u, mode='lines', name='Displacement'))
                        fig_u.update_layout(
                            title='Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Displacement (m)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)

                        fig_v = go.Figure()
                        fig_v.add_trace(go.Scatter(
                            x=t, y=v, mode='lines', name='Velocity', line=dict(color='orange')))
                        fig_v.update_layout(
                            title='Velocity vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Velocity (m/s)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_v, use_container_width=True)

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(
                            x=t, y=a, mode='lines', name='Acceleration', line=dict(color='green')))
                        fig_a.update_layout(
                            title='Acceleration vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Acceleration (m/s²)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": t,
                            "Displacement (m)": u,
                            "Velocity (m/s)": v,
                            "Acceleration (m/s²)": a
                        })
                        csv = results.to_csv(index=False).encode('utf-8')

                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(t, u, "Displacement vs Time", "Time (s)", "Displacement (m)")
                        buffer_v = fig_to_png_bytes(t, v, "Velocity vs Time", "Time (s)", "Velocity (m/s)")
                        buffer_a = fig_to_png_bytes(t, a, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)")

                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Velocity Plot (PNG)",
                                data=buffer_v,
                                file_name="velocity_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Acceleration Plot (PNG)",
                                data=buffer_a,
                                file_name="acceleration_plot.png",
                                mime="image/png"
                            )

            elif time_history_method == "Newmark's Method":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                method_type = st.selectbox("Select Newmark Method:", [
                                           "Average Acceleration", "Linear Acceleration"])

                # Set default gamma and beta based on selection
                if method_type == "Average Acceleration":
                    gamma_default = 0.5
                    beta_default = 1/4
                elif method_type == "Linear Acceleration":
                    gamma_default = 0.5
                    beta_default = 1/6

                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)
                gamma = st.number_input(
                    "Gamma", value=gamma_default, format="%.4f")
                beta = st.number_input(
                    "Beta", value=beta_default, format="%.4f")

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Newmark's Method Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            u, v, a, t = newmark_solver(
                                m, ζ, Tn, accel_new, time_new, gamma, beta)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")

                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=t, y=u, mode='lines', name='Displacement'))
                        fig_u.update_layout(
                            title='Displacement vs Time', xaxis_title='Time (s)', yaxis_title='Displacement (m)')
                        st.plotly_chart(fig_u, use_container_width=True)

                        fig_v = go.Figure()
                        fig_v.add_trace(go.Scatter(
                            x=t, y=v, mode='lines', name='Velocity', line=dict(color='orange')))
                        fig_v.update_layout(
                            title='Velocity vs Time', xaxis_title='Time (s)', yaxis_title='Velocity (m/s)')
                        st.plotly_chart(fig_v, use_container_width=True)

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(
                            x=t, y=a, mode='lines', name='Acceleration', line=dict(color='green')))
                        fig_a.update_layout(
                            title='Acceleration vs Time', xaxis_title='Time (s)', yaxis_title='Acceleration (m/s²)')
                        st.plotly_chart(fig_a, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": t,
                            "Displacement (m)": u,
                            "Velocity (m/s)": v,
                            "Acceleration (m/s²)": a
                        })
                        csv = results.to_csv(index=False).encode('utf-8')

                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(t, u, "Displacement vs Time", "Time (s)", "Displacement (m)")   
                        buffer_v = fig_to_png_bytes(t, v, "Velocity vs Time", "Time (s)", "Velocity (m/s)")
                        buffer_a = fig_to_png_bytes(t, a, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)")

                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Velocity Plot (PNG)",
                                data=buffer_v,
                                file_name="velocity_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Acceleration Plot (PNG)",
                                data=buffer_a,
                                file_name="acceleration_plot.png",
                                mime="image/png"
                            )
            elif time_history_method == "Interpolation of Excitation":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Interpolation of Excitation Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            u, v, t = interpolation_excitation_solver(
                                m, ζ, Tn, accel_new, time_new)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")
                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=t, y=u, mode='lines', name='Displacement'))
                        fig_u.update_layout(
                            title='Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Displacement (m)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)
                        fig_v = go.Figure()
                        fig_v.add_trace(go.Scatter(
                            x=t, y=v, mode='lines', name='Velocity', line=dict(color='orange')))
                        fig_v.update_layout(
                            title='Velocity vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Velocity (m/s)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_v, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": t,
                            "Displacement (m)": u,
                            "Velocity (m/s)": v
                        })
                        csv = results.to_csv(index=False).encode('utf-8')
                        # Create PNGs for both plots
                        buffer_u = fig_to_png_bytes(t, u, "Displacement vs Time", "Time (s)", "Displacement (m)")
                        buffer_v = fig_to_png_bytes(t, v, "Velocity vs Time", "Time (s)", "Velocity (m/s)")



                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Velocity Plot (PNG)",
                                data=buffer_v,
                                file_name="velocity_plot.png",
                                mime="image/png"
                            )
            elif time_history_method == "K R-Alpha Method ":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)
                rho = st.number_input("Rho (default 1)", value=1.0)

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run K R-Alpha Method Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            u, v, a, t = kr_alpha_linear_solver(
                                m, ζ, Tn, accel_new, time_new, rho)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")

                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=t, y=u, mode='lines', name='Displacement'))
                        fig_u.update_layout(
                            title='Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Displacement (m)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)

                        fig_v = go.Figure()
                        fig_v.add_trace(go.Scatter(
                            x=t, y=v, mode='lines', name='Velocity', line=dict(color='orange')))
                        fig_v.update_layout(
                            title='Velocity vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Velocity (m/s)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_v, use_container_width=True)

                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(
                            x=t, y=a, mode='lines', name='Acceleration', line=dict(color='green')))
                        fig_a.update_layout(
                            title='Acceleration vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Acceleration (m/s²)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": t,
                            "Displacement (m)": u,
                            "Velocity (m/s)": v,
                            "Acceleration (m/s²)": a
                        })
                        csv = results.to_csv(index=False).encode('utf-8')
                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(t, u, "Displacement vs Time", "Time (s)", "Displacement (m)")
                        buffer_v = fig_to_png_bytes(t, v, "Velocity vs Time", "Time (s)", "Velocity (m/s)")
                        buffer_a = fig_to_png_bytes(t, a, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)")
                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Velocity Plot (PNG)",
                                data=buffer_v,
                                file_name="velocity_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Acceleration Plot (PNG)",
                                data=buffer_a,
                                file_name="acceleration_plot.png",
                                mime="image/png"
                            )

        elif lin_type == "Response Spectrum":
            st.success("You selected Response Spectrum method.")
            Response_Spectrum_method = st.selectbox("Choose Numerical Method:", [
                                                    "-- Select --", "Interpolation of Excitation", "K R-Alpha Method", "Central Difference", "Newmark's Method"])

            if Response_Spectrum_method == "Central Difference":

                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                ζ = st.number_input("Damping Ratio (default 2%)", value=0.02)
                time, accel = load_raw_ground_motion()

                if time is not None and accel is not None:
                    dt = 0.001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Response Spectrum Simulation"):
                        with st.spinner("Running simulation..."):
                            st_lottie(lottie_eq, speed=1,
                                      height=300, loop=True)
                            Tn_values, max_disp = cd_response_spectrum_solver(
                                ζ, accel_new, time_new)

                        st.success("Simulation completed!")

                        fig_rs = go.Figure()
                        fig_rs.add_trace(go.Scatter(
                            x=Tn_values,
                            y=max_disp,
                            mode='lines',
                            name='Displacement Response Spectrum',
                            line=dict(shape='spline',
                                      color='lightblue', width=3)
                        ))
                        fig_rs.update_layout(
                            title='Displacement Response Spectrum',
                            xaxis_title='Natural Period (s)',
                            yaxis_title='Max Displacement (m)',
                            yaxis=dict(range=[0, max(max_disp) * 1.1]),
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_rs, use_container_width=True)
                        # --- DOWNLOAD SECTION FOR RESPONSE SPECTRUM ---
                        spectrum_df = pd.DataFrame({
                            "Natural Period (s)": Tn_values,
                            "Max Displacement (m)": max_disp
                        })
                        csv_spectrum = spectrum_df.to_csv(
                            index=False).encode('utf-8')

                        # Export response spectrum plot to PNG
                        buffer_spectrum = fig_to_png_bytes(
                            Tn_values, max_disp,
                            "Displacement Response Spectrum",
                            "Natural Period (s)", "Max Displacement (m)"
                        )


                        # Download buttons
                        with st.expander("📥 Download Response Spectrum Outputs"):
                            st.download_button(
                                label="📄 Download Spectrum Data as CSV",
                                data=csv_spectrum,
                                file_name="response_spectrum.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📊 Download Spectrum Plot (PNG)",
                                data=buffer_spectrum,
                                file_name="response_spectrum_plot.png",
                                mime="image/png"
                            )

            elif Response_Spectrum_method == "Newmark's Method":

                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                ζ = st.number_input("Damping Ratio (default 2%)", value=0.02)
                method_type = st.selectbox("Select Newmark Method:", [
                                           "Average Acceleration", "Linear Acceleration"])
                # Set default gamma and beta based on selection
                if method_type == "Average Acceleration":
                    gamma_default = 0.5
                    beta_default = 1/4
                elif method_type == "Linear Acceleration":
                    gamma_default = 0.5
                    beta_default = 1/6

                gamma = st.number_input(
                    "Gamma", value=gamma_default, format="%.4f")
                beta = st.number_input(
                    "Beta", value=beta_default, format="%.4f")

                time, accel = load_raw_ground_motion()

                if time is not None and accel is not None:
                    dt = 0.001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Response Spectrum Simulation"):
                        with st.spinner("Running simulation..."):
                            st_lottie(lottie_eq, speed=1,
                                      height=300, loop=True)

                            Tn_values, max_disp = newmark_response_spectrum_solver(
                                ζ, accel_new, time_new, gamma, beta)

                        st.success("Simulation completed!")

                        fig_rs = go.Figure()
                        fig_rs.add_trace(go.Scatter(
                            x=Tn_values,
                            y=max_disp,
                            mode='lines',
                            name='Displacement Response Spectrum',
                            line=dict(shape='spline',
                                      color='lightblue', width=3)
                        ))
                        fig_rs.update_layout(
                            title='Displacement Response Spectrum',
                            xaxis_title='Natural Period (s)',
                            yaxis_title='Max Displacement (m)',
                            yaxis=dict(range=[0, max(max_disp) * 1.1]),
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_rs, use_container_width=True)
                        # --- DOWNLOAD SECTION FOR RESPONSE SPECTRUM ---
                        spectrum_df = pd.DataFrame({
                            "Natural Period (s)": Tn_values,
                            "Max Displacement (m)": max_disp
                        })
                        csv_spectrum = spectrum_df.to_csv(
                            index=False).encode('utf-8')
                        # Export response spectrum plot to PNG
                        buffer_spectrum = fig_to_png_bytes(
                            Tn_values, max_disp,
                            "Displacement Response Spectrum",
                            "Natural Period (s)", "Max Displacement (m)"
                        )


                        # Download buttons
                        with st.expander("📥 Download Response Spectrum Outputs"):
                            st.download_button(
                                label="📄 Download Spectrum Data as CSV",
                                data=csv_spectrum,
                                file_name="response_spectrum.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📊 Download Spectrum Plot (PNG)",
                                data=buffer_spectrum,
                                file_name="response_spectrum_plot.png",
                                mime="image/png"
                            )

            elif Response_Spectrum_method == "Interpolation of Excitation":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                ζ = st.number_input("Damping Ratio (default 2%)", value=0.02)
                time, accel = load_raw_ground_motion()

                if time is not None and accel is not None:
                    dt = 0.001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Response Spectrum Simulation"):
                        with st.spinner("Running simulation..."):
                            st_lottie(lottie_eq, speed=1,
                                      height=300, loop=True)
                            Tn_values, max_disp = interpolation_response_spectrum_solver(
                                ζ, accel_new, time_new)

                        st.success("Simulation completed!")

                        fig_rs = go.Figure()
                        fig_rs.add_trace(go.Scatter(
                            x=Tn_values,
                            y=max_disp,
                            mode='lines',
                            name='Displacement Response Spectrum',
                            line=dict(shape='spline',
                                      color='lightblue', width=3)
                        ))
                        fig_rs.update_layout(
                            title='Displacement Response Spectrum',
                            xaxis_title='Natural Period (s)',
                            yaxis_title='Max Displacement (m)',
                            yaxis=dict(range=[0, max(max_disp) * 1.1]),
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_rs, use_container_width=True)
                        # --- DOWNLOAD SECTION FOR RESPONSE SPECTRUM ---
                        spectrum_df = pd.DataFrame({
                            "Natural Period (s)": Tn_values,
                            "Max Displacement (m)": max_disp
                        })
                        csv_spectrum = spectrum_df.to_csv(
                            index=False).encode('utf-8')
                        # Export response spectrum plot to PNG
                        buffer_spectrum = fig_to_png_bytes(
                            Tn_values, max_disp,
                            "Displacement Response Spectrum",
                            "Natural Period (s)", "Max Displacement (m)"
                        )

                        # Download buttons
                        with st.expander("📥 Download Response Spectrum Outputs"):
                            st.download_button(
                                label="📄 Download Spectrum Data as CSV",
                                data=csv_spectrum,
                                file_name="response_spectrum.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📊 Download Spectrum Plot (PNG)",
                                data=buffer_spectrum,
                                file_name="response_spectrum_plot.png",
                                mime="image/png"
                            )

            elif Response_Spectrum_method == "K R-Alpha Method":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                ζ = st.number_input("Damping Ratio (default 2%)", value=0.02)
                rho = st.number_input("Rho (default 1)", value=1.0)

                time, accel = load_raw_ground_motion()

                if time is not None and accel is not None:
                    dt = 0.001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Response Spectrum Simulation"):
                        with st.spinner("Running simulation..."):
                            st_lottie(lottie_eq, speed=1,
                                      height=300, loop=True)
                            Tn_values, max_disp = kr_alpha_response_spectrum_solver(
                                ζ, accel_new, time_new, rho)

                        st.success("Simulation completed!")

                        fig_rs = go.Figure()
                        fig_rs.add_trace(go.Scatter(
                            x=Tn_values,
                            y=max_disp,
                            mode='lines',
                            name='Displacement Response Spectrum',
                            line=dict(shape='spline',
                                      color='lightblue', width=3)
                        ))
                        fig_rs.update_layout(
                            title='Displacement Response Spectrum',
                            xaxis_title='Natural Period (s)',
                            yaxis_title='Max Displacement (m)',
                            yaxis=dict(range=[0, max(max_disp) * 1.1]),
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_rs, use_container_width=True)
                        # --- DOWNLOAD SECTION FOR RESPONSE SPECTRUM ---
                        spectrum_df = pd.DataFrame({
                            "Natural Period (s)": Tn_values,
                            "Max Displacement (m)": max_disp
                        })
                        csv_spectrum = spectrum_df.to_csv(
                            index=False).encode('utf-8')
                        # Export response spectrum plot to PNG
                        buffer_spectrum = fig_to_png_bytes(
                            Tn_values, max_disp,
                            "Displacement Response Spectrum",
                            "Natural Period (s)", "Max Displacement (m)"
                        )

                        # Download buttons
                        with st.expander("📥 Download Response Spectrum Outputs"):
                            st.download_button(
                                label="📄 Download Spectrum Data as CSV",
                                data=csv_spectrum,
                                file_name="response_spectrum.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📊 Download Spectrum Plot (PNG)",
                                data=buffer_spectrum,
                                file_name="response_spectrum_plot.png",
                                mime="image/png"
                            )

    elif analysis_type == "Non-Linear":
        lin_type = st.selectbox("Select Response Type:", [
                                "-- Select --", "Time History and Ductility Demand"])
        if lin_type == "Time History and Ductility Demand":
            st.success("You selected Time History and Ductility Demand")
            time_history_method = st.selectbox("Choose Numerical Method:", [
                                               "-- Select --", "Newmark-beta Method"])

            if time_history_method == "Central Difference":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)
                Ry = st.number_input("Response Modification Factor")

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Time History Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            normalized_u_epp, normalized_f_s, time, ductility_demand, normalized_residual_deformation = epp_time_history_solver(
                                m, ζ, Tn, Ry, accel_new, time_new)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")
                        # Display ductility demand and residual deformation
                        st.markdown(
                            f"**Ductility Demand:** {ductility_demand:.2f}")
                        st.markdown(
                            f"**Normalized Residual Deformation:** {normalized_residual_deformation:.2f}")
                        # Plotting results
                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=time, y=normalized_u_epp, mode='lines', name='Normalized Displacement'))
                        fig_u.update_layout(
                            title='Normalized Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Displacement (u/uy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(
                            x=time, y=normalized_f_s, mode='lines', name='Normalized Restoring Force'))
                        fig_f.update_layout(
                            title='Normalized Restoring Force vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_f, use_container_width=True)
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(
                            x=normalized_u_epp, y=normalized_f_s, mode='lines', name='Normalized Displacement vs Normalized Restoring Force'))
                        fig_z.update_layout(
                            title='Normalized Displacement vs Normalized Restoring Force',
                            xaxis_title='Normalized Displacement (u/uy)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_z, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": time,
                            "Normalized Displacement (u/uy)": normalized_u_epp,
                            "Normalized Restoring Force (f_s/Fy)": normalized_f_s
                        })
                        csv = results.to_csv(index=False).encode('utf-8')
                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(
                            time, normalized_u_epp,
                            "Normalized Displacement vs Time",
                            "Time (s)", "Normalized Displacement (u/uy)"
                        )

                        buffer_f = fig_to_png_bytes(
                            time, normalized_f_s,
                            "Normalized Restoring Force vs Time",
                            "Time (s)", "Normalized Restoring Force (f_s/Fy)"
                        )

                        buffer_z = fig_to_png_bytes(
                            normalized_u_epp, normalized_f_s,
                            "Normalized Displacement vs Normalized Restoring Force",
                            "Normalized Displacement (u/uy)", "Normalized Restoring Force (f_s/Fy)"
                        )

                        #Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Normalized Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="normalized_displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Normalized Restoring Force Plot (PNG)",
                                data=buffer_f,
                                file_name="normalized_restoring_force_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Displacement vs Restoring Force Plot (PNG)",
                                data=buffer_z,
                                file_name="displacement_vs_restoring_force_plot.png",
                                mime="image/png"
                            )

            elif time_history_method == "Newmark-beta Method":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)
                Ry = st.number_input("Response Modification Factor")

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Time History Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            normalized_u_epp, normalized_f_s, time, ductility_demand, normalized_residual_deformation = epp_newmark_solver(
                                m, ζ, Tn, Ry, accel_new, time_new)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")
                        # Display ductility demand and residual deformation
                        st.markdown(
                            f"**Ductility Demand:** {ductility_demand:.2f}")
                        st.markdown(
                            f"**Normalized Residual Deformation:** {normalized_residual_deformation:.2f}")
                        # Plotting results
                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=time, y=normalized_u_epp, mode='lines', name='Normalized Displacement'))
                        fig_u.update_layout(
                            title='Normalized Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Displacement (u/uy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(
                            x=time, y=normalized_f_s, mode='lines', name='Normalized Restoring Force'))
                        fig_f.update_layout(
                            title='Normalized Restoring Force vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_f, use_container_width=True)
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(
                            x=normalized_u_epp, y=normalized_f_s, mode='lines', name='Normalized Displacement vs Normalized Restoring Force'))
                        fig_z.update_layout(
                            title='Normalized Displacement vs Normalized Restoring Force',
                            xaxis_title='Normalized Displacement (u/uy)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_z, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": time,
                            "Normalized Displacement (u/uy)": normalized_u_epp,
                            "Normalized Restoring Force (f_s/Fy)": normalized_f_s
                        })
                        csv = results.to_csv(index=False).encode('utf-8')
                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(
                            time, normalized_u_epp,
                            "Normalized Displacement vs Time",
                            "Time (s)", "Normalized Displacement (u/uy)"
                        )
                        buffer_f = fig_to_png_bytes(
                            time, normalized_f_s,
                            "Normalized Restoring Force vs Time",
                            "Time (s)", "Normalized Restoring Force (f_s/Fy)"
                        )
                        buffer_z = fig_to_png_bytes(
                            normalized_u_epp, normalized_f_s,
                            "Normalized Displacement vs Normalized Restoring Force",
                            "Normalized Displacement (u/uy)", "Normalized Restoring Force (f_s/Fy)"
                        )
                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Normalized Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="normalized_displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Normalized Restoring Force Plot (PNG)",
                                data=buffer_f,
                                file_name="normalized_restoring_force_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Displacement vs Restoring Force Plot (PNG)",
                                data=buffer_z,
                                file_name="displacement_vs_restoring_force_plot.png",
                                mime="image/png"
                            )

            elif time_history_method == "K R-Alpha Method":
                st.subheader(
                    "Provide System Parameters and Upload Ground Acceleration File")
                m = st.number_input("Mass (kg)", value=1)
                ζ = st.number_input("Damping Ratio (0-1)", value=0.05)
                Tn = st.number_input("Natural Period (s)", value=1.00)
                Ry = st.number_input("Response Modification Factor")
                rho = st.number_input("Rho (default 1)", value=1.0)

                time, accel = load_raw_ground_motion()
                if time is not None and accel is not None:
                    dt = 0.0001
                    time_new = np.arange(time[0], time[-1], dt)
                    interpolator = interp1d(time, accel, kind='linear')
                    accel_new = interpolator(time_new)

                    if st.button("Run Time History Simulation"):
                        with st.spinner("Running simulation..."):
                            lottie_placeholder = st.empty()
                            lottie_placeholder_lottie = st_lottie(
                                lottie_eq, speed=1, height=300, loop=True, key="loading_anim")
                            normalized_u_epp, normalized_f_s, time, ductility_demand, normalized_residual_deformation = epp_kr_alpha_solver(
                                m, ζ, Tn, Ry, accel_new, time_new, rho)
                        lottie_placeholder.empty()
                        st.success("Simulation completed!")
                        # Display ductility demand and residual deformation
                        st.markdown(
                            f"**Ductility Demand:** {ductility_demand:.2f}")
                        st.markdown(
                            f"**Normalized Residual Deformation:** {normalized_residual_deformation:.2f}")
                        # Plotting results
                        fig_u = go.Figure()
                        fig_u.add_trace(go.Scatter(
                            x=time, y=normalized_u_epp, mode='lines', name='Normalized Displacement'))
                        fig_u.update_layout(
                            title='Normalized Displacement vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Displacement (u/uy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_u, use_container_width=True)
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(
                            x=time, y=normalized_f_s, mode='lines', name='Normalized Restoring Force'))
                        fig_f.update_layout(
                            title='Normalized Restoring Force vs Time',
                            xaxis_title='Time (s)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_f, use_container_width=True)
                        fig_z = go.Figure()
                        fig_z.add_trace(go.Scatter(
                            x=normalized_u_epp, y=normalized_f_s, mode='lines', name='Normalized Displacement vs Normalized Restoring Force'))
                        fig_z.update_layout(
                            title='Normalized Displacement vs Normalized Restoring Force',
                            xaxis_title='Normalized Displacement (u/uy)',
                            yaxis_title='Normalized Restoring Force (f_s/Fy)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_z, use_container_width=True)
                        # --- DOWNLOAD SECTION ---
                        results = pd.DataFrame({
                            "Time (s)": time,
                            "Normalized Displacement (u/uy)": normalized_u_epp,
                            "Normalized Restoring Force (f_s/Fy)": normalized_f_s
                        })
                        csv = results.to_csv(index=False).encode('utf-8')
                        # Create PNGs for all 3 plots
                        buffer_u = fig_to_png_bytes(
                            time, normalized_u_epp,
                            "Normalized Displacement vs Time",
                            "Time (s)", "Normalized Displacement (u/uy)"
                        )
                        buffer_f = fig_to_png_bytes(
                            time, normalized_f_s,
                            "Normalized Restoring Force vs Time",
                            "Time (s)", "Normalized Restoring Force (f_s/Fy)"
                        )
                        buffer_z = fig_to_png_bytes(
                            normalized_u_epp, normalized_f_s,
                            "Normalized Displacement vs Normalized Restoring Force",
                            "Normalized Displacement (u/uy)", "Normalized Restoring Force (f_s/Fy)"
                        )
                        # Download buttons
                        with st.expander("📥 Download Simulation Outputs"):
                            st.download_button(
                                label="📄 Download Data as CSV",
                                data=csv,
                                file_name="simulation_results.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="📉 Download Normalized Displacement Plot (PNG)",
                                data=buffer_u,
                                file_name="normalized_displacement_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📈 Download Normalized Restoring Force Plot (PNG)",
                                data=buffer_f,
                                file_name="normalized_restoring_force_plot.png",
                                mime="image/png"
                            )
                            st.download_button(
                                label="📊 Download Displacement vs Restoring Force Plot (PNG)",
                                data=buffer_z,
                                file_name="displacement_vs_restoring_force_plot.png",
                                mime="image/png"
                            )
