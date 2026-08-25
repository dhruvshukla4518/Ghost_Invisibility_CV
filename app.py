"""
Ghost Invisibility CV - Streamlit Web Application for Streamlit Cloud Deployment.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Ghost Invisibility CV",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.segmentation import PersonSegmenter
from src.mask_processing import MaskProcessor
from src.blending import Blender
from src.ghost_effect import GhostEffectEngine
from src.background import BackgroundManager


@st.cache_resource
def load_cv_modules():
    """
    Initializes and caches CV modules for fast Streamlit re-renders.
    """
    segmenter = PersonSegmenter(threshold=config.SEGMENTATION_THRESHOLD)
    mask_processor = MaskProcessor()
    blender = Blender(default_alpha=config.DEFAULT_ALPHA)
    ghost_engine = GhostEffectEngine(blender=blender)
    bg_manager = BackgroundManager()
    return segmenter, mask_processor, blender, ghost_engine, bg_manager


def main():
    st.title("👻 Real-Time Ghost Invisibility System")
    st.markdown("Web application powered by **MediaPipe AI Person Segmentation** and **OpenCV**.")

    segmenter, mask_processor, blender, ghost_engine, bg_manager = load_cv_modules()

    # Sidebar Controls
    st.sidebar.header("⚙️ Control Panel")

    # Invisibility Toggle Quick Button
    invisible_toggle = st.sidebar.checkbox("👁️ Enable 100% Invisibility", value=False)

    # Operational Mode Radio
    modes = [config.MODE_NORMAL, config.MODE_INVISIBLE, config.MODE_GHOST, config.MODE_NEON_GHOST, config.MODE_GLITCH]
    selected_mode = st.sidebar.radio("Select Vision Mode", modes, index=1 if invisible_toggle else 0)

    # Intensity / Alpha Slider
    ghost_intensity = st.sidebar.slider("Ghost Effect Intensity / Alpha", 0.0, 1.0, 0.7, 0.05)

    # Background Capture Controls
    st.sidebar.subheader("📸 Background Reference")

    if st.sidebar.button("🔄 Reset / Recapture Background"):
        st.session_state["bg_frame"] = None
        st.sidebar.success("Background reset! Take a photo of your empty room to store background.")

    bg_status = "READY ✅" if ("bg_frame" in st.session_state and st.session_state["bg_frame"] is not None) else "NOT SET ⚠️"
    st.sidebar.write(f"**Background Status:** {bg_status}")

    # Set mode and intensity
    active_mode = config.MODE_INVISIBLE if invisible_toggle else selected_mode
    ghost_engine.set_mode(active_mode)
    ghost_engine.set_intensity(ghost_intensity)

    # Main Web Interface Tabs
    tab_cam, tab_help = st.tabs(["📷 Web Camera", "ℹ️ How to Use"])

    with tab_cam:
        st.write("### Take a photo or stream your camera")

        # Background capture photo input
        if "bg_frame" not in st.session_state or st.session_state["bg_frame"] is None:
            st.info("👉 **Step 1:** Step out of camera view and take a photo of your **empty room** to set the clean background.")
            bg_input = st.camera_input("Capture Empty Room Background", key="bg_cam_input")

            if bg_input is not None:
                bytes_data = bg_input.getvalue()
                bg_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state["bg_frame"] = bg_array
                bg_manager.capture_background(bg_array)
                st.success("Clean room background captured! Now step in front of the camera below.")
                st.rerun()

        # Live Camera Input
        st.write("👉 **Step 2:** Step in front of the camera and test invisibility!")
        cam_input = st.camera_input("Live Camera Feed", key="live_cam_input")

        if cam_input is not None and "bg_frame" in st.session_state and st.session_state["bg_frame"] is not None:
            bytes_data = cam_input.getvalue()
            live_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            clean_bg = st.session_state["bg_frame"]

            # Resize background if shape differs
            h, w = live_frame.shape[:2]
            if clean_bg.shape[:2] != (h, w):
                clean_bg = cv2.resize(clean_bg, (w, h))

            # Run CV Pipeline
            raw_mask = segmenter.segment(live_frame, clean_bg)
            refined_mask = mask_processor.process(raw_mask)
            rendered_frame = ghost_engine.render(live_frame, clean_bg, refined_mask)

            # Convert BGR to RGB for Streamlit display
            rendered_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)

            st.subheader(f"Output Frame - [Mode: {active_mode}]")
            st.image(rendered_rgb, use_container_width=True)

    with tab_help:
        st.markdown("""
        ### Instructions for Streamlit Cloud:
        1. **Step 1**: Step out of camera view and click **Capture Empty Room Background**.
        2. **Step 2**: Step in front of the camera and take a photo under **Live Camera Feed**.
        3. Check **Enable 100% Invisibility** in the sidebar control panel to see your body completely vanish into the background!
        """)


if __name__ == "__main__":
    main()
