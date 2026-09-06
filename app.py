"""
Ghost Invisibility CV - Ultra-Sleek Animated Sci-Fi Web Interface for Streamlit.
Features: Real-Time Live Background Inpainting, 1-Click Invisibility, Preset Backgrounds.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

# Set page config with dark sci-fi layout
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
    bg_manager = BackgroundManager(mode=config.BG_MODE_LIVE)
    return segmenter, mask_processor, blender, ghost_engine, bg_manager


def inject_custom_css():
    """
    Injects animated futuristic sci-fi CSS styling.
    """
    st.markdown("""
    <style>
    /* Global Page Background */
    .stApp {
        background: linear-gradient(135deg, #0a0d14 0%, #101624 50%, #080a10 100%);
        color: #e0e6ed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Animated Ghost Title */
    @keyframes floatGhost {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(4deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    .ghost-emoji {
        display: inline-block;
        animation: floatGhost 3s ease-in-out infinite;
        font-size: 3rem;
    }

    /* Glowing Sci-Fi Header */
    .glowing-title {
        background: linear-gradient(90deg, #00f2fe, #4facfe, #00ff87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: 1px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(20, 27, 45, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Pulsing Status Badges */
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 135, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 135, 0); }
    }
    .badge-active {
        display: inline-block;
        background-color: #00ff87;
        color: #080a10;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 20px;
        animation: pulseGreen 2s infinite;
    }

    /* Streamlit Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c101c !important;
        border-right: 1px solid rgba(0, 242, 254, 0.15);
    }

    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #080a10;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)


def create_preset_background(preset_name: str, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates realistic synthetic preset virtual backgrounds.
    """
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    if preset_name == "Cyberpunk Room":
        for y in range(height):
            r = int(20 + (y / height) * 40)
            g = int(10 + (y / height) * 20)
            b = int(60 + (y / height) * 100)
            bg[y, :] = (b, g, r)
        for x in range(0, width, 40):
            cv2.line(bg, (x, 0), (x, height), (255, 0, 150), 1)
        for y in range(0, height, 40):
            cv2.line(bg, (0, y), (width, y), (0, 242, 254), 1)
        cv2.putText(bg, "CYBERPUNK VIRTUAL STUDIO", (width // 2 - 180, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    elif preset_name == "Modern Office":
        for y in range(height):
            val = int(220 - (y / height) * 60)
            bg[y, :] = (val, val, val)
        cv2.rectangle(bg, (0, int(height * 0.65)), (width, height), (90, 70, 50), -1)
        cv2.putText(bg, "VIRTUAL OFFICE BACKGROUND", (width // 2 - 150, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif preset_name == "Deep Space":
        bg[:, :] = (15, 10, 5)
        np.random.seed(42)
        for _ in range(150):
            sx = np.random.randint(0, width)
            sy = np.random.randint(0, height)
            brightness = np.random.randint(150, 255)
            bg[sy, sx] = (brightness, brightness, brightness)
        cv2.putText(bg, "DEEP SPACE OBSERVATORY", (width // 2 - 150, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    return bg


def main():
    inject_custom_css()

    segmenter, mask_processor, blender, ghost_engine, bg_manager = load_cv_modules()

    # Header Banner
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <span class="ghost-emoji">👻</span>
            <span class="glowing-title">GHOST INVISIBILITY VISION</span>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">
                Real-Time Live Background Invisibility Cloak & AI Human Body Detection
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Control Panel
    st.sidebar.markdown("### 🎛️ Vision Control Panel")

    # Master Invisibility Toggle
    go_invisible = st.sidebar.checkbox("👁️ BECOME 100% INVISIBLE", value=True)

    st.sidebar.markdown("---")

    # Vision Effects Mode
    modes = {
        "👻 Invisibility Cloak": config.MODE_INVISIBLE,
        "⚡ Active Camouflage (Refraction)": config.MODE_CAMOUFLAGE,
        "📷 Normal View": config.MODE_NORMAL,
        "✨ Spectral Ghost": config.MODE_GHOST,
        "⚡ Cyberpunk Neon": config.MODE_NEON_GHOST,
        "👾 Sci-Fi Glitch": config.MODE_GLITCH,
    }
    selected_mode_label = st.sidebar.radio("Select Effect Mode", list(modes.keys()), index=0 if go_invisible else 2)
    active_mode = config.MODE_INVISIBLE if go_invisible else modes[selected_mode_label]

    # Ghost Opacity Slider
    ghost_intensity = st.sidebar.slider("Transparency / Opacity Level", 0.0, 1.0, 0.0 if go_invisible else 0.7, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖼️ Background Method")

    bg_choice = st.sidebar.radio(
        "Choose Background Method:",
        [
            "⚡ Live Dynamic Inpainting (Zero setup! Inpaints live background)",
            "📸 Static Captured Background (Empty room photo)",
            "🌌 Cyberpunk Room Preset",
            "🏢 Modern Office Preset",
            "✨ Deep Space Preset",
        ],
        index=0
    )

    ghost_engine.set_mode(active_mode)
    ghost_engine.set_intensity(ghost_intensity)

    # Layout Columns
    col_main, col_info = st.columns([3, 1.2])

    with col_main:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📷 Live Camera")

        # Static background photo step (only if Static Captured option is picked)
        if bg_choice.startswith("📸"):
            st.info("💡 **Step 1:** Step out of camera view & take a photo of your **empty room** to set your static background.")
            bg_cam_input = st.camera_input("Capture Room Background", key="bg_input")
            if bg_cam_input is not None:
                bytes_data = bg_cam_input.getvalue()
                bg_arr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.session_state["stored_bg"] = bg_arr
                st.success("Static background captured successfully!")

            clean_bg = st.session_state.get("stored_bg", None)
            st.markdown("---")
            st.markdown("💡 **Step 2:** Step in front of camera to see yourself vanish!")

        elif bg_choice.startswith("⚡"):
            st.success("🟢 **Live Dynamic Invisibility Active!** No need to step out. The background behind you is reconstructed live!")
            clean_bg = None

        else:
            preset_name = bg_choice.replace("🌌 ", "").replace("🏢 ", "").replace("✨ ", "")
            clean_bg = create_preset_background(preset_name)
            st.success(f"Using **{preset_name}** background preset!")

        live_cam_input = st.camera_input("Take Live Photo / Video Stream", key="live_input")

        if live_cam_input is not None:
            bytes_data = live_cam_input.getvalue()
            live_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            h, w = live_frame.shape[:2]

            # 1. Pure AI Person Segmentation
            raw_mask = segmenter.segment(live_frame)
            refined_mask = mask_processor.process(raw_mask)

            # 2. Background handling
            if bg_choice.startswith("⚡"):
                # Real-Time Dynamic Inpainting on live background
                clean_bg = bg_manager.update_live_background(live_frame, refined_mask)
            elif clean_bg is None:
                clean_bg = bg_manager.update_live_background(live_frame, refined_mask)
            elif clean_bg.shape[:2] != (h, w):
                clean_bg = cv2.resize(clean_bg, (w, h))

            # 3. Render Invisibility / Vision Effect
            rendered_frame = ghost_engine.render(live_frame, clean_bg, refined_mask)

            # Convert BGR to RGB for web rendering
            rendered_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)

            st.markdown("### 🌟 Output Result")
            st.image(rendered_rgb, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 System Telemetry")

        status_text = "INVISIBLE (100%)" if active_mode == config.MODE_INVISIBLE else active_mode
        st.markdown(f"**Status:** <span class='badge-active'>{status_text}</span>", unsafe_allow_html=True)

        bg_mode_display = "Live Inpainted" if bg_choice.startswith("⚡") else ("Preset" if "Preset" in bg_choice else "Static")
        st.markdown(f"**Background:** `{bg_mode_display}`")
        st.markdown(f"**AI Engine:** MediaPipe Neural Segmenter")
        st.markdown(f"**Dynamic Inpainting:** `Active`")
        st.markdown(f"**Invisibility Level:** `{int((1.0 - ghost_intensity) * 100)}%`")

        st.markdown("---")
        st.markdown("#### ⚡ Live Invisibility Tips")
        st.markdown("""
        - **Live Dynamic Inpainting** uses the real-time surrounding room pixels to erase you with zero setup!
        - If you move the camera or change rooms, the background updates **live**!
        - Try **Active Camouflage** for a sci-fi predator-style glass cloak effect!
        """)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
