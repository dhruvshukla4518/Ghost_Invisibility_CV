import os
from enum import Enum

# Project Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assets & Outputs Directories
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
BACKGROUNDS_DIR = os.path.join(ASSETS_DIR, "backgrounds")
ASSETS_SCREENSHOTS_DIR = os.path.join(ASSETS_DIR, "screenshots")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")
VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")

# Ensure required directories exist
for path in [ASSETS_DIR, BACKGROUNDS_DIR, ASSETS_SCREENSHOTS_DIR, OUTPUTS_DIR, SCREENSHOTS_DIR, VIDEOS_DIR]:
    os.makedirs(path, exist_ok=True)

class OperationalMode(str, Enum):
    CLOAK = "CLOAK"              # Full Invisibility Cloak (Replace subject/color with bg)
    GHOST = "GHOST"              # Semi-transparent phantom with motion trails & aura
    SWAP = "BACKGROUND_SWAP"     # AI background replacement
    NORMAL = "NORMAL"            # Unfiltered camera feed with HUD

class SegmentationMethod(str, Enum):
    HSV_COLOR = "HSV_COLOR"       # Color-based segmentation (cloth/cloak)
    MEDIAPIPE = "MEDIAPIPE_AI"    # AI Selfie / Person Segmentation

# Camera Settings
DEFAULT_CAMERA_ID = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_TARGET_FPS = 30

# Background Subtraction & Capture Settings
BACKGROUND_CAPTURE_FRAMES = 30  # Number of initial frames to average for clean bg
BG_BLUR_KERNEL_SIZE = 5         # Gaussian blur kernel for noise reduction

# HSV Color Threshold Presets for Cloak/Cloth
# Format: Lower HSV, Upper HSV (H: 0-180, S: 0-255, V: 0-255)
# Note: Red wraps around HSV 0/180, handled as dual ranges in segmentation.py
HSV_PRESETS = {
    "RED": {
        "lower1": [0, 120, 70],
        "upper1": [10, 255, 255],
        "lower2": [170, 120, 70],
        "upper2": [180, 255, 255]
    },
    "GREEN": {
        "lower1": [35, 80, 50],
        "upper1": [85, 255, 255],
        "lower2": None,
        "upper2": None
    },
    "BLUE": {
        "lower1": [90, 80, 50],
        "upper1": [135, 255, 255],
        "lower2": None,
        "upper2": None
    },
    "CYAN": {
        "lower1": [80, 100, 100],
        "upper1": [100, 255, 255],
        "lower2": None,
        "upper2": None
    },
    "MAGENTA": {
        "lower1": [140, 100, 100],
        "upper1": [165, 255, 255],
        "lower2": None,
        "upper2": None
    },
    "YELLOW": {
        "lower1": [20, 100, 100],
        "upper1": [35, 255, 255],
        "lower2": None,
        "upper2": None
    }
}
DEFAULT_HSV_COLOR = "RED"

# MediaPipe AI Segmentation Settings
MEDIAPIPE_THRESHOLD = 0.5  # Confidence threshold for person segmentation

# Mask Processing / Morphology Settings
MORPH_KERNEL_SIZE = 5      # Kernel size for Morphological Opening & Closing
GAUSSIAN_BLUR_SIGMA = 5    # Edge softening blur sigma for mask blending
MIN_CONTOUR_AREA = 500     # Min contour area (pixels) to filter out noise specks

# Ghost Visual Effects Config
DEFAULT_GHOST_OPACITY = 0.0     # Default transparency (0.0 = fully invisible, 1.0 = opaque)
MAX_MOTION_TRAIL_FRAMES = 8     # Number of past frames in ghost motion trail queue
TRAIL_DECAY_FACTOR = 0.7        # Exponential decay factor for older frames in trail
AURA_GLOW_RADIUS = 15           # Pixel thickness of ghost glowing edge aura
AURA_COLOR = (255, 255, 100)    # BGR color for ghostly aura (Cyan-Yellow glow)
COLORMAP_CHOICES = ["BONE", "OCEAN", "JET", "PLASMA", "CYBERPUNK"]
DEFAULT_COLORMAP = "BONE"

# GUI & HUD Settings
WINDOW_TITLE = "Ghost Invisibility CV - AI Vision System"
HUD_TEXT_COLOR = (0, 255, 200)      # Neon teal text
HUD_HIGHLIGHT_COLOR = (0, 165, 255) # Electric orange
RECORDING_PULSE_SPEED = 0.15        # Pulse speed for red recording dot
DEFAULT_SHOW_HUD = False            # Sci-Fi HUD overlay hidden/off by default

# Keybindings
KEYS = {
    "QUIT": [ord('q'), 27],          # 'q' or ESC
    "CAPTURE_BG": [ord('b'), ord('B')],
    "TOGGLE_HUD": [ord('h'), ord('H')],
    "CYCLE_MODE": [ord('c'), ord('C')],
    "CYCLE_ENGINE": [ord('m'), ord('M'), ord('e'), ord('E')],
    "CYCLE_COLOR": [ord('k'), ord('K')],
    "TOGGLE_GHOST": [ord('g'), ord('G')],
    "SCREENSHOT": [ord('s'), ord('S')],
    "RECORD": [ord('r'), ord('R')],
    "OPACITY_UP": [ord('+'), ord('=')],
    "OPACITY_DOWN": [ord('-'), ord('_')]
}
