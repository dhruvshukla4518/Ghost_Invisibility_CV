"""
Configuration settings for the Real-Time Ghost Invisibility System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
UTILS_DIR = BASE_DIR / "utils"
ASSETS_DIR = BASE_DIR / "assets"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
ASSETS_SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
OUTPUT_VIDEOS_DIR = OUTPUTS_DIR / "videos"

# Ensure required directories exist
for folder in [BACKGROUNDS_DIR, ASSETS_SCREENSHOTS_DIR, OUTPUT_SCREENSHOTS_DIR, OUTPUT_VIDEOS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Segmentation Settings
SEGMENTATION_THRESHOLD = 0.4
MASK_BLUR_SIZE = (15, 15)
USE_MEDIAPIPE_SEGMENTATION = True

# Mask Processing Parameters
MORPH_KERNEL_SIZE = (5, 5)
ERODE_ITERATIONS = 1
DILATE_ITERATIONS = 2
GAUSSIAN_BLUR_KERNEL = (15, 15)

# Ghost / Invisibility Effect Parameters
DEFAULT_ALPHA = 0.0          # 0.0 = full background (invisible), 1.0 = full person
DEFAULT_GHOST_INTENSITY = 0.7  # Ghost effect strength/opacity
GHOST_TRAIL_LENGTH = 5        # Number of historical frames for ghost trail
ENABLE_HAND_TRACKING = False   # Set False by default for 30+ FPS performance

# Modes
MODE_NORMAL = "NORMAL"
MODE_GHOST = "GHOST"
MODE_INVISIBLE = "INVISIBLE"
MODE_NEON_GHOST = "NEON_GHOST"
MODE_GLITCH = "GLITCH"

DEFAULT_MODE = MODE_NORMAL

# Keybindings (OpenCV WaitKey lowercased)
KEY_QUIT = ord('q')
KEY_BACKGROUND = ord('b')
KEY_GHOST_MODE = ord('g')
KEY_INVISIBLE_MODE = ord('i')
KEY_NORMAL_MODE = ord('n')
KEY_INCREASE_INTENSITY = ord('+')
KEY_INCREASE_INTENSITY_EQUALS = ord('=')  # Convenience for keyboard without shift
KEY_DECREASE_INTENSITY = ord('-')
KEY_SCREENSHOT = ord('s')
KEY_RESET = ord('r')
KEY_TOGGLE_GESTURE = ord('t')
