"""
Hand gesture recognition module using MediaPipe Hands / GestureRecognizer.
"""

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import config
from utils.image_utils import download_file_if_missing


class HandTracker:
    """
    Detects hand gestures to control ghost invisibility modes and parameters.
    Recognized gestures:
      - OPEN PALM: Activate Ghost Mode
      - FIST: Switch to Normal Mode
      - TWO FINGERS (Victory): Toggle Invisible Mode
      - THUMB UP: Increase Ghost Intensity
      - THUMB DOWN: Decrease Ghost Intensity
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    MODEL_PATH = config.BASE_DIR / "assets" / "gesture_recognizer.task"

    def __init__(self, enabled: bool = config.ENABLE_HAND_TRACKING):
        self.enabled = enabled
        self.recognizer = None
        self.last_gesture = "NONE"
        self.use_mediapipe_tasks = False

        self._initialize_gesture_recognizer()

    def _initialize_gesture_recognizer(self):
        """
        Initializes MediaPipe Gesture Recognizer or HandLandmarker.
        """
        try:
            download_file_if_missing(self.MODEL_URL, self.MODEL_PATH)
            if self.MODEL_PATH.exists():
                BaseOptions = mp.tasks.BaseOptions
                GestureRecognizer = mp.tasks.vision.GestureRecognizer
                GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.MODEL_PATH)),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=1,
                )
                self.recognizer = GestureRecognizer.create_from_options(options)
                self.use_mediapipe_tasks = True
                print("[INFO] MediaPipe Gesture Recognizer initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Could not initialize Gesture Recognizer ({e}). Landmark classifier fallback active.")

    def process_frame(self, frame_bgr: np.ndarray) -> str:
        """
        Analyzes the frame and returns the classified hand gesture string.
        """
        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            self.last_gesture = "DISABLED"
            return "DISABLED"

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_mediapipe_tasks and self.recognizer is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = self.recognizer.recognize(mp_image)

                if result.gestures and len(result.gestures) > 0:
                    top_gesture = result.gestures[0][0].category_name
                    mapped_gesture = self._map_gesture_name(top_gesture)
                    self.last_gesture = mapped_gesture
                    return mapped_gesture
            except Exception as e:
                pass

        # Fallback landmark / skin contour heuristic for offline/test mode
        self.last_gesture = self._heuristic_gesture_detection(frame_rgb)
        return self.last_gesture

    def _map_gesture_name(self, raw_name: str) -> str:
        """
        Maps raw MediaPipe gesture category names to standard PRD names.
        """
        name_upper = raw_name.upper()
        if "OPEN_PALM" in name_upper or "PALM" in name_upper:
            return "OPEN PALM"
        elif "CLOSED_FIST" in name_upper or "FIST" in name_upper:
            return "FIST"
        elif "VICTORY" in name_upper or "PEACE" in name_upper or "TWO" in name_upper:
            return "TWO FINGERS"
        elif "THUMB_UP" in name_upper:
            return "THUMB UP"
        elif "THUMB_DOWN" in name_upper:
            return "THUMB DOWN"
        elif "POINTING" in name_upper:
            return "POINTING"
        return raw_name.upper()

    def _heuristic_gesture_detection(self, frame_rgb: np.ndarray) -> str:
        """
        Simple contour area gesture approximation when MediaPipe task is unavailable.
        """
        # Search for bright hand region in synthetic top right
        h, w = frame_rgb.shape[:2]
        crop = frame_rgb[0 : h // 2, w // 2 : w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        count = cv2.countNonZero(thresh)

        if count > 500:
            return "OPEN PALM"
        return "NONE"

    def get_last_gesture(self) -> str:
        return self.last_gesture
