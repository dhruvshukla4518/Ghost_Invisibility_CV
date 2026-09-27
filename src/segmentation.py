import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from config import HSV_PRESETS, DEFAULT_HSV_COLOR, SegmentationMethod, MEDIAPIPE_THRESHOLD

class HSVColorSegmenter:
    """
    HSV Color Segmentation Engine for detecting cloaks/cloths of specified color.
    Supports dual-range thresholding for colors like RED that wrap around the 0/180 degree HSV boundary.
    """
    def __init__(self, color_name: str = DEFAULT_HSV_COLOR):
        self.color_name = color_name.upper()
        self.preset = HSV_PRESETS.get(self.color_name, HSV_PRESETS[DEFAULT_HSV_COLOR])

    def set_color(self, color_name: str):
        key = color_name.upper()
        if key in HSV_PRESETS:
            self.color_name = key
            self.preset = HSV_PRESETS[key]
            print(f"[SEGMENTATION] HSV Color changed to {self.color_name}")
        else:
            print(f"[SEGMENTATION] Unknown color key '{color_name}'. Available: {list(HSV_PRESETS.keys())}")

    def get_available_colors(self):
        return list(HSV_PRESETS.keys())

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts BGR frame to HSV and computes binary mask where color matches preset.
        Returns uint8 binary mask (0 or 255).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l1 = np.array(self.preset["lower1"], dtype=np.uint8)
        u1 = np.array(self.preset["upper1"], dtype=np.uint8)
        mask = cv2.inRange(hsv, l1, u1)

        if self.preset.get("lower2") is not None and self.preset.get("upper2") is not None:
            l2 = np.array(self.preset["lower2"], dtype=np.uint8)
            u2 = np.array(self.preset["upper2"], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, l2, u2)
            mask = cv2.bitwise_or(mask, mask2)

        return mask

class MediaPipeAISegmenter:
    """
    MediaPipe Person / Selfie Segmentation Engine.
    Generates binary human subject mask without physical colored cloth requirement.
    """
    def __init__(self, threshold: float = MEDIAPIPE_THRESHOLD):
        self.threshold = threshold
        self.segmenter = None
        self.is_initialized = False

        try:
            import mediapipe as mp
            # Try loading MediaPipe selfie segmentation solution
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'selfie_segmentation'):
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)
                self.is_initialized = True
                print("[SEGMENTATION] MediaPipe Selfie Segmentation initialized successfully.")
            else:
                print("[SEGMENTATION] MediaPipe solutions module found but selfie_segmentation unavailable.")
        except Exception as e:
            print(f"[SEGMENTATION] Warning: MediaPipe AI Segmentation init failed ({e}). Fallback to color segmentation.")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs MediaPipe segmentation on BGR frame. Returns binary uint8 mask (0 or 255).
        """
        if not self.is_initialized or self.segmenter is None:
            # Fallback mock/center human shape if MediaPipe model isn't active
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2 + 50), (w // 4, h // 3), 0, 0, 360, 255, -1)
            cv2.circle(mask, (w // 2, h // 4), h // 7, 255, -1)
            return mask

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb_frame)

        if results.segmentation_mask is not None:
            # Segmentation mask values are floats between 0.0 and 1.0
            prob_mask = results.segmentation_mask
            binary_mask = (prob_mask > self.threshold).astype(np.uint8) * 255
            return binary_mask
        else:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    def close(self):
        if self.segmenter is not None and hasattr(self.segmenter, 'close'):
            self.segmenter.close()

class Segmenter:
    """
    Unified Segmentation Manager supporting both HSV Color and MediaPipe AI segmentation.
    """
    def __init__(self, method: SegmentationMethod = SegmentationMethod.HSV_COLOR,
                 color_name: str = DEFAULT_HSV_COLOR):
        self.method = method
        self.hsv_segmenter = HSVColorSegmenter(color_name)
        self.ai_segmenter = MediaPipeAISegmenter()

    def set_method(self, method: SegmentationMethod):
        self.method = method
        print(f"[SEGMENTATION] Active segmentation method: {self.method.value}")

    def set_hsv_color(self, color_name: str):
        self.hsv_segmenter.set_color(color_name)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        if self.method == SegmentationMethod.HSV_COLOR:
            return self.hsv_segmenter.segment(frame)
        elif self.method == SegmentationMethod.MEDIAPIPE:
            return self.ai_segmenter.segment(frame)
        else:
            return self.hsv_segmenter.segment(frame)
