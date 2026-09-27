import os
import cv2
import numpy as np
from typing import Optional, List
from config import BACKGROUND_CAPTURE_FRAMES, BG_BLUR_KERNEL_SIZE, BACKGROUNDS_DIR

class BackgroundManager:
    """
    Manages background frame capture, multi-frame temporal median averaging,
    static background image loading, and dynamic background updates.
    """
    def __init__(self, target_width: int, target_height: int):
        self.width = target_width
        self.height = target_height
        self.background_frame: Optional[np.ndarray] = None
        self.is_custom_image: bool = False
        self.captured_count: int = 0
        self._frame_buffer: List[np.ndarray] = []

    def capture_frame(self, frame: np.ndarray) -> bool:
        """
        Appends a live camera frame to the background buffer.
        Returns True when required frame count is reached and background is finalized.
        """
        resized = cv2.resize(frame, (self.width, self.height))
        self._frame_buffer.append(resized.copy())
        self.captured_count += 1

        if len(self._frame_buffer) >= BACKGROUND_CAPTURE_FRAMES:
            self.finalize_background()
            return True
        return False

    def finalize_background(self):
        """
        Computes the temporal median frame over captured buffer frames to remove static noise.
        """
        if not self._frame_buffer:
            print("[BACKGROUND] Warning: Empty background buffer.")
            return

        print(f"[BACKGROUND] Finalizing background using temporal median across {len(self._frame_buffer)} frames...")
        stack = np.stack(self._frame_buffer, axis=0)
        median_bg = np.median(stack, axis=0).astype(np.uint8)

        # Optional light blur to remove high-frequency noise
        if BG_BLUR_KERNEL_SIZE > 0:
            k = BG_BLUR_KERNEL_SIZE if BG_BLUR_KERNEL_SIZE % 2 != 0 else BG_BLUR_KERNEL_SIZE + 1
            median_bg = cv2.GaussianBlur(median_bg, (k, k), 0)

        self.background_frame = median_bg
        self.is_custom_image = False
        self._frame_buffer.clear()
        print("[BACKGROUND] Background captured successfully!")

    def load_static_background(self, image_path: str) -> bool:
        """
        Loads a custom background image from disk.
        """
        if not os.path.exists(image_path):
            print(f"[BACKGROUND] Error: Image not found: {image_path}")
            return False

        img = cv2.imread(image_path)
        if img is None:
            print(f"[BACKGROUND] Error: Failed to read image: {image_path}")
            return False

        self.background_frame = cv2.resize(img, (self.width, self.height))
        self.is_custom_image = True
        print(f"[BACKGROUND] Loaded static custom background: {image_path}")
        return True

    def get_background(self, fallback_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the active background frame. If none captured yet, uses fallback_frame.
        """
        if self.background_frame is not None:
            return self.background_frame.copy()

        if fallback_frame is not None:
            res = cv2.resize(fallback_frame, (self.width, self.height))
            # Apply slight blur as temporary bg
            return cv2.GaussianBlur(res, (15, 15), 0)

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def is_ready(self) -> bool:
        return self.background_frame is not None

    def reset(self):
        self.background_frame = None
        self._frame_buffer.clear()
        self.captured_count = 0
        self.is_custom_image = False
        print("[BACKGROUND] Background reset.")
