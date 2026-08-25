"""
Background capture and storage management module.
"""

from pathlib import Path
import cv2
import numpy as np
import config


class BackgroundManager:
    """
    Captures, stores, and manages clean background reference frames.
    """

    def __init__(self, storage_path: Path = config.BACKGROUNDS_DIR / "clean_background.png"):
        self.storage_path = storage_path
        self.background_frame = None
        self.is_captured = False

        # Attempt auto-load existing saved background image if available
        self._load_saved_background()

    def _load_saved_background(self):
        """
        Loads saved background image file if present on disk.
        """
        if self.storage_path.exists():
            img = cv2.imread(str(self.storage_path))
            if img is not None and img.size > 0:
                self.background_frame = img
                self.is_captured = True
                print(f"[INFO] Loaded background from: {self.storage_path}")

    def capture_background(self, current_frame: np.ndarray) -> bool:
        """
        Captures the current frame as the reference clean background.
        """
        if current_frame is None or current_frame.size == 0:
            print("[WARNING] Cannot capture empty frame as background.")
            return False

        self.background_frame = current_frame.copy()
        self.is_captured = True

        # Save to disk
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.storage_path), self.background_frame)
        print(f"[INFO] Captured clean background frame ({self.background_frame.shape[1]}x{self.background_frame.shape[0]}).")
        return True

    def get_background(self, target_shape: tuple = None) -> np.ndarray:
        """
        Returns the captured background image.
        If resized target_shape (width, height) is provided, resizes background accordingly.
        """
        if not self.is_captured or self.background_frame is None:
            return None

        bg = self.background_frame.copy()
        if target_shape is not None:
            w, h = target_shape
            if bg.shape[1] != w or bg.shape[0] != h:
                bg = cv2.resize(bg, (w, h))

        return bg

    def has_background(self) -> bool:
        """
        Returns True if a clean background frame is captured.
        """
        return self.is_captured and self.background_frame is not None
