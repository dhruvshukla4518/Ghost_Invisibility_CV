"""
Clean Background Manager for Real-Time Human Invisibility.
Captures the real room background before the human is in frame or loads realistic presets.
Seamlessly adapts brightness and color to match live ambient room lighting.
"""

from pathlib import Path
from collections import deque
import cv2
import numpy as np
import config


class BackgroundManager:
    """
    Manages clean reference backgrounds for human invisibility.
    1. Auto-captures the clean room background before a human enters the frame.
    2. Allows manual capture via 'B' key when stepping out of view.
    3. Provides instant realistic presets via 'P' key.
    4. Matches live ambient exposure so the background blends with current lighting.
    """

    PRESETS = ["CLEAN_ROOM", "MODERN_STUDIO", "CYBERPUNK_ROOM", "DEEP_SPACE"]

    def __init__(
        self,
        storage_path: Path = config.BACKGROUNDS_DIR / "clean_background.png",
    ):
        self.storage_path = storage_path
        self.background_frame = None
        self.is_captured = False
        self.current_preset_idx = 0
        self.is_preset_active = False

        # Auto-capture accumulation buffer (averages 5 clean frames for noise-free capture)
        self.capture_buffer = deque(maxlen=5)

        # Attempt to load saved background from disk
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
                print(f"[INFO] Loaded clean background from: {self.storage_path}")

    def capture_background(self, current_frame: np.ndarray) -> bool:
        """
        Captures the current frame as the clean background.
        """
        if current_frame is None or current_frame.size == 0:
            return False

        self.background_frame = current_frame.copy()
        self.is_captured = True
        self.is_preset_active = False

        # Save to disk
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.storage_path), self.background_frame)
        print(f"[INFO] Clean background captured ({self.background_frame.shape[1]}x{self.background_frame.shape[0]}).")
        return True

    def auto_capture_if_clear(self, current_frame: np.ndarray, person_mask: np.ndarray) -> bool:
        """
        Automatically captures clean background when no human is detected in frame.
        Returns True if a capture was just completed.
        """
        if current_frame is None or person_mask is None:
            return False

        # Check if human is present in frame (coverage > 2% of frame)
        human_coverage = np.mean(person_mask > 0.3)
        if human_coverage < 0.02:
            self.capture_buffer.append(current_frame.copy())
            if len(self.capture_buffer) == 5:
                # Average frames to reduce sensor noise
                avg_bg = np.mean(self.capture_buffer, axis=0).astype(np.uint8)
                self.capture_background(avg_bg)
                self.capture_buffer.clear()
                return True
        else:
            self.capture_buffer.clear()

        return False

    def cycle_preset(self, target_shape: tuple = None) -> str:
        """
        Cycles through realistic background presets (via 'P' key).
        """
        self.current_preset_idx = (self.current_preset_idx + 1) % len(self.PRESETS)
        preset_name = self.PRESETS[self.current_preset_idx]
        self.background_frame = self._generate_preset(preset_name, target_shape)
        self.is_captured = True
        self.is_preset_active = True
        print(f"[INFO] Switched to background preset: {preset_name}")
        return preset_name

    def _generate_preset(self, preset_name: str, target_shape: tuple = None) -> np.ndarray:
        """
        Generates realistic room/studio background presets.
        """
        w, h = target_shape if target_shape is not None else (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        if preset_name == "CLEAN_ROOM":
            # Neutral warm wall & floor
            for y in range(h):
                ratio = y / float(h)
                if ratio < 0.7:
                    # Wall: warm cream
                    bg[y, :] = (int(215 - ratio * 20), int(225 - ratio * 15), int(235 - ratio * 10))
                else:
                    # Floor: wooden brown
                    floor_ratio = (ratio - 0.7) / 0.3
                    bg[y, :] = (int(50 + floor_ratio * 30), int(80 + floor_ratio * 40), int(120 + floor_ratio * 50))
            # Wall molding line
            cv2.line(bg, (0, int(h * 0.7)), (w, int(h * 0.7)), (180, 185, 190), 3)

        elif preset_name == "MODERN_STUDIO":
            for y in range(h):
                val = int(240 - (y / h) * 70)
                bg[y, :] = (val, val, val)
            cv2.rectangle(bg, (0, int(h * 0.65)), (w, h), (70, 60, 50), -1)

        elif preset_name == "CYBERPUNK_ROOM":
            for y in range(h):
                r = int(25 + (y / h) * 45)
                g = int(15 + (y / h) * 25)
                b = int(70 + (y / h) * 110)
                bg[y, :] = (b, g, r)
            for x in range(0, w, 40):
                cv2.line(bg, (x, 0), (x, h), (255, 0, 150), 1)
            for y in range(0, h, 40):
                cv2.line(bg, (0, y), (w, y), (0, 242, 254), 1)

        else: # DEEP_SPACE
            bg[:, :] = (15, 10, 5)
            np.random.seed(42)
            for _ in range(120):
                sx = np.random.randint(0, w)
                sy = np.random.randint(0, h)
                b_val = np.random.randint(180, 255)
                bg[sy, sx] = (b_val, b_val, b_val)

        return bg

    def get_background(
        self,
        target_shape: tuple = None,
        live_frame: np.ndarray = None,
        person_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the clean background frame resized to target_shape.
        Adapts brightness to match live ambient lighting if live_frame is provided.
        """
        if self.background_frame is None:
            if live_frame is not None:
                # Generate clean room preset as immediate fallback
                self.background_frame = self._generate_preset("CLEAN_ROOM", target_shape)
                self.is_captured = True
            else:
                return None

        bg = self.background_frame.copy()

        # Resize if target_shape is specified
        if target_shape is not None:
            w, h = target_shape
            if bg.shape[1] != w or bg.shape[0] != h:
                bg = cv2.resize(bg, (w, h))

        # Live Exposure & Lighting Adaptation:
        # If live_frame is given, adjust background brightness to match live lighting
        if live_frame is not None and person_mask is not None and not self.is_preset_active:
            try:
                unmasked = person_mask < 0.2
                if np.sum(unmasked) > 500:
                    live_brightness = np.mean(live_frame[unmasked])
                    bg_brightness = np.mean(bg[unmasked])
                    ratio = live_brightness / (bg_brightness + 1e-5)
                    # Clip ratio between 0.7 and 1.3 to avoid extreme shifts
                    ratio = np.clip(ratio, 0.7, 1.3)
                    bg = np.clip(bg.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
            except Exception:
                pass

        return bg

    def has_background(self) -> bool:
        """
        Returns True if a background is captured or ready.
        """
        return self.is_captured and self.background_frame is not None
