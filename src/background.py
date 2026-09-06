"""
Dynamic Background Manager with Live Inpainting and Pure Background Preservation.
Keeps 100% of the live background intact while erasing the detected human being in real time.
"""

from pathlib import Path
import cv2
import numpy as np
import config


class BackgroundManager:
    """
    Manages background frames. Supports:
    1. LIVE Mode: Real-time dynamic background reconstruction. Keeps 100% of the
       live background feed intact as-is, and in-paints the detected human region.
    2. STATIC Mode: Traditional frozen background snapshot (via 'B' key).
    """

    def __init__(
        self,
        storage_path: Path = config.BACKGROUNDS_DIR / "clean_background.png",
        mode: str = config.DEFAULT_BG_MODE,
    ):
        self.storage_path = storage_path
        self.mode = mode
        self.static_background = None
        self.is_captured = False

        # Live Dynamic Memory Buffer
        self.temporal_bg = None
        self.bg_seen_mask = None
        self.inpaint_scale = config.INPAINT_SCALE
        self.inpaint_radius = config.INPAINT_RADIUS

        # Attempt auto-load existing saved background image if available
        self._load_saved_background()

    def _load_saved_background(self):
        """
        Loads saved background image file if present on disk.
        """
        if self.storage_path.exists():
            img = cv2.imread(str(self.storage_path))
            if img is not None and img.size > 0:
                self.static_background = img
                self.is_captured = True
                print(f"[INFO] Loaded static background from: {self.storage_path}")

    def set_mode(self, mode: str):
        """
        Switches between LIVE and STATIC background modes.
        """
        if mode in [config.BG_MODE_LIVE, config.BG_MODE_STATIC]:
            self.mode = mode
            print(f"[INFO] Background mode set to: {self.mode}")

    def toggle_mode(self) -> str:
        """
        Toggles between LIVE and STATIC background modes.
        """
        if self.mode == config.BG_MODE_LIVE:
            self.set_mode(config.BG_MODE_STATIC)
        else:
            self.set_mode(config.BG_MODE_LIVE)
        return self.mode

    def capture_background(self, current_frame: np.ndarray) -> bool:
        """
        Captures the current frame as the static reference clean background.
        """
        if current_frame is None or current_frame.size == 0:
            print("[WARNING] Cannot capture empty frame as background.")
            return False

        self.static_background = current_frame.copy()
        self.is_captured = True

        # Save to disk
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.storage_path), self.static_background)
        print(f"[INFO] Captured clean background snapshot ({self.static_background.shape[1]}x{self.static_background.shape[0]}).")
        return True

    def update_live_background(
        self,
        current_frame: np.ndarray,
        person_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generates live background:
        - Outside the person mask: EXACT live camera frame is preserved 100% as it is.
        - Inside the person mask: Live multi-scale inpainting from surrounding live pixels,
          fused with temporal memory of previously uncovered scene pixels.
        """
        if current_frame is None or current_frame.size == 0:
            return current_frame

        h, w = current_frame.shape[:2]

        # Initialize temporal memory buffers if needed
        if self.temporal_bg is None or self.temporal_bg.shape[:2] != (h, w):
            self.temporal_bg = current_frame.copy()
            self.bg_seen_mask = np.zeros((h, w), dtype=np.uint8)

        if person_mask is None:
            self.temporal_bg = current_frame.copy()
            self.bg_seen_mask.fill(255)
            return current_frame.copy()

        # Ensure mask matches frame dimensions
        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))

        # 1. Update temporal memory for all pixels outside the person
        unmasked = person_mask < 0.15
        self.temporal_bg[unmasked] = current_frame[unmasked]
        self.bg_seen_mask[unmasked] = 255

        # 2. Start from the exact current live frame (keeps live bg 100% as it is)
        live_bg = current_frame.copy()

        # 3. For pixels occluded by the person:
        # Perform fast real-time inpainting using the surrounding live frame pixels
        inpainted = self._fast_inpaint(current_frame, person_mask)

        # 4. Check if any occluded pixels were previously seen in temporal memory
        masked = ~unmasked
        previously_seen = masked & (self.bg_seen_mask == 255)

        # Blend: use temporal memory where previously revealed, live inpainting elsewhere
        inpaint_fill = inpainted.copy()
        if np.any(previously_seen):
            inpaint_fill[previously_seen] = self.temporal_bg[previously_seen]

        # Insert reconstructed background only inside the person mask
        live_bg[masked] = inpaint_fill[masked]

        return live_bg

    def _fast_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Performs fast multi-scale inpainting on the live frame using Telea algorithm.
        Downscales for real-time 30+ FPS speed, then upscales smoothly.
        """
        h, w = frame.shape[:2]
        scale = self.inpaint_scale

        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))

        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        small_mask = cv2.resize((mask > 0.2).astype(np.uint8) * 255, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # Inpaint scaled image using Telea algorithm
        small_inpainted = cv2.inpaint(
            small_frame,
            small_mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )

        # Upscale back to original resolution
        inpainted_full = cv2.resize(small_inpainted, (w, h), interpolation=cv2.INTER_LINEAR)
        return inpainted_full

    def get_background(
        self,
        target_shape: tuple = None,
        current_frame: np.ndarray = None,
        person_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the background image:
        - In LIVE mode: dynamically reconstructs and returns live background with 100% live BG preservation.
        - In STATIC mode: returns the captured static snapshot (or falls back to live).
        """
        # A. LIVE Mode
        if self.mode == config.BG_MODE_LIVE:
            if current_frame is not None:
                bg = self.update_live_background(current_frame, person_mask)
            elif self.temporal_bg is not None:
                bg = self.temporal_bg.copy()
            elif self.static_background is not None:
                bg = self.static_background.copy()
            else:
                return current_frame

        # B. STATIC Mode
        else:
            if self.is_captured and self.static_background is not None:
                bg = self.static_background.copy()
            elif current_frame is not None:
                bg = self.update_live_background(current_frame, person_mask)
            elif self.temporal_bg is not None:
                bg = self.temporal_bg.copy()
            else:
                return current_frame

        # Resize if target shape (width, height) is specified
        if bg is not None and target_shape is not None:
            w, h = target_shape
            if bg.shape[1] != w or bg.shape[0] != h:
                bg = cv2.resize(bg, (w, h))

        return bg

    def has_background(self) -> bool:
        """
        Returns True if a valid background (live or static) is ready.
        """
        if self.mode == config.BG_MODE_LIVE:
            return True  # Live mode generates background on-the-fly
        return self.is_captured and self.static_background is not None
