"""
Dynamic Background Manager with Real-Time Live Background Inpainting and Camera Tracking.
Provides both live dynamic background reconstruction and static snapshot modes.
"""

from pathlib import Path
import cv2
import numpy as np
import config


class BackgroundManager:
    """
    Manages background frames. Supports:
    1. LIVE Mode: Real-time dynamic background reconstruction, live inpainting,
       ambient exposure adaptation, and camera motion compensation.
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

        # Live Dynamic Background Buffers
        self.live_background = None
        self.bg_confidence = None
        self.prev_gray = None
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
        Updates the live background model using the incoming camera frame and person mask.
        - Updates unmasked background pixels with the current live scene.
        - Fills the person silhouette using real-time multi-scale inpainting & memory.
        - Adapts to camera motion and ambient lighting changes.
        """
        if current_frame is None or current_frame.size == 0:
            return self.live_background

        h, w = current_frame.shape[:2]

        # Initialize buffers if not created or dimensions changed
        if self.live_background is None or self.live_background.shape[:2] != (h, w):
            self.live_background = current_frame.copy()
            self.bg_confidence = np.zeros((h, w), dtype=np.float32)

        if person_mask is None:
            # Entire frame is background
            self.live_background = current_frame.copy()
            self.bg_confidence.fill(1.0)
            return self.live_background

        # Resize mask if shape differs
        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))

        # Binary unmasked area (where person is NOT present)
        unmasked = person_mask < 0.2
        masked = ~unmasked

        # 1. Camera Motion Compensation (Optional feature tracking on background)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if config.ENABLE_CAMERA_STABILIZATION and self.prev_gray is not None and self.prev_gray.shape == (h, w):
            self._compensate_camera_motion(curr_gray, unmasked)
        self.prev_gray = curr_gray

        # 2. Update all visible background pixels directly from current live frame
        # Smoothly update unmasked background to absorb lighting changes
        self.live_background[unmasked] = current_frame[unmasked]
        self.bg_confidence[unmasked] = 1.0

        # Slowly decay confidence of occluded regions
        self.bg_confidence[masked] = np.maximum(0.0, self.bg_confidence[masked] - 0.02)

        # 3. For occluded regions with low confidence or when camera changed:
        # Generate real-time live inpainting from surrounding live background
        needs_inpaint = masked & (self.bg_confidence < 0.3)
        if np.any(needs_inpaint):
            inpainted = self._fast_inpaint(current_frame, person_mask)
            # Blend inpainted pixels into low-confidence regions
            self.live_background[needs_inpaint] = inpainted[needs_inpaint]

        # 4. Live Exposure & Color Adaptation
        # If lighting in the room changed, adjust brightness of the occluded background
        if np.any(unmasked):
            live_mean = np.mean(current_frame[unmasked], axis=0)
            bg_mean = np.mean(self.live_background[unmasked], axis=0)
            diff = live_mean - bg_mean
            if np.linalg.norm(diff) > 2.0:
                # Apply color/brightness shift to occluded background pixels
                adjusted_bg = np.clip(self.live_background.astype(np.float32) + diff, 0, 255).astype(np.uint8)
                self.live_background[masked] = adjusted_bg[masked]

        return self.live_background

    def _fast_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Performs fast multi-scale inpainting on the live frame using Telea algorithm.
        Downscales the image and mask for real-time 30+ FPS speed, then upscales.
        """
        h, w = frame.shape[:2]
        scale = self.inpaint_scale

        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))

        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        small_mask = cv2.resize((mask > 0.3).astype(np.uint8) * 255, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

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

    def _compensate_camera_motion(self, curr_gray: np.ndarray, unmasked: np.ndarray):
        """
        Estimates camera translation across frames using optical flow on background points.
        """
        try:
            # Find feature points in unmasked background of previous frame
            mask_uint8 = unmasked.astype(np.uint8) * 255
            prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=60,
                qualityLevel=0.03,
                minDistance=15,
                mask=mask_uint8,
            )

            if prev_pts is not None and len(prev_pts) >= 10:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, prev_pts, None
                )
                valid_prev = prev_pts[status == 1]
                valid_curr = curr_pts[status == 1]

                if len(valid_prev) >= 8:
                    M, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
                    if M is not None:
                        # Warp the background buffer and confidence map by the camera shift
                        h, w = curr_gray.shape[:2]
                        self.live_background = cv2.warpAffine(
                            self.live_background, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
                        )
                        self.bg_confidence = cv2.warpAffine(
                            self.bg_confidence, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                        )
        except Exception:
            pass

    def get_background(
        self,
        target_shape: tuple = None,
        current_frame: np.ndarray = None,
        person_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Returns the background image:
        - In LIVE mode: dynamically reconstructs and returns the live background.
        - In STATIC mode: returns the captured static snapshot (or falls back to live).
        """
        # A. LIVE Mode
        if self.mode == config.BG_MODE_LIVE:
            if current_frame is not None:
                bg = self.update_live_background(current_frame, person_mask)
            elif self.live_background is not None:
                bg = self.live_background.copy()
            elif self.static_background is not None:
                bg = self.static_background.copy()
            else:
                return None

        # B. STATIC Mode
        else:
            if self.is_captured and self.static_background is not None:
                bg = self.static_background.copy()
            elif current_frame is not None:
                # Fallback to live background if no static snapshot captured yet
                bg = self.update_live_background(current_frame, person_mask)
            elif self.live_background is not None:
                bg = self.live_background.copy()
            else:
                return None

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
