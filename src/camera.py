import cv2
import numpy as np
import time
from typing import Tuple, Optional
from config import DEFAULT_CAMERA_ID, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT

class SyntheticCamera:
    """
    Generates synthetic video frames with animated subjects and backgrounds
    for testing when physical webcam is not present.
    """
    def __init__(self, width: int = DEFAULT_FRAME_WIDTH, height: int = DEFAULT_FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.start_time = time.time()
        # Pre-generate static background image
        self.bg_frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Create a cool geometric grid background
        for y in range(0, height, 40):
            cv2.line(self.bg_frame, (0, y), (width, y), (40, 40, 50), 1)
        for x in range(0, width, 40):
            cv2.line(self.bg_frame, (x, 0), (x, height), (40, 40, 50), 1)
        cv2.putText(self.bg_frame, "SYNTHETIC BACKGROUND FRAME", (width // 2 - 250, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 150, 100), 2)

    def read(self) -> Tuple[bool, np.ndarray]:
        t = time.time() - self.start_time
        # Start with static background frame copy
        frame = self.bg_frame.copy()

        # Animate a bouncing "human cloak target" (Red/Green rectangle & circle)
        cx = int(self.width / 2 + np.sin(t * 1.5) * (self.width * 0.3))
        cy = int(self.height / 2 + np.cos(t * 2.0) * (self.height * 0.2))

        # Draw red cloak region (pure RED for HSV segmentation testing)
        cv2.rectangle(frame, (cx - 80, cy - 100), (cx + 80, cy + 120), (0, 0, 220), -1) # BGR Red
        cv2.circle(frame, (cx, cy - 130), 40, (200, 180, 150), -1) # Skin-colored head

        # Add text label on animated object
        cv2.putText(frame, "Synthetic Cloak Subject", (cx - 90, cy + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return True, frame

class CameraManager:
    """
    Manages live camera capture via OpenCV VideoCapture, automatically falling back
    to SyntheticCamera if hardware is unavailable.
    """
    def __init__(self, camera_id: int = DEFAULT_CAMERA_ID,
                 width: int = DEFAULT_FRAME_WIDTH,
                 height: int = DEFAULT_FRAME_HEIGHT,
                 force_synthetic: bool = False):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.is_synthetic = force_synthetic
        self.cap: Optional[cv2.VideoCapture] = None
        self.synth_cam: Optional[SyntheticCamera] = None

        if force_synthetic:
            self._init_synthetic()
        else:
            self._init_hardware()

    def _init_hardware(self):
        print(f"[CAMERA] Attempting to initialize camera ID {self.camera_id} ({self.width}x{self.height})...")
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)

        if not self.cap.isOpened():
            print(f"[CAMERA] Warning: Unable to open camera ID {self.camera_id}. Switching to Synthetic Camera.")
            self._init_synthetic()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Verify frame read
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[CAMERA] Warning: Camera opened but failed to read frame. Switching to Synthetic Camera.")
            self.cap.release()
            self.cap = None
            self._init_synthetic()
        else:
            self.height, self.width = frame.shape[:2]
            print(f"[CAMERA] Physical camera active. Resolution: {self.width}x{self.height}")

    def _init_synthetic(self):
        self.is_synthetic = True
        self.synth_cam = SyntheticCamera(self.width, self.height)
        print(f"[CAMERA] Synthetic camera active ({self.width}x{self.height}).")

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.is_synthetic and self.synth_cam is not None:
            return self.synth_cam.read()

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[CAMERA] Frame dropped. Switching to synthetic fallback.")
                self._init_synthetic()
                return self.synth_cam.read()
            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)
            return True, frame

        return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[CAMERA] Camera hardware released.")
