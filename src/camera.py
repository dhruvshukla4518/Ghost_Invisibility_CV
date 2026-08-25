"""
Webcam capture manager with synthetic fallback capability.
"""

import time
import cv2
import numpy as np
import config


class CameraManager:
    """
    Manages webcam acquisition and frame generation.
    Supports synthetic frame generator for testing/demo without webcam.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        force_synthetic: bool = False,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.is_synthetic = force_synthetic
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()

        if not self.is_synthetic:
            self._initialize_camera()

        if self.cap is None or not self.cap.isOpened():
            print("[WARNING] Hardware camera unavailable. Switching to Synthetic Camera Feed.")
            self.is_synthetic = True

    def _initialize_camera(self):
        """
        Attempts to open hardware webcam device.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                # Read actual dimensions
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_w > 0 and actual_h > 0:
                    self.width = actual_w
                    self.height = actual_h
                print(f"[INFO] Camera initialized successfully ({self.width}x{self.height}).")
            else:
                print("[ERROR] Unable to access webcam. Please check camera permissions.")
        except Exception as e:
            print(f"[ERROR] Exception during camera initialization: {e}")
            self.cap = None

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Reads next frame from webcam or generates synthetic frame.
        Returns (success, frame_bgr).
        """
        if not self.is_synthetic and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
            else:
                print("[WARNING] Invalid or empty frame received from camera.")
                return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            return True, self._generate_synthetic_frame()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generates a dynamic synthetic frame simulating room background + moving person.
        """
        self.frame_count += 1
        t = time.time() - self.start_time

        # Create room background (gradient wall + floor)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            ratio = y / float(self.height)
            frame[y, :, 0] = int(180 - ratio * 60)  # B
            frame[y, :, 1] = int(140 - ratio * 40)  # G
            frame[y, :, 2] = int(100 + ratio * 80)  # R

        # Draw static room furniture/grid lines
        cv2.rectangle(frame, (50, 200), (180, 450), (60, 40, 30), -1)
        cv2.rectangle(frame, (460, 150), (590, 450), (40, 60, 80), -1)

        # Draw moving person (head, body, arms) moving left-right
        offset_x = int(np.sin(t * 1.5) * 80)
        center_x = self.width // 2 + offset_x
        center_y = self.height // 2 + 30

        # Person shirt color
        shirt_color = (40, 120, 220)  # Blue/Orange mix
        head_color = (180, 200, 220)  # Skin tone tint

        # Body (ellipse/capsule)
        cv2.ellipse(frame, (center_x, center_y + 40), (45, 90), 0, 0, 360, shirt_color, -1)
        # Head (circle)
        cv2.circle(frame, (center_x, center_y - 65), 35, head_color, -1)

        # Draw a moving hand raised for gesture demo (toggling between open palm & fist)
        hand_x = center_x + 65 + int(np.sin(t * 3.0) * 15)
        hand_y = center_y - 40 + int(np.cos(t * 3.0) * 15)
        cv2.circle(frame, (hand_x, hand_y), 15, head_color, -1)

        return frame

    def release(self):
        """
        Releases camera resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[INFO] Camera released.")
