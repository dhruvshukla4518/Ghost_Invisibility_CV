import os
import cv2
import time
import numpy as np
from typing import Tuple, Optional
from config import SCREENSHOTS_DIR, VIDEOS_DIR

def resize_keep_aspect(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resizes an image maintaining aspect ratio and pads to fit target resolution exactly.
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded black background canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def save_screenshot(frame: np.ndarray, prefix: str = "ghost_cv") -> str:
    """
    Saves current frame to output screenshots directory with timestamp filename.
    """
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(SCREENSHOTS_DIR, filename)

    cv2.imwrite(filepath, frame)
    print(f"[IMAGE_UTILS] Screenshot saved: {filepath}")
    return filepath

class VideoRecorder:
    """
    Helper to record real-time video output to MP4 files using OpenCV VideoWriter.
    """
    def __init__(self, width: int, height: int, fps: float = 30.0, prefix: str = "ghost_cv_rec"):
        self.width = width
        self.height = height
        self.fps = fps
        self.prefix = prefix
        self.writer: Optional[cv2.VideoWriter] = None
        self.filepath: Optional[str] = None
        self.is_recording: bool = False

    def start(self) -> str:
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}.mp4"
        self.filepath = os.path.join(VIDEOS_DIR, filename)

        # Try mp4v codec, fallback to MJPG / XVID if needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))

        if not self.writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.filepath = self.filepath.replace('.mp4', '.avi')
            self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))

        self.is_recording = True
        print(f"[VIDEO_RECORDER] Recording started: {self.filepath}")
        return self.filepath

    def write_frame(self, frame: np.ndarray):
        if self.is_recording and self.writer is not None:
            # Ensure frame matches target size
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def stop(self) -> Optional[str]:
        if self.is_recording and self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            print(f"[VIDEO_RECORDER] Recording saved & stopped: {self.filepath}")
            return self.filepath
        return None
