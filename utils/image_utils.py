"""
Image and output file management utilities.
"""

import datetime
from pathlib import Path
import cv2
import numpy as np
import requests

import config


def save_screenshot(frame: np.ndarray, output_dir: Path = config.OUTPUT_SCREENSHOTS_DIR) -> str:
    """
    Saves the provided image frame with a timestamped filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"ghost_capture_{timestamp}.png"
    filepath = output_dir / filename
    
    success = cv2.imwrite(str(filepath), frame)
    if success:
        print(f"[INFO] Screenshot saved to: {filepath}")
        return str(filepath)
    else:
        print(f"[ERROR] Failed to save screenshot to: {filepath}")
        return ""


class VideoRecorder:
    """
    Manages real-time video recording to disk.
    """

    def __init__(self, output_dir: Path = config.OUTPUT_VIDEOS_DIR, fps: int = config.FPS):
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.is_recording = False
        self.output_path = ""

    def start(self, frame_width: int, frame_height: int):
        """
        Initializes video writer for recording.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ghost_video_{timestamp}.mp4"
        self.output_path = str(self.output_dir / filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (frame_width, frame_height))
        self.is_recording = True
        print(f"[INFO] Video recording started: {self.output_path}")

    def write(self, frame: np.ndarray):
        """
        Writes a single frame to the video file.
        """
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        """
        Stops recording and releases video writer resources.
        """
        if self.is_recording and self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            print(f"[INFO] Video recording saved to: {self.output_path}")


def download_file_if_missing(url: str, destination: Path) -> bool:
    """
    Downloads a remote file if it does not exist locally.
    """
    if destination.exists():
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading model file from {url} to {destination}...")
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Download completed: {destination}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not download file from {url}: {e}")
        return False
