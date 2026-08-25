"""
Real-time FPS tracking utility.
"""

import time
from collections import deque


class FPSCounter:
    """
    Computes smooth real-time frames per second (FPS) using a sliding window.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        """
        Call once per frame loop to update time delta and calculate FPS.
        """
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 0:
            self.frame_times.append(dt)

        if len(self.frame_times) > 0:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            self.fps = 0.0

        return self.fps

    def get_fps(self) -> float:
        """
        Returns the current calculated FPS.
        """
        return self.fps
