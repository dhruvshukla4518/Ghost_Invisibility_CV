import time

class FPSCounter:
    """
    Computes smoothed and instantaneous Frames Per Second (FPS).
    """
    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.prev_time = time.time()
        self.curr_fps = 0.0
        self.smoothed_fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

    def update(self) -> float:
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time
        self.frame_count += 1

        if delta > 0:
            self.curr_fps = 1.0 / delta
            if self.smoothed_fps == 0.0:
                self.smoothed_fps = self.curr_fps
            else:
                self.smoothed_fps = (self.smoothing_factor * self.smoothed_fps) + ((1.0 - self.smoothing_factor) * self.curr_fps)

        return self.smoothed_fps

    def get_fps(self) -> float:
        return self.smoothed_fps

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
