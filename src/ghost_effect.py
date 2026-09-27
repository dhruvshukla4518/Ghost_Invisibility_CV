import cv2
import numpy as np
from collections import deque
from typing import Optional, List
from config import MAX_MOTION_TRAIL_FRAMES, TRAIL_DECAY_FACTOR, AURA_GLOW_RADIUS, AURA_COLOR, COLORMAP_CHOICES, DEFAULT_COLORMAP

class GhostEffectEngine:
    """
    Engine for creating paranormal/ghost visual effects including motion trails,
    glowing spectral auras, colormap transformations, and scanline glitches.
    """
    def __init__(self, max_trail: int = MAX_MOTION_TRAIL_FRAMES,
                 colormap_name: str = DEFAULT_COLORMAP):
        self.max_trail = max_trail
        self.trail_buffer = deque(maxlen=max_trail)
        self.colormap_name = colormap_name.upper()

        self.colormaps = {
            "BONE": cv2.COLORMAP_BONE,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "JET": cv2.COLORMAP_JET,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "CYBERPUNK": cv2.COLORMAP_COOL
        }

    def cycle_colormap(self) -> str:
        names = list(self.colormaps.keys())
        idx = (names.index(self.colormap_name) + 1) % len(names)
        self.colormap_name = names[idx]
        print(f"[GHOST_EFFECT] Active spectral colormap: {self.colormap_name}")
        return self.colormap_name

    def apply_spectral_colormap(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies chosen OpenCV colormap inside the masked subject area.
        """
        cmap_id = self.colormaps.get(self.colormap_name, cv2.COLORMAP_BONE)

        # Convert BGR to Grayscale then apply colormap
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cmap_id)

        # Blend original frame and colormap frame inside mask
        blended_ghost = cv2.addWeighted(frame, 0.4, colored, 0.6, 0)
        return blended_ghost

    def generate_glowing_aura(self, mask: np.ndarray,
                              aura_color: tuple = AURA_COLOR,
                              glow_radius: int = AURA_GLOW_RADIUS) -> np.ndarray:
        """
        Extracts edges of binary mask and creates a glowing spectral neon aura frame.
        """
        if glow_radius <= 0:
            return np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Canny edge detection on mask
        edges = cv2.Canny(mask, 100, 200)

        # Dilate edges to expand aura boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glow_radius, glow_radius))
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        # Gaussian blur dilated edges to create soft glow gradient
        ksize = glow_radius * 2 + 1
        blurred_glow = cv2.GaussianBlur(dilated_edges, (ksize, ksize), 0)

        # Colorize the glow
        glow_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        norm_glow = blurred_glow.astype(np.float32) / 255.0

        for i in range(3): # BGR channels
            glow_canvas[:, :, i] = (norm_glow * aura_color[i]).astype(np.uint8)

        return glow_canvas

    def update_motion_trail(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Adds current subject frame to motion trail queue and computes motion trail layer.
        """
        # Isolate subject from current frame
        subject_isolated = cv2.bitwise_and(frame, frame, mask=mask)
        self.trail_buffer.append(subject_isolated.copy())

        trail_canvas = np.zeros_like(frame, dtype=np.float32)
        total_weight = 0.0

        # Accumulate past frames in queue with exponential decay
        for idx, past_frame in enumerate(reversed(self.trail_buffer)):
            weight = (TRAIL_DECAY_FACTOR ** idx)
            trail_canvas += past_frame.astype(np.float32) * weight
            total_weight += weight

        if total_weight > 0:
            trail_canvas /= total_weight

        return trail_canvas.astype(np.uint8)

    def apply_scanline_glitch(self, frame: np.ndarray, line_spacing: int = 6) -> np.ndarray:
        """
        Applies subtle horizontal scanlines across frame for futuristic ghost aesthetic.
        """
        glitch = frame.copy()
        glitch[::line_spacing, :, :] = (glitch[::line_spacing, :, :] * 0.7).astype(np.uint8)
        return glitch

    def reset(self):
        self.trail_buffer.clear()
