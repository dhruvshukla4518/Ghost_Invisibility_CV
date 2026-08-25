"""
Ghost effect rendering engine with multi-mode visual effects and ghost trails.
"""

from collections import deque
import cv2
import numpy as np
import config
from src.blending import Blender


class GhostEffectEngine:
    """
    Renders ghost visual effects, custom color shaders, neon glow, glitch artifacts,
    and temporal ghost trails.
    """

    def __init__(self, blender: Blender = None):
        self.blender = blender if blender is not None else Blender()
        self.mode = config.DEFAULT_MODE
        self.intensity = config.DEFAULT_GHOST_INTENSITY
        self.ghost_trail_enabled = True
        self.trail_queue = deque(maxlen=config.GHOST_TRAIL_LENGTH)

    def set_mode(self, mode: str):
        """
        Updates active rendering mode.
        """
        if mode in [
            config.MODE_NORMAL,
            config.MODE_GHOST,
            config.MODE_INVISIBLE,
            config.MODE_NEON_GHOST,
            config.MODE_GLITCH,
        ]:
            self.mode = mode
            print(f"[INFO] Mode changed to: {self.mode}")

    def set_intensity(self, intensity: float):
        """
        Adjusts ghost intensity factor [0.0, 1.0].
        """
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self.blender.set_alpha(self.intensity)

    def render(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Renders the active effect onto the current frame.
        """
        if self.mode == config.MODE_NORMAL or background is None:
            return foreground.copy()

        if self.mode == config.MODE_INVISIBLE:
            return self.blender.blend_invisible(foreground, background, mask)

        if self.mode == config.MODE_GHOST:
            base_ghost = self.blender.blend_ghost(foreground, background, mask, alpha=self.intensity)
            if self.ghost_trail_enabled:
                return self._apply_ghost_trail(base_ghost, foreground, mask)
            return base_ghost

        if self.mode == config.MODE_NEON_GHOST:
            return self._render_neon_ghost(foreground, background, mask)

        if self.mode == config.MODE_GLITCH:
            return self._render_glitch_ghost(foreground, background, mask)

        return foreground.copy()

    def _apply_ghost_trail(self, base_output: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Composites past ghost frames with decaying alpha for a motion trail effect.
        """
        self.trail_queue.append(foreground.copy())

        if len(self.trail_queue) < 2:
            return base_output

        output = base_output.astype(np.float32)
        n = len(self.trail_queue)

        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        for i, past_frame in enumerate(reversed(self.trail_queue)):
            decay = 0.15 * (1.0 - i / float(n))
            trail_blend = past_frame.astype(np.float32) * mask_3d * decay
            output = cv2.add(output, trail_blend)

        return np.clip(output, 0, 255).astype(np.uint8)

    def _render_neon_ghost(self, foreground: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Renders a cyan/magenta chromatic neon outline around the ghost figure.
        """
        # Base ghost blend
        ghost_base = self.blender.blend_ghost(foreground, background, mask, alpha=self.intensity)

        # Detect edge boundaries of mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 50, 150)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)

        # Create cyan/magenta neon edge color map
        neon_layer = np.zeros_like(foreground)
        neon_layer[:, :, 0] = edges_dilated  # Cyan Blue
        neon_layer[:, :, 1] = edges_dilated  # Cyan Green
        neon_layer[:, :, 2] = (edges_dilated * 0.7).astype(np.uint8)  # Magenta/Pink accent

        # Add neon glow onto ghost frame
        result = cv2.addWeighted(ghost_base, 1.0, neon_layer, 0.8, 0)
        return result

    def _render_glitch_ghost(self, foreground: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies random digital displacement/glitch artifacts within the person region.
        """
        glitched_fg = foreground.copy()
        h, w = foreground.shape[:2]

        # Shift random horizontal scanlines
        num_glitches = np.random.randint(3, 10)
        for _ in range(num_glitches):
            y1 = np.random.randint(0, h - 20)
            height = np.random.randint(5, 20)
            shift = np.random.randint(-25, 25)

            glitched_fg[y1 : y1 + height, :] = np.roll(glitched_fg[y1 : y1 + height, :], shift, axis=1)

        return self.blender.blend_ghost(glitched_fg, background, mask, alpha=self.intensity)
