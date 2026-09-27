import cv2
import numpy as np
from typing import Tuple, Optional
from config import OperationalMode, DEFAULT_GHOST_OPACITY

class Blender:
    """
    Seamless multi-layer alpha compositor for blending live camera feed,
    background frame, ghost visual effects, and soft binary/alpha masks.
    """
    def __init__(self, opacity: float = DEFAULT_GHOST_OPACITY):
        self.opacity = opacity # 0.0 = completely invisible background, 1.0 = fully opaque foreground

    def set_opacity(self, opacity: float):
        self.opacity = max(0.0, min(1.0, opacity))

    def blend(self, mode: OperationalMode,
              live_frame: np.ndarray,
              bg_frame: np.ndarray,
              mask: np.ndarray,
              soft_mask: np.ndarray,
              ghost_layer: Optional[np.ndarray] = None,
              aura_layer: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main composition function that returns final composite image depending on active OperationalMode.
        """
        # Ensure dimensions match live frame
        h, w = live_frame.shape[:2]
        if bg_frame.shape[:2] != (h, w):
            bg_frame = cv2.resize(bg_frame, (w, h))

        if mode == OperationalMode.NORMAL:
            return live_frame.copy()

        elif mode == OperationalMode.CLOAK:
            # Full Invisibility Cloak
            # Masked area = Background Frame, Unmasked area = Live Frame
            # Smooth 3-channel alpha blending using soft_mask
            alpha_3d = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)

            # Invisibility cloak blends bg_frame into masked region
            # opacity parameter allows user to adjust cloak transparency!
            effective_bg = cv2.addWeighted(bg_frame, 1.0 - self.opacity, live_frame, self.opacity, 0)

            composite = (effective_bg * alpha_3d + live_frame * (1.0 - alpha_3d)).astype(np.uint8)
            return composite

        elif mode == OperationalMode.GHOST:
            # Ghost Phantom Mode
            # Blends live frame, ghost spectral layer, motion trail, aura, and background
            alpha_3d = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)

            # Base composition inside subject mask: blend live subject with ghost layer (colormap / motion trail)
            if ghost_layer is not None:
                phantom_subject = cv2.addWeighted(live_frame, 0.4, ghost_layer, 0.6, 0)
            else:
                phantom_subject = live_frame.copy()

            # Blend phantom subject with background frame according to self.opacity
            blended_phantom = cv2.addWeighted(phantom_subject, self.opacity, bg_frame, 1.0 - self.opacity, 0)

            # Compose subject region over background
            composite = (blended_phantom * alpha_3d + bg_frame * (1.0 - alpha_3d)).astype(np.uint8)

            # Add glowing spectral aura if available
            if aura_layer is not None:
                composite = cv2.add(composite, aura_layer)

            return composite

        elif mode == OperationalMode.SWAP:
            # Background Swap (Keep subject, replace rest of scene with background frame)
            alpha_3d = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)
            composite = (live_frame * alpha_3d + bg_frame * (1.0 - alpha_3d)).astype(np.uint8)
            return composite

        return live_frame.copy()
