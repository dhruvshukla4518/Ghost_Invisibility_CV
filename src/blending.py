"""
Alpha blending engine for ghost and invisibility compositing.
"""

import cv2
import numpy as np
import config


class Blender:
    """
    Combines foreground camera frame, captured background frame,
    and refined person mask using alpha compositing equations.
    """

    def __init__(self, default_alpha: float = config.DEFAULT_ALPHA):
        self.alpha = default_alpha

    def set_alpha(self, alpha: float):
        """
        Updates blending alpha factor in range [0.0, 1.0].
        """
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def get_alpha(self) -> float:
        return self.alpha

    def _ensure_matching_shape(self, bg: np.ndarray, target_fg: np.ndarray) -> np.ndarray:
        """
        Resizes background image if shape does not match target foreground frame.
        """
        if bg is None:
            return target_fg.copy()
        h, w = target_fg.shape[:2]
        if bg.shape[:2] != (h, w):
            return cv2.resize(bg, (w, h))
        return bg

    def blend_ghost(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        alpha: float = None,
    ) -> np.ndarray:
        """
        Performs ghost compositing:
        Inside person mask: blend foreground with background scaled by alpha.
        Outside person mask: show current foreground.
        """
        if alpha is None:
            alpha = self.alpha

        if background is None:
            return foreground.copy()

        background = self._ensure_matching_shape(background, foreground)

        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        fg_float = foreground.astype(np.float32)
        bg_float = background.astype(np.float32)

        # Ghost region blend
        ghost_region = alpha * fg_float + (1.0 - alpha) * bg_float

        # Composite ghost region inside person mask, keeping normal scene outside
        output_float = mask_3d * ghost_region + (1.0 - mask_3d) * fg_float

        return np.clip(output_float, 0, 255).astype(np.uint8)

    def blend_invisible(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Performs pure invisibility compositing:
        Inside person mask: replace 100% with clean background.
        Outside person mask: keep current live camera foreground.
        """
        if background is None:
            return foreground.copy()

        background = self._ensure_matching_shape(background, foreground)

        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        fg_float = foreground.astype(np.float32)
        bg_float = background.astype(np.float32)

        # Background inside person mask, live camera outside
        output_float = mask_3d * bg_float + (1.0 - mask_3d) * fg_float

        return np.clip(output_float, 0, 255).astype(np.uint8)
