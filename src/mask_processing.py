"""
Mask processing and refinement module.
"""

import cv2
import numpy as np
import config


class MaskProcessor:
    """
    Refines raw probability segmentation masks using morphological operations,
    boundary dilation expansion, and Gaussian edge softening.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        morph_kernel_size: tuple = (7, 7),
        blur_kernel_size: tuple = (21, 21),
        erode_iters: int = 0,
        dilate_iters: int = 3,
    ):
        self.threshold = threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
        self.blur_kernel_size = blur_kernel_size
        self.erode_iters = erode_iters
        self.dilate_iters = dilate_iters

    def process(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Processes float raw_mask [0.0, 1.0] and returns a refined float mask [0.0, 1.0].
        """
        if raw_mask is None or raw_mask.size == 0:
            return np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.float32)

        h, w = raw_mask.shape[:2]

        # 1. Binarize raw mask at low threshold to capture all hand & finger pixels
        binary_uint8 = (raw_mask > self.threshold).astype(np.uint8) * 255

        # 2. Morphological Closing (Fill gaps inside hands, fingers, hair, torso)
        closed = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # 3. Morphological Opening (Clean small background specks)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # 4. Contour Area Filtering (Keep substantial human contours)
        filtered = self._filter_contours(opened, min_area=300)

        # 5. Boundary Dilation Expansion (Expands mask slightly so no hand/finger fringes remain)
        if self.dilate_iters > 0:
            filtered = cv2.dilate(filtered, self.kernel, iterations=self.dilate_iters)

        # 6. Gaussian Blur Edge Softening / Feathering
        k_w = self.blur_kernel_size[0] if self.blur_kernel_size[0] % 2 == 1 else self.blur_kernel_size[0] + 1
        k_h = self.blur_kernel_size[1] if self.blur_kernel_size[1] % 2 == 1 else self.blur_kernel_size[1] + 1

        blurred = cv2.GaussianBlur(filtered.astype(np.float32), (k_w, k_h), sigmaX=0)

        # Normalize back to float32 range [0.0, 1.0]
        smooth_mask = np.clip(blurred / 255.0, 0.0, 1.0).astype(np.float32)

        return smooth_mask

    def _filter_contours(self, binary_mask: np.ndarray, min_area: int = 300) -> np.ndarray:
        """
        Finds contours in binary mask and retains contours with area >= min_area.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_mask = np.zeros_like(binary_mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(output_mask, [cnt], -1, 255, -1)

        return output_mask
