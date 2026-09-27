import cv2
import numpy as np
from typing import Tuple
from config import MORPH_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA, MIN_CONTOUR_AREA

class MaskProcessor:
    """
    Refines raw binary segmentation masks using morphological cleanup,
    noise filtering, contour filtering, and edge-softening blurs.
    """
    def __init__(self, kernel_size: int = MORPH_KERNEL_SIZE,
                 blur_sigma: float = GAUSSIAN_BLUR_SIGMA,
                 min_area: int = MIN_CONTOUR_AREA):
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        self.blur_sigma = blur_sigma
        self.min_area = min_area

    def clean_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological opening (noise removal) and closing (hole filling).
        Filters out small isolated noise contours.
        """
        # Morphological Opening (remove white noise pixels)
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)

        # Morphological Dilate/Closing (fill gaps inside mask)
        dilated = cv2.dilate(opened, self.kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # Filter contours by minimum area
        cleaned = self._filter_contours(closed)
        return cleaned

    def _filter_contours(self, mask: np.ndarray) -> np.ndarray:
        """
        Finds contours in binary mask and keeps only those above MIN_CONTOUR_AREA.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

        return filtered_mask

    def get_soft_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Generates a smooth float normalized alpha mask [0.0 - 1.0] with Gaussian blurred edges
        to avoid harsh pixelated edges during image blending.
        """
        if self.blur_sigma <= 0:
            return (binary_mask / 255.0).astype(np.float32)

        ksize = int(self.blur_sigma * 3)
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(binary_mask, (ksize, ksize), self.blur_sigma)
        soft_mask = (blurred / 255.0).astype(np.float32)
        return soft_mask
