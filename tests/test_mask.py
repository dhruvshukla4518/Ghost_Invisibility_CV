import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mask_processing import MaskProcessor

def test_mask_cleaning_noise_removal():
    processor = MaskProcessor(kernel_size=5, min_area=100)
    # Create mask with single small noise spot (5x5 pixels = area 25 < min_area 100)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[10:15, 10:15] = 255

    cleaned = processor.clean_mask(raw_mask)
    # Small noise spot should be removed (all zeros)
    assert np.max(cleaned) == 0

def test_mask_cleaning_valid_contour():
    processor = MaskProcessor(kernel_size=5, min_area=100)
    # Create valid large box contour (50x50 pixels = area 2500 > min_area 100)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[20:70, 20:70] = 255

    cleaned = processor.clean_mask(raw_mask)
    assert np.max(cleaned) == 255
    assert np.count_nonzero(cleaned) > 2000

def test_soft_mask_generation():
    processor = MaskProcessor(blur_sigma=3.0)
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[20:80, 20:80] = 255

    soft_mask = processor.get_soft_mask(raw_mask)
    assert soft_mask.shape == (100, 100)
    assert soft_mask.dtype == np.float32
    assert 0.0 <= soft_mask.min() <= soft_mask.max() <= 1.0
