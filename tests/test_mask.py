"""
Unit tests for mask processing and refinement module.
"""

import pytest
import numpy as np
from src.mask_processing import MaskProcessor


def test_mask_processor_smoothing():
    processor = MaskProcessor(threshold=0.5)
    raw_mask = np.zeros((100, 100), dtype=np.float32)
    # Add a square person mask region in the center
    raw_mask[30:70, 30:70] = 0.95
    # Add speckle noise
    raw_mask[10, 10] = 0.9

    refined_mask = processor.process(raw_mask)

    assert refined_mask is not None
    assert refined_mask.shape == (100, 100)
    assert refined_mask.dtype == np.float32
    assert np.min(refined_mask) >= 0.0
    assert np.max(refined_mask) <= 1.0

    # Speckle noise at (10, 10) should be removed by opening
    assert refined_mask[10, 10] < 0.1


def test_mask_processor_empty_mask():
    processor = MaskProcessor()
    empty_mask = np.array([], dtype=np.float32)

    refined = processor.process(empty_mask)
    assert refined is not None
    assert np.all(refined == 0.0)
