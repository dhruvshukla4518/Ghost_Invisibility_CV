"""
Unit tests for person segmentation module.
"""

import pytest
import numpy as np
from src.segmentation import PersonSegmenter


def test_segmenter_initialization():
    segmenter = PersonSegmenter(threshold=0.5)
    assert segmenter is not None
    assert 0.0 <= segmenter.threshold <= 1.0


def test_segmentation_output_shape_and_range():
    segmenter = PersonSegmenter(threshold=0.5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bg = np.ones((480, 640, 3), dtype=np.uint8) * 100

    mask = segmenter.segment(dummy_frame, dummy_bg)

    assert mask is not None
    assert mask.shape == (480, 640)
    assert mask.dtype == np.float32
    assert np.min(mask) >= 0.0
    assert np.max(mask) <= 1.0


def test_segmentation_empty_frame():
    segmenter = PersonSegmenter(threshold=0.5)
    empty_frame = np.array([], dtype=np.uint8)

    mask = segmenter.segment(empty_frame)
    assert mask is not None
    assert np.all(mask == 0.0)
