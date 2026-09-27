import pytest
import numpy as np
import cv2
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.segmentation import HSVColorSegmenter, Segmenter
from config import SegmentationMethod

def test_hsv_color_segmentation_red():
    segmenter = HSVColorSegmenter("RED")
    # Create pure red frame (BGR: 0, 0, 255)
    red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    red_frame[:, :] = (0, 0, 255)

    mask = segmenter.segment(red_frame)
    assert mask.shape == (100, 100)
    # Mask should be fully 255 (detected as red)
    assert np.mean(mask) > 250

def test_hsv_color_segmentation_green():
    segmenter = HSVColorSegmenter("GREEN")
    # Create pure green frame (BGR: 0, 255, 0)
    green_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    green_frame[:, :] = (0, 255, 0)

    mask = segmenter.segment(green_frame)
    assert mask.shape == (100, 100)
    assert np.mean(mask) > 250

def test_hsv_color_segmentation_blue():
    segmenter = HSVColorSegmenter("BLUE")
    # Create pure blue frame (BGR: 255, 0, 0)
    blue_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_frame[:, :] = (255, 0, 0)

    mask = segmenter.segment(blue_frame)
    assert mask.shape == (100, 100)
    assert np.mean(mask) > 250

def test_unified_segmenter_switch_method():
    seg = Segmenter(method=SegmentationMethod.HSV_COLOR, color_name="RED")
    assert seg.method == SegmentationMethod.HSV_COLOR

    seg.set_method(SegmentationMethod.MEDIAPIPE)
    assert seg.method == SegmentationMethod.MEDIAPIPE
