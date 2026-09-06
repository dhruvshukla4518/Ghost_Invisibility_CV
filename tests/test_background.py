"""
Unit tests for BackgroundManager and clean room background capture.
"""

import pytest
import numpy as np
from src.background import BackgroundManager


def test_background_capture_and_retrieve():
    bg_mgr = BackgroundManager()

    # Synthetic clean room frame
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 150
    bg_mgr.capture_background(frame)

    assert bg_mgr.has_background()
    bg = bg_mgr.get_background((100, 100))
    assert bg is not None
    assert bg.shape == (100, 100, 3)
    assert np.allclose(bg, 150)


def test_auto_capture_when_no_human():
    bg_mgr = BackgroundManager()

    frame = np.ones((80, 80, 3), dtype=np.uint8) * 200
    empty_mask = np.zeros((80, 80), dtype=np.float32)  # No human in frame

    # 5 frames required to average and capture
    for _ in range(4):
        res = bg_mgr.auto_capture_if_clear(frame, empty_mask)
        assert res is False

    res = bg_mgr.auto_capture_if_clear(frame, empty_mask)
    assert res is True
    assert bg_mgr.has_background()


def test_cycle_presets():
    bg_mgr = BackgroundManager()
    initial_idx = bg_mgr.current_preset_idx

    preset_name = bg_mgr.cycle_preset((60, 60))
    assert preset_name in bg_mgr.PRESETS
    assert bg_mgr.is_preset_active is True
    assert bg_mgr.has_background() is True
