"""
Unit tests for dynamic BackgroundManager and real-time live inpainting.
"""

import pytest
import numpy as np
from src.background import BackgroundManager
import config


def test_live_background_update():
    bg_mgr = BackgroundManager(mode=config.BG_MODE_LIVE)

    # Frame of size 100x100
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
    # Person mask covering center 40x40 area
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0

    # In LIVE mode, update background
    live_bg = bg_mgr.update_live_background(frame, mask)

    assert live_bg is not None
    assert live_bg.shape == (100, 100, 3)

    # Outside the mask, background should match live frame exactly (120)
    assert np.allclose(live_bg[10, 10], [120, 120, 120], atol=5)


def test_live_background_mode_toggle():
    bg_mgr = BackgroundManager(mode=config.BG_MODE_LIVE)
    assert bg_mgr.mode == config.BG_MODE_LIVE

    toggled = bg_mgr.toggle_mode()
    assert toggled == config.BG_MODE_STATIC
    assert bg_mgr.mode == config.BG_MODE_STATIC

    toggled_again = bg_mgr.toggle_mode()
    assert toggled_again == config.BG_MODE_LIVE


def test_live_inpainting_reconstructs_masked_region():
    bg_mgr = BackgroundManager(mode=config.BG_MODE_LIVE)

    # Gradient or solid scene with a masked hole
    frame = np.ones((80, 80, 3), dtype=np.uint8) * 180
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[25:55, 25:55] = 1.0

    inpainted = bg_mgr._fast_inpaint(frame, mask)

    assert inpainted is not None
    assert inpainted.shape == (80, 80, 3)
    # The center hole should be smoothly filled with values near 180
    assert np.allclose(inpainted[40, 40], [180, 180, 180], atol=15)
