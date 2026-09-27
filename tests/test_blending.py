"""
Unit tests for alpha blending and camouflage engine.
"""

import pytest
import numpy as np
from src.blending import Blender


def test_blend_ghost_alpha_zero():
    blender = Blender(default_alpha=0.0)
    fg = np.ones((50, 50, 3), dtype=np.uint8) * 200  # Light Gray
    bg = np.ones((50, 50, 3), dtype=np.uint8) * 50   # Dark Gray
    mask = np.ones((50, 50), dtype=np.float32)       # Full person mask

    # Alpha 0.0 means 100% background inside person region
    result = blender.blend_ghost(fg, bg, mask, alpha=0.0)

    assert result is not None
    assert result.shape == (50, 50, 3)
    assert np.allclose(result, 50, atol=2)


def test_blend_ghost_alpha_one():
    blender = Blender(default_alpha=1.0)
    fg = np.ones((50, 50, 3), dtype=np.uint8) * 200
    bg = np.ones((50, 50, 3), dtype=np.uint8) * 50
    mask = np.ones((50, 50), dtype=np.float32)

    # Alpha 1.0 means 100% foreground inside person region
    result = blender.blend_ghost(fg, bg, mask, alpha=1.0)

    assert result is not None
    assert np.allclose(result, 200, atol=2)


def test_blend_invisible():
    blender = Blender()
    fg = np.ones((50, 50, 3), dtype=np.uint8) * 200
    bg = np.ones((50, 50, 3), dtype=np.uint8) * 50
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10:40, 10:40] = 1.0  # Center square is person

    result = blender.blend_invisible(fg, bg, mask)

    # Center square should be background (50)
    assert np.allclose(result[25, 25], [50, 50, 50], atol=2)
    # Outside should be foreground (200)
    assert np.allclose(result[5, 5], [200, 200, 200], atol=2)


def test_blend_camouflage():
    blender = Blender()
    fg = np.ones((50, 50, 3), dtype=np.uint8) * 200
    bg = np.ones((50, 50, 3), dtype=np.uint8) * 50
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10:40, 10:40] = 1.0

    result = blender.blend_camouflage(fg, bg, mask, distortion=5.0)

    assert result is not None
    assert result.shape == (50, 50, 3)
