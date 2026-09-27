import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blending import Blender
from config import OperationalMode

def test_blender_cloak_mode():
    blender = Blender(opacity=0.0) # Full invisibility cloak

    # Live frame: Red (BGR: 0, 0, 255)
    live_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    live_frame[:, :] = (0, 0, 255)

    # BG frame: Blue (BGR: 255, 0, 0)
    bg_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bg_frame[:, :] = (255, 0, 0)

    # Full mask (100x100)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    soft_mask = np.ones((100, 100), dtype=np.float32)

    composite = blender.blend(
        mode=OperationalMode.CLOAK,
        live_frame=live_frame,
        bg_frame=bg_frame,
        mask=mask,
        soft_mask=soft_mask
    )

    assert composite.shape == (100, 100, 3)
    # Inside masked region with opacity=0.0, output should match bg_frame (Blue)
    assert np.array_equal(composite[50, 50], [255, 0, 0])

def test_blender_normal_mode():
    blender = Blender()

    live_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    live_frame[:, :] = (0, 255, 0) # Green
    bg_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    soft_mask = np.zeros((50, 50), dtype=np.float32)

    composite = blender.blend(
        mode=OperationalMode.NORMAL,
        live_frame=live_frame,
        bg_frame=bg_frame,
        mask=mask,
        soft_mask=soft_mask
    )

    assert np.array_equal(composite, live_frame)
