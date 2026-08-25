import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera import CameraManager
from src.background import BackgroundManager
from src.segmentation import Segmenter
from src.mask_processing import MaskProcessor
from src.ghost_effect import GhostEffectEngine
from src.blending import Blender
from src.hud import HUDRenderer
from utils.fps import FPSCounter
from utils.image_utils import save_screenshot
from config import OperationalMode, SegmentationMethod

print("[TEST_RUN] Verifying full pipeline with synthetic camera...")

camera = CameraManager(force_synthetic=True)
frame_w, frame_h = camera.get_resolution()
bg_manager = BackgroundManager(frame_w, frame_h)
segmenter = Segmenter(color_name="RED")
mask_processor = MaskProcessor()
ghost_engine = GhostEffectEngine()
blender = Blender()
hud = HUDRenderer(frame_w, frame_h)
fps_counter = FPSCounter()

# Simulate 35 frames (5 for background capture, 30 for processing live effects)
for i in range(35):
    ret, frame = camera.read()
    assert ret and frame is not None
    fps = fps_counter.update()

    if not bg_manager.is_ready():
        bg_manager.capture_frame(frame)
        continue

    bg_frame = bg_manager.get_background(frame)
    raw_mask = segmenter.segment(frame)
    cleaned_mask = mask_processor.clean_mask(raw_mask)
    soft_mask = mask_processor.get_soft_mask(cleaned_mask)

    ghost_layer = ghost_engine.update_motion_trail(frame, cleaned_mask)
    aura_layer = ghost_engine.generate_glowing_aura(cleaned_mask)

    composite = blender.blend(
        mode=OperationalMode.GHOST,
        live_frame=frame,
        bg_frame=bg_frame,
        mask=cleaned_mask,
        soft_mask=soft_mask,
        ghost_layer=ghost_layer,
        aura_layer=aura_layer
    )

    final_frame = hud.render(
        frame=composite,
        mode=OperationalMode.GHOST,
        method=segmenter.method,
        color_name="RED",
        opacity=0.5,
        fps=fps,
        is_synthetic=True
    )

# Save sample screenshot
shot_path = save_screenshot(final_frame, prefix="verification_synthetic_demo")
print(f"[TEST_RUN] Pipeline verification completed successfully! Sample image saved to: {shot_path}")
camera.release()
