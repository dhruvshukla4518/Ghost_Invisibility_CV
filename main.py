import sys
import os
import cv2
import time
import argparse
import numpy as np

# Ensure project directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OperationalMode, SegmentationMethod, DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, DEFAULT_HSV_COLOR,
    DEFAULT_SHOW_HUD, WINDOW_TITLE, KEYS
)
from src.camera import CameraManager
from src.background import BackgroundManager
from src.segmentation import Segmenter
from src.mask_processing import MaskProcessor
from src.ghost_effect import GhostEffectEngine
from src.blending import Blender
from src.hand_tracking import HandGestureTracker
from src.hud import HUDRenderer
from utils.fps import FPSCounter
from utils.image_utils import save_screenshot, VideoRecorder

def parse_args():
    parser = argparse.ArgumentParser(description="Ghost Invisibility CV - AI Vision System")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_ID, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Frame height (default: 720)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic animated camera mode for testing without webcam")
    parser.add_argument("--mode", type=str, default="CLOAK", choices=["CLOAK", "GHOST", "SWAP", "NORMAL"], help="Initial operational mode")
    parser.add_argument("--color", type=str, default=DEFAULT_HSV_COLOR, help="Initial HSV cloak color preset (RED, GREEN, BLUE, etc.)")
    parser.add_argument("--no-hand-tracking", action="store_true", help="Disable hand gesture tracking")
    parser.add_argument("--background", type=str, default="", help="Path to static custom background image")
    parser.add_argument("--hud", action="store_true", default=DEFAULT_SHOW_HUD, help="Enable Sci-Fi HUD overlay (hidden by default)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print("      GHOST INVISIBILITY CV - REAL-TIME VISION SYSTEM")
    print("=" * 65)

    # 1. Initialize Camera
    camera = CameraManager(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        force_synthetic=args.synthetic
    )
    frame_w, frame_h = camera.get_resolution()

    # 2. Initialize Subsystems
    bg_manager = BackgroundManager(frame_w, frame_h)
    segmenter = Segmenter(method=SegmentationMethod.HSV_COLOR, color_name=args.color)
    mask_processor = MaskProcessor()
    ghost_engine = GhostEffectEngine()
    blender = Blender()
    hud = HUDRenderer(frame_w, frame_h, show_hud=args.hud)
    fps_counter = FPSCounter()
    recorder = VideoRecorder(frame_w, frame_h)

    hand_tracker = None
    if not args.no_hand_tracking:
        hand_tracker = HandGestureTracker()

    # Parse initial mode
    active_mode = OperationalMode[args.mode.upper()]

    # Load custom static background if provided
    if args.background and os.path.exists(args.background):
        bg_manager.load_static_background(args.background)

    # Create OpenCV window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, frame_w, frame_h)

    print("\n[CONTROLS]")
    print("  'b' - Capture / Recapture Background Frame (Step out of frame first!)")
    print("  'm' - Toggle Engine (HSV Color Cloak <-> MediaPipe AI Person Segmenter)")
    print("  'c' - Cycle Operational Mode (CLOAK -> GHOST -> SWAP -> NORMAL)")
    print("  'k' - Cycle HSV Cloak Color (RED -> GREEN -> BLUE -> CYAN -> MAGENTA -> YELLOW)")
    print("  'g' - Cycle Spectral Ghost Colormap (BONE -> OCEAN -> JET -> PLASMA -> CYBERPUNK)")
    print("  'h' - Toggle Sci-Fi HUD Overlay (Default: OFF)")
    print("  's' - Save Timestamped Screenshot")
    print("  'r' - Toggle MP4 Video Recording")
    print("  '+' / '-' - Adjust Transparency / Opacity")
    print("  'q' or ESC - Exit Application\n")

    bg_capturing = True
    bg_start_notice = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[MAIN] Error: Failed to retrieve frame.")
                break

            fps = fps_counter.update()

            # Automatic initial background capture sequence (if no custom background loaded)
            if bg_capturing and not bg_manager.is_ready():
                captured = bg_manager.capture_frame(frame)

                # Show initialization progress on screen
                init_canvas = frame.copy()
                progress_pct = int((bg_manager.captured_count / 30) * 100)
                cv2.putText(init_canvas, "CAPTURING BACKGROUND... PLEASE STEP OUT OF FRAME",
                            (frame_w // 2 - 350, frame_h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(init_canvas, (frame_w // 2 - 200, frame_h // 2 + 20),
                              (frame_w // 2 - 200 + progress_pct * 4, frame_h // 2 + 40), (0, 255, 255), -1)
                cv2.rectangle(init_canvas, (frame_w // 2 - 200, frame_h // 2 + 20),
                              (frame_w // 2 + 200, frame_h // 2 + 40), (255, 255, 255), 2)

                cv2.imshow(WINDOW_TITLE, init_canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in KEYS["QUIT"]:
                    break
                continue
            else:
                bg_capturing = False

            # Get background frame
            bg_frame = bg_manager.get_background(fallback_frame=frame)

            # Process Hand Tracking Gestures
            gesture_info = None
            if hand_tracker is not None:
                gesture_info = hand_tracker.process(frame)
                if gesture_info['detected']:
                    g = gesture_info['gesture']
                    if g == 'PINCH':
                        blender.set_opacity(gesture_info['pinch_dist'])
                    elif g == 'OPEN_PALM' and active_mode != OperationalMode.CLOAK:
                        active_mode = OperationalMode.CLOAK
                        print("[GESTURE] Switched mode to CLOAK")
                    elif g == 'FIST' and (time.time() - bg_start_notice > 3.0):
                        bg_manager.reset()
                        bg_capturing = True
                        bg_start_notice = time.time()
                        print("[GESTURE] Triggered Background Recapture")

            # 1. Segmentation
            raw_mask = segmenter.segment(frame)

            # 2. Mask Processing & Morphological Cleaning
            cleaned_mask = mask_processor.clean_mask(raw_mask)
            soft_mask = mask_processor.get_soft_mask(cleaned_mask)

            # 3. Compute Ghost Visual Effects (Colormaps, Aura, Motion Trails)
            ghost_layer = None
            aura_layer = None
            if active_mode == OperationalMode.GHOST:
                colormapped_ghost = ghost_engine.apply_spectral_colormap(frame, cleaned_mask)
                ghost_layer = ghost_engine.update_motion_trail(colormapped_ghost, cleaned_mask)
                aura_layer = ghost_engine.generate_glowing_aura(cleaned_mask)

            # 4. Multi-Layer Image Blending
            composite = blender.blend(
                mode=active_mode,
                live_frame=frame,
                bg_frame=bg_frame,
                mask=cleaned_mask,
                soft_mask=soft_mask,
                ghost_layer=ghost_layer,
                aura_layer=aura_layer
            )

            # Draw Hand Landmarks if active
            if hand_tracker is not None and gesture_info is not None and hud.show_hud:
                hand_tracker.draw_landmarks(composite, gesture_info)

            # 5. Render Sci-Fi HUD Overlay
            final_frame = hud.render(
                frame=composite,
                mode=active_mode,
                method=segmenter.method,
                color_name=segmenter.hsv_segmenter.color_name,
                opacity=blender.opacity,
                fps=fps,
                is_recording=recorder.is_recording,
                is_synthetic=camera.is_synthetic,
                gesture_info=gesture_info
            )

            # Write frame to video recorder if active
            if recorder.is_recording:
                recorder.write_frame(final_frame)

            # Display composite window
            cv2.imshow(WINDOW_TITLE, final_frame)

            # Handle Key Events
            key = cv2.waitKey(1) & 0xFF
            if key in KEYS["QUIT"]:
                print("[MAIN] Exiting application...")
                break
            elif key in KEYS["CAPTURE_BG"]:
                bg_manager.reset()
                bg_capturing = True
                print("[MAIN] Resetting background. Recapturing...")
            elif key in KEYS["CYCLE_MODE"]:
                modes = list(OperationalMode)
                idx = (modes.index(active_mode) + 1) % len(modes)
                active_mode = modes[idx]
                print(f"[MAIN] Operational Mode changed to {active_mode.value}")
            elif key in KEYS["CYCLE_ENGINE"]:
                methods = list(SegmentationMethod)
                idx = (methods.index(segmenter.method) + 1) % len(methods)
                segmenter.set_method(methods[idx])
                print(f"[MAIN] Segmentation Engine changed to {segmenter.method.value}")
            elif key in KEYS["CYCLE_COLOR"]:
                colors = segmenter.hsv_segmenter.get_available_colors()
                curr = segmenter.hsv_segmenter.color_name
                idx = (colors.index(curr) + 1) % len(colors)
                segmenter.set_hsv_color(colors[idx])
            elif key in KEYS["TOGGLE_GHOST"]:
                new_cmap = ghost_engine.cycle_colormap()
                if active_mode != OperationalMode.GHOST:
                    active_mode = OperationalMode.GHOST
            elif key in KEYS["TOGGLE_HUD"]:
                is_shown = hud.toggle()
                status = "ENABLED" if is_shown else "DISABLED"
                print(f"[MAIN] Sci-Fi HUD overlay {status}")
            elif key in KEYS["SCREENSHOT"]:
                save_screenshot(final_frame, prefix=f"ghost_{active_mode.value.lower()}")
            elif key in KEYS["RECORD"]:
                if recorder.is_recording:
                    recorder.stop()
                else:
                    recorder.start()
            elif key in KEYS["OPACITY_UP"]:
                blender.set_opacity(blender.opacity + 0.05)
                print(f"[MAIN] Opacity increased to {blender.opacity:.2f}")
            elif key in KEYS["OPACITY_DOWN"]:
                blender.set_opacity(blender.opacity - 0.05)
                print(f"[MAIN] Opacity decreased to {blender.opacity:.2f}")

    finally:
        # Cleanup
        if recorder.is_recording:
            recorder.stop()
        if hand_tracker is not None:
            hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        print("[MAIN] Application terminated cleanly.")

if __name__ == "__main__":
    main()
