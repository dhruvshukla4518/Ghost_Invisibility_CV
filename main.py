"""
Real-Time Ghost Invisibility System - Main Application Entry Point.
Featuring Clean Room Background Capture & Real-Time Invisibility.
"""

import sys
import argparse
import cv2
import numpy as np

import config
from src.camera import CameraManager
from src.segmentation import PersonSegmenter
from src.background import BackgroundManager
from src.mask_processing import MaskProcessor
from src.blending import Blender
from src.ghost_effect import GhostEffectEngine
from src.hand_tracking import HandTracker
from src.hud import HUD
from utils.fps import FPSCounter
from utils.image_utils import save_screenshot, VideoRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Ghost Invisibility System")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera device index (0 for default, 1 or 2 for external/mobile camera)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic camera mode for testing/demo without webcam")
    parser.add_argument("--duration", type=float, default=0, help="Run headless for specified duration in seconds (for CI/automated verification)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=====================================================")
    print("      REAL-TIME GHOST INVISIBILITY SYSTEM            ")
    print("=====================================================")

    # 1. Initialize Camera
    camera = CameraManager(
        camera_index=args.camera,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        force_synthetic=args.synthetic,
    )

    # 2. Initialize CV Pipeline Modules
    segmenter = PersonSegmenter(threshold=config.SEGMENTATION_THRESHOLD)
    bg_manager = BackgroundManager()
    mask_processor = MaskProcessor()
    blender = Blender(default_alpha=config.DEFAULT_ALPHA)
    ghost_engine = GhostEffectEngine(blender=blender)
    hand_tracker = HandTracker(enabled=config.ENABLE_HAND_TRACKING)
    hud = HUD()
    fps_counter = FPSCounter()
    recorder = VideoRecorder()

    window_name = "Ghost Vision System - OpenCV"
    headless = args.duration > 0

    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[WARNING] GUI window display unavailable: {e}. Switching to headless mode.")
            headless = True

    print("\n[CONTROLS GUIDE]")
    print("  'I' or 'i' : Toggle INVISIBLE Mode (Make Human Body 100% Invisible!)")
    print("  'B' or 'b' : Capture Clean Room Background (Step out of view for 1s)")
    print("  'P' or 'p' : Cycle Background Presets (Room / Studio / Cyberpunk)")
    print("  'C' or 'c' : Toggle Active Camouflage (Refract live background)")
    print("  'N' or 'n' : Return to NORMAL camera feed")
    print("  'G' or 'g' : Toggle Spectral GHOST Mode")
    print("  'S' or 's' : Take Timestamped Screenshot")
    print("  'Q' or ESC : Exit Application\n")

    # Start in DEFAULT_MODE (MODE_NORMAL)
    ghost_engine.set_mode(config.MODE_NORMAL)

    start_timestamp = cv2.getTickCount()

    try:
        while True:
            # Check duration exit for automated testing
            if args.duration > 0:
                elapsed = (cv2.getTickCount() - start_timestamp) / cv2.getTickFrequency()
                if elapsed >= args.duration:
                    print(f"[INFO] Target duration ({args.duration}s) reached. Exiting cleanly.")
                    break

            # A. Capture Live Frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # B. Update FPS
            current_fps = fps_counter.update()

            # C. Hand Gesture Tracking
            gesture = hand_tracker.process_frame(frame)

            # D. Pure AI Person Segmentation
            raw_mask = segmenter.segment(frame)

            # E. Mask Refinement & Contour Cleaning
            refined_mask = mask_processor.process(raw_mask)

            # F. Check Human Presence
            human_coverage = np.mean(refined_mask > 0.3)
            is_human_detected = human_coverage > 0.02

            # G. Auto-Capture Clean Room Background when no human is in frame
            if not is_human_detected and not bg_manager.is_preset_active:
                bg_manager.auto_capture_if_clear(frame, refined_mask)

            # H. Retrieve Clean Background Frame
            clean_bg = bg_manager.get_background(
                target_shape=(frame.shape[1], frame.shape[0]),
                live_frame=frame,
                person_mask=refined_mask,
            )

            # I. Render Invisibility / Vision Effect
            rendered_frame = ghost_engine.render(frame, clean_bg, refined_mask)

            # J. HUD Overlay
            output_frame = hud.draw(
                frame=rendered_frame,
                mode=ghost_engine.mode,
                fps=current_fps,
                segmentation_on=segmenter.use_mediapipe or True,
                gesture_status=gesture,
                has_background=bg_manager.has_background(),
                is_human_detected=is_human_detected,
                is_preset=bg_manager.is_preset_active,
                is_recording=recorder.is_recording,
            )

            # Write frame if video recording is active
            recorder.write(output_frame)

            # K. Display Window
            if not headless:
                try:
                    cv2.imshow(window_name, output_frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as e:
                    print(f"[WARNING] cv2.imshow failed: {e}. Disabling GUI display.")
                    headless = True
                    key = 255

                if key != 255:
                    char_key = chr(key).lower() if 0 <= key < 256 else ''

                    if char_key == 'q' or key == 27:
                        break
                    elif char_key == 'i':
                        if ghost_engine.mode == config.MODE_INVISIBLE:
                            ghost_engine.set_mode(config.MODE_NORMAL)
                            print("[INFO] Invisibility deactivated -> Mode: NORMAL")
                        else:
                            ghost_engine.set_mode(config.MODE_INVISIBLE)
                            print("[INFO] INVISIBILITY ACTIVATED!")
                    elif char_key == 'b':
                        bg_manager.capture_background(frame)
                        print("[INFO] Captured background frame manually via 'B' key.")
                    elif char_key == 'p':
                        preset_name = bg_manager.cycle_preset((frame.shape[1], frame.shape[0]))
                        print(f"[INFO] Active preset changed to: {preset_name}")
                    elif char_key == 'c':
                        if ghost_engine.mode == config.MODE_CAMOUFLAGE:
                            ghost_engine.set_mode(config.MODE_NORMAL)
                        else:
                            ghost_engine.set_mode(config.MODE_CAMOUFLAGE)
                            print("[INFO] Active Camouflage Cloak ACTIVATED!")
                    elif char_key == 'g':
                        ghost_engine.set_mode(config.MODE_GHOST)
                    elif char_key == 'n':
                        ghost_engine.set_mode(config.MODE_NORMAL)
                    elif char_key == 's':
                        save_screenshot(output_frame)
                    elif char_key == 't':
                        hand_tracker.enabled = not hand_tracker.enabled
                        print(f"[INFO] Hand gesture control set to: {hand_tracker.enabled}")

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected.")
    finally:
        recorder.stop()
        camera.release()
        if not headless:
            cv2.destroyAllWindows()
        print("[INFO] Application shutdown complete.")


if __name__ == "__main__":
    main()
