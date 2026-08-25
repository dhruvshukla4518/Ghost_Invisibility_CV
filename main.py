"""
Real-Time Ghost Invisibility System - Main Application Entry Point.
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
    print("  'B' or 'b' : Capture/Recapture Clean Background (Step out of view first!)")
    print("  'N' or 'n' : Return to NORMAL camera feed")
    print("  'G' or 'g' : Toggle Spectral GHOST Mode")
    print("  'S' or 's' : Take Timestamped Screenshot")
    print("  'Q' or ESC : Exit Application\n")

    # Start in DEFAULT_MODE (MODE_NORMAL)
    ghost_engine.set_mode(config.MODE_NORMAL)

    # Read initial frame
    ret, initial_frame = camera.read()
    if ret and not bg_manager.has_background():
        print("[NOTICE] Capturing initial background reference frame...")
        bg_manager.capture_background(initial_frame)

    start_timestamp = cv2.getTickCount()

    try:
        while True:
            # Check duration exit for automated testing
            if args.duration > 0:
                elapsed = (cv2.getTickCount() - start_timestamp) / cv2.getTickFrequency()
                if elapsed >= args.duration:
                    print(f"[INFO] Target duration ({args.duration}s) reached. Exiting cleanly.")
                    break

            # A. Capture Frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # B. Update FPS
            current_fps = fps_counter.update()

            # C. Hand Gesture Tracking & Parameter Control
            gesture = hand_tracker.process_frame(frame)

            # D. Person Segmentation (Advanced MediaPipe AI Detection)
            clean_bg = bg_manager.get_background((frame.shape[1], frame.shape[0]))
            raw_mask = segmenter.segment(frame, clean_bg)

            # E. Mask Refinement & Contour Cleaning
            refined_mask = mask_processor.process(raw_mask)

            # F. Render Ghost/Invisibility Effect
            rendered_frame = ghost_engine.render(frame, clean_bg, refined_mask)

            # G. HUD Overlay
            output_frame = hud.draw(
                frame=rendered_frame,
                mode=ghost_engine.mode,
                fps=current_fps,
                segmentation_on=segmenter.use_mediapipe or True,
                gesture_status=gesture,
                has_background=bg_manager.has_background(),
                alpha=ghost_engine.intensity,
                is_recording=recorder.is_recording,
            )

            # Write frame if video recording is active
            recorder.write(output_frame)

            # H. Display Window
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
                    elif char_key == 'b':
                        bg_manager.capture_background(frame)
                        print("[INFO] Captured new background frame! Ensure you were out of camera view.")
                    elif char_key == 'i':
                        if ghost_engine.mode == config.MODE_INVISIBLE:
                            ghost_engine.set_mode(config.MODE_NORMAL)
                            print("[INFO] Invisibility deactivated -> Mode: NORMAL")
                        else:
                            ghost_engine.set_mode(config.MODE_INVISIBLE)
                            print("[INFO] INVISIBILITY ACTIVATED! -> Mode: INVISIBLE")
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
