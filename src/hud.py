"""
Heads-up display (HUD) overlay renderer with real room background telemetry and countdown.
"""

import cv2
import numpy as np


class HUD:
    """
    Renders real-time telemetry, mode status, FPS, gesture status,
    countdown timer, and keyboard control legends onto output video frames.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_title = 0.65
        self.font_scale_info = 0.5
        self.font_scale_legend = 0.42

    def draw(
        self,
        frame: np.ndarray,
        mode: str,
        fps: float,
        segmentation_on: bool,
        gesture_status: str,
        has_background: bool,
        is_human_detected: bool,
        is_preset: bool = False,
        countdown_sec: int = 0,
        show_success: bool = False,
        is_recording: bool = False,
    ) -> np.ndarray:
        """
        Draws HUD banner overlay onto frame.
        """
        if frame is None or frame.size == 0:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        # Draw semi-transparent dark banner background at top (height 75px)
        banner_h = 75
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 25), -1)

        # Draw bottom control bar (height 30px)
        cv2.rectangle(overlay, (0, h - 30), (w, h), (15, 15, 25), -1)

        # Alpha blend banner overlay
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        # Top Banner Text Row 1: Title & Main Status
        title_text = "GHOST VISION"
        cv2.putText(output, title_text, (15, 25), self.font, self.font_scale_title, (0, 255, 255), 2, cv2.LINE_AA)

        if mode == "INVISIBLE":
            mode_color = (0, 255, 0)
            mode_text = "Mode: INVISIBLE (100%)"
        elif mode == "CAMOUFLAGE":
            mode_color = (0, 242, 254)
            mode_text = "Mode: CAMOUFLAGE (Refract)"
        elif mode == "GHOST":
            mode_color = (0, 200, 255)
            mode_text = "Mode: GHOST"
        else:
            mode_color = (200, 200, 200)
            mode_text = "Mode: NORMAL"

        cv2.putText(output, mode_text, (160, 25), self.font, self.font_scale_info, mode_color, 2 if mode in ["INVISIBLE", "CAMOUFLAGE"] else 1, cv2.LINE_AA)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(output, fps_text, (380, 25), self.font, self.font_scale_info, (255, 255, 255), 1, cv2.LINE_AA)

        det_color = (0, 255, 0) if is_human_detected else (150, 150, 150)
        det_str = "HUMAN DETECTED" if is_human_detected else "NO HUMAN"
        cv2.putText(output, det_str, (480, 25), self.font, self.font_scale_info, det_color, 2 if is_human_detected else 1, cv2.LINE_AA)

        # Top Banner Text Row 2: Secondary Telemetry & Guidance
        if is_preset:
            bg_str = "PRESET ROOM (Press 'B' for YOUR room)"
            bg_color = (0, 242, 254)
        elif has_background:
            bg_str = "YOUR REAL ROOM (Captured & Ready)"
            bg_color = (0, 255, 0)
        else:
            bg_str = "Press 'B' to capture your empty room"
            bg_color = (0, 165, 255)

        cv2.putText(output, f"BG: {bg_str}", (15, 55), self.font, self.font_scale_info, bg_color, 1, cv2.LINE_AA)

        if mode == "INVISIBLE":
            action_hint = "[I] Active (Press 'I' for Normal)"
            hint_color = (0, 255, 100)
        else:
            action_hint = "[I] Press 'I' to become INVISIBLE"
            hint_color = (0, 200, 255)

        cv2.putText(output, action_hint, (w - 320, 55), self.font, self.font_scale_info, hint_color, 1, cv2.LINE_AA)

        if is_recording:
            cv2.circle(output, (w - 20, 22), 7, (0, 0, 255), -1)
            cv2.putText(output, "REC", (w - 55, 26), self.font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Center Screen Countdown Overlay
        if countdown_sec > 0:
            box_w, box_h = 440, 130
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            # Dark backing box
            sub_overlay = output.copy()
            cv2.rectangle(sub_overlay, (bx, by), (bx + box_w, by + box_h), (10, 10, 20), -1)
            cv2.rectangle(sub_overlay, (bx, by), (bx + box_w, by + box_h), (0, 255, 255), 3)
            cv2.addWeighted(sub_overlay, 0.85, output, 0.15, 0, output)

            cv2.putText(output, "STEP OUT OF CAMERA VIEW!", (bx + 25, by + 45), self.font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            count_str = f"Capturing clean room in: {countdown_sec}s"
            cv2.putText(output, count_str, (bx + 40, by + 95), self.font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        elif show_success:
            box_w, box_h = 460, 100
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            sub_overlay = output.copy()
            cv2.rectangle(sub_overlay, (bx, by), (bx + box_w, by + box_h), (10, 30, 10), -1)
            cv2.rectangle(sub_overlay, (bx, by), (bx + box_w, by + box_h), (0, 255, 0), 3)
            cv2.addWeighted(sub_overlay, 0.85, output, 0.15, 0, output)

            cv2.putText(output, "REAL ROOM CAPTURED!", (bx + 60, by + 40), self.font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(output, "Step back in & press 'I' to be invisible!", (bx + 30, by + 75), self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Bar: Keyboard Shortcuts Legend
        legend = "[I] Invisibility  [B] 3s Room Capture  [P] Preset BG  [C] Camouflage  [N] Normal  [Q] Quit"
        cv2.putText(output, legend, (10, h - 10), self.font, self.font_scale_legend, (220, 220, 220), 1, cv2.LINE_AA)

        return output
