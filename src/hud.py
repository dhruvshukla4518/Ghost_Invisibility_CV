"""
Heads-up display (HUD) overlay renderer.
"""

import cv2
import numpy as np


class HUD:
    """
    Renders real-time telemetry, mode status, FPS, gesture status,
    and keyboard control legends onto output video frames.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_title = 0.65
        self.font_scale_info = 0.5
        self.font_scale_legend = 0.45

    def draw(
        self,
        frame: np.ndarray,
        mode: str,
        fps: float,
        segmentation_on: bool,
        gesture_status: str,
        has_background: bool,
        alpha: float,
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
        elif mode == "GHOST":
            mode_color = (0, 200, 255)
            mode_text = "Mode: GHOST"
        else:
            mode_color = (200, 200, 200)
            mode_text = "Mode: NORMAL"

        cv2.putText(output, mode_text, (160, 25), self.font, self.font_scale_info, mode_color, 2 if mode == "INVISIBLE" else 1, cv2.LINE_AA)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(output, fps_text, (370, 25), self.font, self.font_scale_info, (255, 255, 255), 1, cv2.LINE_AA)

        seg_str = "ON (AI)" if segmentation_on else "OFF"
        cv2.putText(output, f"Person Detection: {seg_str}", (470, 25), self.font, self.font_scale_info, (255, 200, 0), 1, cv2.LINE_AA)

        # Top Banner Text Row 2: Secondary Telemetry & Guidance
        if not has_background:
            bg_str = "NOT SET (Step out of view & Press B)"
            bg_color = (0, 0, 255)
        else:
            bg_str = "READY (Press B to recaptured)"
            bg_color = (0, 255, 0)

        cv2.putText(output, f"Background: {bg_str}", (15, 55), self.font, self.font_scale_info, bg_color, 1, cv2.LINE_AA)

        if mode == "INVISIBLE":
            action_hint = "[I] Active - Press 'I' for Normal"
            hint_color = (0, 255, 100)
        else:
            action_hint = "Press 'I' to become INVISIBLE"
            hint_color = (0, 200, 255)

        cv2.putText(output, action_hint, (w - 320, 55), self.font, self.font_scale_info, hint_color, 1, cv2.LINE_AA)

        if is_recording:
            cv2.circle(output, (w - 20, 22), 7, (0, 0, 255), -1)
            cv2.putText(output, "REC", (w - 55, 26), self.font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Bottom Bar: Keyboard Shortcuts Legend
        legend = "[I] Toggle Invisibility  [B] Capture Background  [N] Normal  [G] Ghost  [Q] Quit"
        cv2.putText(output, legend, (10, h - 10), self.font, self.font_scale_legend, (220, 220, 220), 1, cv2.LINE_AA)

        return output
