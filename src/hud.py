import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from config import (
    OperationalMode, SegmentationMethod, HUD_TEXT_COLOR,
    HUD_HIGHLIGHT_COLOR, RECORDING_PULSE_SPEED, DEFAULT_SHOW_HUD
)

class HUDRenderer:
    """
    Renders futuristic Sci-Fi Heads-Up Display (HUD) overlay over composite frames.
    """
    def __init__(self, width: int, height: int, show_hud: bool = DEFAULT_SHOW_HUD):
        self.width = width
        self.height = height
        self.show_hud = show_hud

    def toggle(self) -> bool:
        self.show_hud = not self.show_hud
        return self.show_hud

    def render(self, frame: np.ndarray,
               mode: OperationalMode,
               method: SegmentationMethod,
               color_name: str,
               opacity: float,
               fps: float,
               is_recording: bool = False,
               is_synthetic: bool = False,
               gesture_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Renders HUD layer on top of frame if show_hud is True.
        """
        if not self.show_hud:
            return frame

        hud_canvas = frame.copy()
        overlay = frame.copy()

        # 1. Top Header Banner Translucent Overlay
        cv2.rectangle(overlay, (0, 0), (self.width, 60), (10, 15, 25), -1)

        # 2. Operational Mode Badge
        mode_text = f"MODE: {mode.value}"
        cv2.putText(overlay, mode_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. Segmentation Engine / Preset Badge
        if method == SegmentationMethod.HSV_COLOR:
            seg_text = f"ENGINE: HSV CLOAK [{color_name}]"
        else:
            seg_text = f"ENGINE: MEDIAPIPE AI"
        cv2.putText(overlay, seg_text, (320, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HUD_TEXT_COLOR, 2)

        # 4. Performance FPS & Resolution Counter
        fps_text = f"FPS: {fps:.1f} | {self.width}x{self.height}"
        if is_synthetic:
            fps_text += " [SYNTHETIC]"
        cv2.putText(overlay, fps_text, (self.width - 340, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        # 5. Dynamic Recording Indicator (Pulsing Red Dot)
        if is_recording:
            pulse = (int(time.time() / RECORDING_PULSE_SPEED) % 2 == 0)
            dot_color = (0, 0, 255) if pulse else (0, 0, 120)
            cv2.circle(overlay, (self.width - 40, 32), 10, dot_color, -1)
            cv2.putText(overlay, "REC", (self.width - 85, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 6. Bottom Status Bar & Controls Overlay Box
        cv2.rectangle(overlay, (15, self.height - 110), (450, self.height - 15), (10, 15, 25), -1)
        cv2.rectangle(overlay, (15, self.height - 110), (450, self.height - 15), HUD_TEXT_COLOR, 1)

        # Opacity Bar Graphic
        cv2.putText(overlay, f"OPACITY: {int(opacity * 100)}%", (30, self.height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_HIGHLIGHT_COLOR, 2)

        # Draw Opacity Slider Bar
        bar_x, bar_y, bar_w, bar_h = 170, self.height - 92, 250, 14
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * opacity)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), HUD_HIGHLIGHT_COLOR, -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # Keybindings Short Guide
        guide_text = "[M]: Engine | [C]: Mode | [K]: Color | [B]: Capture BG | [S]: Shot | [R]: Rec | [Q]: Quit"
        cv2.putText(overlay, guide_text, (30, self.height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 220), 1)

        # Hand Gesture Feedback Badge
        if gesture_info and gesture_info.get('detected'):
            g_text = f"HAND: {gesture_info['gesture']}"
            cv2.rectangle(overlay, (self.width - 250, self.height - 50), (self.width - 20, self.height - 15), (10, 25, 15), -1)
            cv2.rectangle(overlay, (self.width - 250, self.height - 50), (self.width - 20, self.height - 15), (0, 255, 150), 1)
            cv2.putText(overlay, g_text, (self.width - 235, self.height - 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)

        # Blend semi-transparent HUD overlay (alpha = 0.75)
        cv2.addWeighted(overlay, 0.75, hud_canvas, 0.25, 0, hud_canvas)
        return hud_canvas
