# Real-Time Ghost Invisibility System 👻

A real-time computer vision application built with **Python**, **OpenCV**, and **MediaPipe** that creates interactive ghost and invisibility visual effects on camera streams.

---

## 🌟 Features

- **Real-Time Webcam Processing**: Continuous high-FPS frame processing with automatic camera detection and synthetic frame fallback.
- **Background Capture & Storage**: Capture and persist clean background frames on demand.
- **AI Person Segmentation**: Powered by MediaPipe image segmentation with an adaptive OpenCV background subtraction fallback.
- **Advanced Mask Refinement**: Morphological noise reduction (opening, closing, erosion, dilation) and Gaussian edge softening.
- **Multi-Mode Ghost Engine**:
  - `GHOST`: Ethereal semi-transparent blending with soft glow.
  - `INVISIBLE`: Complete background replacement within the person region.
  - `NEON_GHOST`: Chromatic cyan/magenta edge aura effect.
  - `GLITCH`: Digital artifact displacement effect.
  - `NORMAL`: Unfiltered live camera stream.
- **Hand Gesture Control**: Optional MediaPipe hand landmark gesture control (Open Palm, Fist, Victory, Thumb Up, Thumb Down).
- **Interactive Telemetry HUD**: Live FPS, current mode, segmentation state, gesture recognition status, and keyboard legends.
- **Outputs & Recording**: Save high-res screenshots and record video output directly.

---

## 📁 Project Structure

```
ghost_invisibility_cv/
├── main.py                     # Entry point & application orchestrator
├── config.py                   # Centralized configuration parameters
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── src/                        # Modular CV components
│   ├── __init__.py
│   ├── camera.py               # Webcam & synthetic camera manager
│   ├── segmentation.py         # MediaPipe / OpenCV person segmenter
│   ├── background.py           # Background frame capture & storage
│   ├── mask_processing.py      # Morphological mask smoothing
│   ├── blending.py             # Alpha blending engine
│   ├── hand_tracking.py        # Gesture tracking & classifier
│   ├── ghost_effect.py         # Multi-effect rendering engine
│   └── hud.py                  # Telemetry heads-up display overlay
├── utils/                      # Helper utilities
│   ├── __init__.py
│   ├── fps.py                  # Smooth FPS tracker
│   └── image_utils.py          # Screenshot & video saving helpers
├── assets/                     # Default backgrounds & assets
│   ├── backgrounds/
│   └── screenshots/
├── outputs/                    # Exported outputs
│   ├── screenshots/
│   └── videos/
└── tests/                      # PyTest automated unit tests
    ├── test_segmentation.py
    ├── test_mask.py
    └── test_blending.py
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
python main.py
```

To run in synthetic demo mode (without webcam hardware attached):
```bash
python main.py --synthetic
```

---

## 🎮 Controls

### Keyboard Shortcuts
| Key | Action |
| --- | --- |
| `B` | Capture clean background frame |
| `G` | Switch to **Ghost Mode** |
| `I` | Switch to **Invisible Mode** |
| `N` | Switch to **Normal Mode** |
| `+` / `=` | Increase Ghost Effect Intensity ($\alpha$) |
| `-` | Decrease Ghost Effect Intensity ($\alpha$) |
| `S` | Save timestamped screenshot |
| `R` | Reset background & effect parameters |
| `T` | Toggle Hand Gesture Control ON / OFF |
| `Q` | Quit application |

### Hand Gestures (when Hand Tracking is enabled)
| Gesture | Action |
| --- | --- |
| 🖐️ **Open Palm** | Activate Ghost Mode |
| ✊ **Fist** | Switch to Normal Mode |
| ✌️ **Two Fingers (Victory)** | Toggle Invisible Mode |
| 👍 **Thumb Up** | Increase effect intensity |
| 👎 **Thumb Down** | Decrease effect intensity |

---

## 🧪 Testing

Run unit tests with pytest:
```bash
pytest tests/ -v
```
