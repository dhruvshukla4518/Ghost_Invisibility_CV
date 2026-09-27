# Ghost Invisibility CV - AI Vision System

A modular, real-time Computer Vision application written in Python featuring **Invisibility Cloak Effects**, **Spectral Ghost Visuals**, **Dual-Mode Segmentation (HSV & MediaPipe AI)**, **MediaPipe Hand Gesture Control**, **Futuristic Sci-Fi HUD**, **MP4 Video & Screenshot Recording**, and automated tests.

---

## 🌟 Features

- 🧙‍♂️ **Invisibility Cloak Engine**: Seamlessly replaces red/colored cloaks (or AI-segmented human figures) with a temporal median background.
- 👻 **Spectral Ghost Mode**: Renders transparent phantom effects with **motion trails**, **glowing spectral auras**, and **OpenCV colormaps** (Bone, Ocean, Jet, Plasma, Cyberpunk).
- 🤖 **Dual-Engine Segmentation**:
  - **HSV Color Segmentation**: Thresholds colors (Red, Green, Blue, Cyan, Magenta, Yellow) with dual-range boundary handling for red hues.
  - **MediaPipe AI Person Segmentation**: Neural-network selfie segmenter for cloaking without requiring physical colored cloth.
- 🖐️ **Hand Gesture Controls**:
  - **Pinch (Thumb + Index)**: Dynamically adjusts ghost opacity / transparency in real time.
  - **Open Palm**: Switches mode to Invisibility Cloak.
  - **Fist**: Triggers background frame recapture.
  - **Peace Sign (V)**: Cycles spectral colormaps.
- 🖥️ **Sci-Fi Heads-Up Display (HUD)**: Interactive neon UI (hidden by default for a clean video feed; toggle with `h` or launch with `--hud`).
- 📸 **Recording & Screenshots**: Capture timestamped PNG screenshots (`outputs/screenshots/`) or record MP4 video streams (`outputs/videos/`).
- 🎮 **Synthetic Camera Generator**: Built-in animated fallback camera to test and evaluate the entire vision pipeline without requiring a physical webcam!

---

## 📁 Directory Structure

```
ghost_invisibility_cv/
│
├── main.py                     # Main application entry point & event loop
├── config.py                   # Centralized configuration, HSV presets & keybindings
├── requirements.txt            # Dependencies list
├── README.md                   # Project documentation
│
├── src/
│   ├── __init__.py
│   ├── camera.py               # OpenCV VideoCapture wrapper & Synthetic Camera fallback
│   ├── background.py           # Multi-frame temporal median background manager
│   ├── segmentation.py         # Dual HSV Color & MediaPipe AI Segmentation
│   ├── mask_processing.py      # Morphological cleanup, contour filtering & edge softening
│   ├── ghost_effect.py         # Motion trails, glowing spectral aura & colormap engine
│   ├── blending.py             # Multi-layer smooth alpha compositor
│   ├── hand_tracking.py        # MediaPipe hand tracking & gesture recognition
│   └── hud.py                  # Sci-Fi HUD overlay renderer
│
├── utils/
│   ├── __init__.py
│   ├── fps.py                  # Smoothed FPS counter
│   └── image_utils.py          # Screenshot saver & VideoRecorder helper
│
├── assets/
│   ├── backgrounds/            # Custom static background images
│   └── screenshots/            # Asset samples
│
├── outputs/
│   ├── screenshots/            # Saved PNG screenshots
│   └── videos/                 # Recorded MP4 output videos
│
└── tests/
    ├── test_segmentation.py    # HSV & AI segmentation unit tests
    ├── test_mask.py            # Morphological mask processing unit tests
    └── test_blending.py        # Image composition unit tests
```

---

## ⚡ Quick Start

### 1. Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

Launch with standard live webcam:

```bash
python main.py
```

Launch in **Synthetic Test Mode** (No webcam required):

```bash
python main.py --synthetic
```

Launch with specific initial mode or cloak color:

```bash
python main.py --mode GHOST --color BLUE
```

Disable hand tracking:

```bash
python main.py --no-hand-tracking
```

Use custom static background image:

```bash
python main.py --background assets/backgrounds/my_room.jpg
```

Launch with Sci-Fi HUD enabled on startup:

```bash
python main.py --hud
```

---

## ⌨️ Controls & Shortcuts

| Key / Gesture | Action |
|---|---|
| **`b` / Fist Gesture** | Capture / Recapture Background Frame |
| **`c` / Open Palm** | Cycle Operational Mode (`CLOAK` ➔ `GHOST` ➔ `SWAP` ➔ `NORMAL`) |
| **`k`** | Cycle HSV Cloak Color (`RED` ➔ `GREEN` ➔ `BLUE` ➔ `CYAN` ➔ `MAGENTA` ➔ `YELLOW`) |
| **`g` / Peace Sign** | Cycle Spectral Ghost Colormaps (`BONE` ➔ `OCEAN` ➔ `JET` ➔ `PLASMA` ➔ `CYBERPUNK`) |
| **`h`** | Toggle Sci-Fi HUD Overlay (Default: OFF) |
| **`s`** | Take Timestamped PNG Screenshot |
| **`r`** | Start / Stop MP4 Video Recording |
| **`+` / `-` / Pinch** | Adjust Transparency / Opacity Level |
| **`q` / ESC** | Exit Application |

---

## 🧪 Running Unit Tests

Run automated tests via `pytest`:

```bash
pytest tests/
```

---

## 📜 License

MIT License. Developed with OpenCV & MediaPipe.
