# QuadroLite

A memory-efficient computer vision pipeline for hand gesture recognition on Raspberry Pi and Ubuntu 24.04 (with optional ROS 2 integration).

## Hardware

- Raspberry Pi 4 (4 GB RAM) **or** any Ubuntu 24.04 aarch64/x86_64 machine
- Camera: OV5647 CSI module (Pi), or any USB / V4L2 camera (Ubuntu)

## Setup

### Option A — Raspberry Pi OS Bookworm 64-bit

Python 3.11+ (including 3.13) is supported.

```bash
sudo apt install -y python3-picamera2 python3-venv

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B — Ubuntu 24.04 (standalone, no ROS)

Uses the OpenCV camera backend automatically (picamera2 is not available).

```bash
sudo apt install -y python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option C — Ubuntu 24.04 + ROS 2 Jazzy

```bash
# 1. Install ROS 2 Jazzy (follow https://docs.ros.org/en/jazzy/Installation.html)
source /opt/ros/jazzy/setup.bash

# 2. Install pipeline dependencies into the ROS Python environment
pip install -r requirements.txt

# 3. Install cv_bridge (if not already present)
sudo apt install -y ros-jazzy-cv-bridge
```

Two ONNX model files (palm detection + hand pose estimation) are downloaded automatically on first run into `models/`.

## Usage

### Standalone pipeline

```bash
python -m src.main
python -m src.main --config config/pipeline.yaml
```

The camera backend is selected via `camera.backend` in the config:

| Value | Behaviour |
|---|---|
| `auto` (default) | Try picamera2, fall back to OpenCV |
| `picamera2` | Raspberry Pi CSI camera via libcamera |
| `opencv` | Any V4L2 / USB camera via `cv2.VideoCapture` |

### ROS 2 node

Subscribe to an existing camera topic:

```bash
# Start a camera publisher (e.g. v4l2_camera)
ros2 run v4l2_camera v4l2_camera_node

# In another terminal — start the gesture node
python3 -m src.ros2_node --ros-args -p camera_topic:=/image_raw
```

Or use the built-in OpenCV camera (no external camera node):

```bash
python3 -m src.ros2_node --ros-args -p use_internal_camera:=true
```

Gesture results are published as JSON on `~/gesture`:

```bash
ros2 topic echo /quadrolite_gesture/gesture
```

## Architecture

The pipeline runs three threads connected by bounded queues:

1. **Capture** — reads frames via picamera2 (Pi) or OpenCV VideoCapture (Ubuntu)
2. **Inference** — preprocesses the frame, runs palm detection + hand landmark estimation (ONNX models via OpenCV DNN), classifies the gesture via rule-based heuristics
3. **Dispatch** — maps recognized gestures to actions (terminal log, MJPEG preview, or servo control)

Stale frames are dropped (newest-wins) so inference latency never backs up the camera.

The ROS 2 node (`src/ros2_node.py`) runs the same inference stages but replaces the threaded orchestrator with ROS callbacks and publishes results to a topic.

## Configuration

All tunables live in `config/pipeline.yaml`: camera backend/device, resolution/fps, detection confidence thresholds, gesture-to-action mappings, and output handler selection.

## Memory Budget

### Raspberry Pi OS (headless)

| Component | Estimate |
|---|---|
| OS (headless) | ~300 MB |
| Python + libs | ~200 MB |
| Camera buffers | ~4 MB |
| ONNX models (palm + hand) | ~8 MB |
| Inference working memory | ~100 MB |
| **Total** | **~620 MB** |
| **Headroom (4 GB Pi)** | **~3.4 GB** |

### Ubuntu 24.04 + ROS 2 Jazzy

| Component | Estimate |
|---|---|
| OS (headless) | ~350 MB |
| ROS 2 daemon + core nodes | ~150 MB |
| Python + libs + cv_bridge | ~250 MB |
| Camera buffers | ~4 MB |
| ONNX models (palm + hand) | ~8 MB |
| Inference working memory | ~100 MB |
| **Total** | **~860 MB** |
| **Headroom (4 GB Pi)** | **~3.1 GB** |
