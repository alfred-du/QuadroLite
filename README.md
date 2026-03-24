# QuadroLite

A memory-efficient computer vision pipeline for the OV5647 camera on a 4GB Raspberry Pi, targeting static hand gesture recognition.

## Hardware

- Raspberry Pi 4 (4GB RAM)
- OV5647 Camera Module (Pi Camera v1) connected via CSI-2

## Setup

**Raspberry Pi OS Bookworm 64-bit** is required.  Python 3.11+ (including 3.13) is supported.

```bash
# Install picamera2 via apt (ensures libcamera compatibility)
sudo apt install -y python3-picamera2 python3-venv

# Create a virtual environment with access to system-level picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

Two ONNX model files (palm detection + hand pose estimation) are downloaded automatically on first run into `models/`.

## Usage

```bash
python -m src.main
```

Override the config file:

```bash
python -m src.main --config config/pipeline.yaml
```

## Architecture

The pipeline runs three threads connected by bounded queues:

1. **Capture** -- reads frames from the OV5647 via picamera2
2. **Inference** -- preprocesses the frame, runs palm detection + hand landmark estimation (ONNX models via OpenCV DNN), classifies the gesture via rule-based heuristics
3. **Dispatch** -- maps recognized gestures to actions (terminal output now, servo control later)

Stale frames are dropped (newest-wins) so inference latency never backs up the camera.

## Configuration

All tunables live in `config/pipeline.yaml`: camera resolution/fps, detection confidence thresholds, gesture-to-action mappings, and output handler selection.

## Memory Budget

| Component | Estimate |
|---|---|
| OS (headless) | ~300 MB |
| Python + libs | ~200 MB |
| Camera buffers | ~4 MB |
| ONNX models (palm + hand) | ~8 MB |
| Inference working memory | ~100 MB |
| **Total** | **~620 MB** |
| **Headroom** | **~3.4 GB** |
