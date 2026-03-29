"""Camera capture backends.

Supports two backends selected via ``camera.backend`` in the pipeline config:

* **picamera2** -- Raspberry Pi CSI camera via libcamera (requires picamera2).
* **opencv**    -- Any V4L2 / USB / platform camera via ``cv2.VideoCapture``.
* **auto**      -- Try picamera2 first; fall back to opencv.
"""
from __future__ import annotations

import abc
import logging
from typing import Any

import cv2 as cv
import numpy as np

from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Backend interface
# ------------------------------------------------------------------

class _CaptureBackend(abc.ABC):
    @abc.abstractmethod
    def open(self, config: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def read(self) -> np.ndarray: ...

    @abc.abstractmethod
    def close(self) -> None: ...


# ------------------------------------------------------------------
# picamera2 backend (Raspberry Pi)
# ------------------------------------------------------------------

class _PicameraBackend(_CaptureBackend):
    def __init__(self) -> None:
        self._cam: Any = None

    def open(self, config: dict[str, Any]) -> None:
        from libcamera import Transform, controls
        from picamera2 import Picamera2

        width = config.get("width", 640)
        height = config.get("height", 480)
        fps = config.get("fps", 30)
        hflip = config.get("hflip", False)
        vflip = config.get("vflip", False)
        awb_mode = config.get("awb_mode", "auto")

        awb_modes = {
            "auto": controls.AwbModeEnum.Auto,
            "incandescent": controls.AwbModeEnum.Incandescent,
            "tungsten": controls.AwbModeEnum.Tungsten,
            "fluorescent": controls.AwbModeEnum.Fluorescent,
            "indoor": controls.AwbModeEnum.Indoor,
            "daylight": controls.AwbModeEnum.Daylight,
            "cloudy": controls.AwbModeEnum.Cloudy,
        }

        cam_controls: dict = {"FrameRate": fps}

        colour_gains = config.get("colour_gains")
        if colour_gains is not None:
            cam_controls["ColourGains"] = tuple(colour_gains)
        else:
            cam_controls["AwbEnable"] = True
            cam_controls["AwbMode"] = awb_modes.get(
                awb_mode, controls.AwbModeEnum.Auto,
            )

        self._cam = Picamera2()
        cam_config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=cam_controls,
            transform=Transform(hflip=hflip, vflip=vflip),
        )
        self._cam.configure(cam_config)
        self._cam.start()
        logger.info(
            "picamera2 started: %dx%d @ %d fps (hflip=%s, vflip=%s, awb=%s)",
            width, height, fps, hflip, vflip, awb_mode,
        )

    def read(self) -> np.ndarray:
        frame = self._cam.capture_array()
        return frame[:, :, ::-1].copy()

    def close(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()


# ------------------------------------------------------------------
# OpenCV backend (Ubuntu / generic V4L2 / USB)
# ------------------------------------------------------------------

class _OpenCVBackend(_CaptureBackend):
    def __init__(self) -> None:
        self._cap: cv.VideoCapture | None = None
        self._flip_code: int | None = None

    def open(self, config: dict[str, Any]) -> None:
        device = config.get("device", 0)
        width = config.get("width", 640)
        height = config.get("height", 480)
        fps = config.get("fps", 30)

        self._cap = cv.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV VideoCapture failed to open device {device!r}. "
                "Check that /dev/video* exists and is accessible."
            )

        self._cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv.CAP_PROP_FPS, fps)

        actual_w = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv.CAP_PROP_FPS)

        hflip = config.get("hflip", False)
        vflip = config.get("vflip", False)
        if hflip and vflip:
            self._flip_code = -1
        elif hflip:
            self._flip_code = 1
        elif vflip:
            self._flip_code = 0

        logger.info(
            "OpenCV camera started: device=%s %dx%d @ %.0f fps "
            "(requested %dx%d @ %d)",
            device, actual_w, actual_h, actual_fps, width, height, fps,
        )

    def read(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("OpenCV VideoCapture.read() returned empty frame")
        if self._flip_code is not None:
            frame = cv.flip(frame, self._flip_code)
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ------------------------------------------------------------------
# Backend factory
# ------------------------------------------------------------------

def _create_backend(name: str) -> _CaptureBackend:
    if name == "picamera2":
        return _PicameraBackend()
    if name == "opencv":
        return _OpenCVBackend()
    if name == "auto":
        try:
            import picamera2  # noqa: F401
            logger.info("Auto-detected picamera2 backend")
            return _PicameraBackend()
        except ImportError:
            logger.info("picamera2 not available; falling back to OpenCV backend")
            return _OpenCVBackend()
    raise ValueError(f"Unknown camera backend: {name!r}")


# ------------------------------------------------------------------
# Pipeline stage
# ------------------------------------------------------------------

class CaptureStage(Stage):
    """Captures frames from a camera.

    Backend is selected via ``camera.backend`` in config:
    ``auto`` (default) | ``picamera2`` | ``opencv``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("capture", config)
        backend_name = self.config.get("backend", "auto")
        self._backend = _create_backend(backend_name)

    def setup(self) -> None:
        self._backend.open(self.config)

    def process(self, _item: Any = None) -> np.ndarray:
        return self._backend.read()

    def cleanup(self) -> None:
        self._backend.close()
        self._log.info("Camera closed.")
