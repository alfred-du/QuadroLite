from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class CaptureStage(Stage):
    """Captures frames from the OV5647 via picamera2.

    This stage is the pipeline *source*: ``process()`` ignores its input
    and returns a fresh RGB numpy array from the camera.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("capture", config)
        self._picam2: Any = None

    def setup(self) -> None:
        from libcamera import Transform, controls
        from picamera2 import Picamera2

        width = self.config.get("width", 640)
        height = self.config.get("height", 480)
        fps = self.config.get("fps", 30)
        hflip = self.config.get("hflip", False)
        vflip = self.config.get("vflip", False)
        awb_mode = self.config.get("awb_mode", "auto")

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

        colour_gains = self.config.get("colour_gains")
        if colour_gains is not None:
            cam_controls["ColourGains"] = tuple(colour_gains)
        else:
            cam_controls["AwbEnable"] = True
            cam_controls["AwbMode"] = awb_modes.get(
                awb_mode, controls.AwbModeEnum.Auto,
            )

        self._picam2 = Picamera2()
        cam_config = self._picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls=cam_controls,
            transform=Transform(hflip=hflip, vflip=vflip),
        )
        self._picam2.configure(cam_config)
        self._picam2.start()
        self._log.info(
            "Camera started: %dx%d @ %d fps (hflip=%s, vflip=%s, awb=%s)",
            width, height, fps, hflip, vflip, awb_mode,
        )

    def process(self, _item: Any = None) -> np.ndarray:
        frame = self._picam2.capture_array()
        return frame[:, :, ::-1].copy()

    def cleanup(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._log.info("Camera closed.")
