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
        from picamera2 import Picamera2

        width = self.config.get("width", 640)
        height = self.config.get("height", 480)
        fps = self.config.get("fps", 30)

        self._picam2 = Picamera2()
        cam_config = self._picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._picam2.configure(cam_config)

        hflip = self.config.get("hflip", False)
        vflip = self.config.get("vflip", False)
        self._picam2.set_controls(
            {"HorizontalFlip": int(hflip), "VerticalFlip": int(vflip)}
        )

        self._picam2.start()
        self._log.info(
            "Camera started: %dx%d @ %d fps (hflip=%s, vflip=%s)",
            width, height, fps, hflip, vflip,
        )

    def process(self, _item: Any = None) -> np.ndarray:
        return self._picam2.capture_array()

    def cleanup(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._log.info("Camera closed.")
