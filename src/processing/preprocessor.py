from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.pipeline.stage import Stage


class PreprocessStage(Stage):
    """Resize, colour-convert, and optionally flip a captured frame.

    Input : RGB ``np.ndarray`` from the camera.
    Output: RGB ``np.ndarray`` ready for inference.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("preprocess", config)
        self._target_w: int = 0
        self._target_h: int = 0

    def setup(self) -> None:
        self._target_w = self.config.get("width", 640)
        self._target_h = self.config.get("height", 480)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None  # type: ignore[return-value]

        h, w = frame.shape[:2]
        if (w, h) != (self._target_w, self._target_h):
            frame = cv2.resize(
                frame,
                (self._target_w, self._target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame
