from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.pipeline.stage import Stage


class PreprocessStage(Stage):
    """Resize, undistort, and prepare a captured frame for inference.

    Input : RGB ``np.ndarray`` from the camera.
    Output: RGB ``np.ndarray`` ready for inference.
    """

    # OV5647 approximate intrinsics at 640x480 and radial distortion
    # coefficients.  Derived from the stock Pi Camera v1 lens.  For
    # precision work, run a full checkerboard calibration.
    _DEFAULT_CAMERA_MATRIX = np.array([
        [340.0,   0.0, 320.0],
        [  0.0, 340.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)
    _DEFAULT_DIST_COEFFS = np.array(
        [-0.28, 0.07, 0.0, 0.0, 0.0], dtype=np.float64,
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("preprocess", config)
        self._target_w: int = 0
        self._target_h: int = 0
        self._undistort: bool = True
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    def setup(self) -> None:
        self._target_w = self.config.get("width", 640)
        self._target_h = self.config.get("height", 480)
        self._undistort = self.config.get("undistort", True)

        if self._undistort:
            cam_mtx = self._DEFAULT_CAMERA_MATRIX.copy()
            cam_mtx[0, 0] = cam_mtx[0, 0] * self._target_w / 640.0
            cam_mtx[1, 1] = cam_mtx[1, 1] * self._target_h / 480.0
            cam_mtx[0, 2] = self._target_w / 2.0
            cam_mtx[1, 2] = self._target_h / 2.0

            new_mtx, _roi = cv2.getOptimalNewCameraMatrix(
                cam_mtx, self._DEFAULT_DIST_COEFFS,
                (self._target_w, self._target_h), 1,
                (self._target_w, self._target_h),
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                cam_mtx, self._DEFAULT_DIST_COEFFS, None, new_mtx,
                (self._target_w, self._target_h), cv2.CV_16SC2,
            )

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

        if self._undistort and self._map1 is not None:
            frame = cv2.remap(
                frame, self._map1, self._map2, cv2.INTER_LINEAR,
            )

        return frame
