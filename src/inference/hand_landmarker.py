"""Hand landmark detection using OpenCV DNN with MediaPipe ONNX models.

Replaces the mediapipe Tasks API with cv2.dnn for Python 3.13 compatibility.
Uses the same underlying MediaPipe hand models converted to ONNX by OpenCV Zoo.
"""
from __future__ import annotations

import logging
import pathlib
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cv2 as cv
import numpy as np

from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

_PALM_DET_URL = (
    "https://huggingface.co/opencv/palm_detection_mediapipe"
    "/resolve/main/palm_detection_mediapipe_2023feb.onnx"
)
_HAND_POSE_URL = (
    "https://huggingface.co/opencv/handpose_estimation_mediapipe"
    "/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
)


@dataclass
class HandResult:
    """21 landmarks (x, y, z each normalised 0-1) plus handedness."""

    landmarks: list[tuple[float, float, float]]
    handedness: str = "Left"
    score: float = 0.0
    frame: np.ndarray | None = None


@dataclass
class FrameResult:
    """All hands detected in a single frame."""

    hands: list[HandResult] = field(default_factory=list)
    frame: np.ndarray | None = None


# ------------------------------------------------------------------
# SSD anchor generation for the 192x192 palm detection model
# ------------------------------------------------------------------

def _generate_palm_det_anchors() -> np.ndarray:
    anchors: list[list[float]] = []
    for grid_size, n_anchors in [(24, 2), (12, 6)]:
        for y in range(grid_size):
            for x in range(grid_size):
                cx = (x + 0.5) / grid_size
                cy = (y + 0.5) / grid_size
                for _ in range(n_anchors):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


# ------------------------------------------------------------------
# Palm detector
# ------------------------------------------------------------------

class _PalmDetector:
    INPUT_SIZE = np.array([192, 192])

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> None:
        self._model = cv.dnn.readNet(model_path)
        self._score_thr = score_threshold
        self._nms_thr = nms_threshold
        self._anchors = _generate_palm_det_anchors()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return Nx19 array: [x1,y1,x2,y2, 7*(lx,ly), score] in pixels."""
        h, w = bgr.shape[:2]
        blob, pad_bias = self._preprocess(bgr)
        self._model.setInput(blob)
        out = self._model.forward(self._model.getUnconnectedOutLayersNames())
        return self._postprocess(out, np.array([w, h]), pad_bias)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pad_bias = np.array([0.0, 0.0])
        ratio = float(min(self.INPUT_SIZE / img.shape[:2]))

        if img.shape[0] != self.INPUT_SIZE[0] or img.shape[1] != self.INPUT_SIZE[1]:
            new_hw = (np.array(img.shape[:2]) * ratio).astype(np.int32)
            img = cv.resize(img, (int(new_hw[1]), int(new_hw[0])))
            pad_h = int(self.INPUT_SIZE[0] - new_hw[0])
            pad_w = int(self.INPUT_SIZE[1] - new_hw[1])
            left = pad_w // 2
            top = pad_h // 2
            pad_bias[:] = [left, top]
            img = cv.copyMakeBorder(
                img, top, pad_h - top, left, pad_w - left,
                cv.BORDER_CONSTANT, value=(0, 0, 0),
            )

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pad_bias = (pad_bias / ratio).astype(np.int32)
        return img[np.newaxis], pad_bias

    def _postprocess(
        self, out: list[np.ndarray], orig_wh: np.ndarray, pad_bias: np.ndarray,
    ) -> np.ndarray:
        score_raw = out[1][0, :, 0].astype(np.float64)
        score = 1.0 / (1.0 + np.exp(-score_raw))

        box_delta = out[0][0, :, 0:4]
        lm_delta = out[0][0, :, 4:]
        scale = float(max(orig_wh))

        cxy = box_delta[:, :2] / self.INPUT_SIZE
        wh = box_delta[:, 2:] / self.INPUT_SIZE
        xy1 = (cxy - wh / 2 + self._anchors) * scale
        xy2 = (cxy + wh / 2 + self._anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]

        keep = cv.dnn.NMSBoxes(
            boxes, score, self._score_thr, self._nms_thr, top_k=5000,
        )
        keep = np.asarray(keep).flatten()
        if keep.size == 0:
            return np.empty(shape=(0, 19))

        sel_boxes = boxes[keep]
        sel_score = score[keep]

        sel_lm = lm_delta[keep].reshape(-1, 7, 2) / self.INPUT_SIZE
        sel_anchors = self._anchors[keep]
        for i in range(len(sel_lm)):
            sel_lm[i] += sel_anchors[i]
        sel_lm *= scale
        sel_lm -= pad_bias

        return np.c_[
            sel_boxes.reshape(-1, 4),
            sel_lm.reshape(-1, 14),
            sel_score.reshape(-1, 1),
        ]


# ------------------------------------------------------------------
# Hand pose estimator
# ------------------------------------------------------------------

class _HandPoseEstimator:
    INPUT_SIZE = np.array([224, 224])

    _PALM_BASE_IDX = 0
    _MIDDLE_BASE_IDX = 2
    _PRE_SHIFT = np.array([0.0, 0.0])
    _PRE_ENLARGE = 4.0
    _SHIFT = np.array([0.0, -0.4])
    _ENLARGE = 3.0
    _HAND_SHIFT = np.array([0.0, -0.1])
    _HAND_ENLARGE = 1.65

    def __init__(self, model_path: str, conf_threshold: float = 0.8) -> None:
        self._model = cv.dnn.readNet(model_path)
        self._conf_thr = conf_threshold

    def estimate(self, bgr: np.ndarray, palm: np.ndarray) -> np.ndarray | None:
        """Return 132-element vector or None if confidence is too low.

        Layout: bbox(4), screen_lm(63), world_lm(63), handedness(1), conf(1)
        """
        blob, rot_bbox, angle, rot_mat, pad_bias = self._preprocess(bgr, palm)
        self._model.setInput(blob)
        out = self._model.forward(self._model.getUnconnectedOutLayersNames())
        return self._postprocess(out, rot_bbox, angle, rot_mat, pad_bias)

    # --- crop helpers ------------------------------------------------

    @staticmethod
    def _pad_to_square(img: np.ndarray, use_diag: bool) -> tuple[np.ndarray, int, int]:
        side = int(np.linalg.norm(img.shape[:2]) if use_diag else max(img.shape[:2]))
        pad_h = side - img.shape[0]
        pad_w = side - img.shape[1]
        left = pad_w // 2
        top = pad_h // 2
        out = cv.copyMakeBorder(
            img, top, pad_h - top, left, pad_w - left,
            cv.BORDER_CONSTANT, value=(0, 0, 0),
        )
        return out, left, top

    def _crop_palm(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        for_rotation: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wh = bbox[1] - bbox[0]
        shift = (self._PRE_SHIFT if for_rotation else self._SHIFT) * wh
        bbox = bbox + shift

        center = np.sum(bbox, axis=0) / 2
        wh = bbox[1] - bbox[0]
        enlarge = self._PRE_ENLARGE if for_rotation else self._ENLARGE
        half = wh * enlarge / 2
        bbox = np.array([center - half, center + half]).astype(np.int32)
        bbox[:, 0] = np.clip(bbox[:, 0], 0, img.shape[1])
        bbox[:, 1] = np.clip(bbox[:, 1], 0, img.shape[0])

        crop = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        if crop.size == 0:
            crop = np.zeros((1, 1, 3), dtype=img.dtype)

        crop, left, top = self._pad_to_square(crop, for_rotation)
        bias = bbox[0] - np.array([left, top])
        return crop, bbox, bias

    # --- preprocess --------------------------------------------------

    def _preprocess(
        self, img: np.ndarray, palm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        pad_bias = np.array([0, 0], dtype=np.int32)
        bbox = palm[0:4].reshape(2, 2)
        img, bbox, bias = self._crop_palm(img, bbox, True)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pad_bias += bias

        bbox_local = bbox - pad_bias
        lm = palm[4:18].reshape(7, 2) - pad_bias
        p1, p2 = lm[self._PALM_BASE_IDX], lm[self._MIDDLE_BASE_IDX]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))

        center = tuple((np.sum(bbox_local, axis=0) / 2).tolist())
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

        hom = np.c_[lm, np.ones(lm.shape[0])]
        rot_lm = np.array([hom @ rot_mat[0], hom @ rot_mat[1]])
        rot_bbox = np.array([np.amin(rot_lm, axis=1), np.amax(rot_lm, axis=1)])

        crop, rot_bbox, _ = self._crop_palm(rotated, rot_bbox, False)
        blob = cv.resize(crop, tuple(self.INPUT_SIZE), interpolation=cv.INTER_AREA)
        blob = blob.astype(np.float32) / 255.0
        return blob[np.newaxis], rot_bbox, angle, rot_mat, pad_bias

    # --- postprocess -------------------------------------------------

    def _postprocess(
        self,
        out: list[np.ndarray],
        rot_bbox: np.ndarray,
        angle: float,
        rot_mat: np.ndarray,
        pad_bias: np.ndarray,
    ) -> np.ndarray | None:
        landmarks, conf, handedness, landmarks_world = out

        conf_val = float(conf[0][0])
        if conf_val < self._conf_thr:
            return None

        lm = landmarks[0].reshape(-1, 3)
        lm_w = landmarks_world[0].reshape(-1, 3)

        wh = rot_bbox[1] - rot_bbox[0]
        sf = float(max(wh / self.INPUT_SIZE))
        lm[:, :2] = (lm[:, :2] - self.INPUT_SIZE / 2) * sf
        lm[:, 2] *= sf

        c_rot = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rot_lm = np.c_[lm[:, :2] @ c_rot[:, :2], lm[:, 2]]
        rot_lm_w = np.c_[lm_w[:, :2] @ c_rot[:, :2], lm_w[:, 2]]

        R = np.array([[rot_mat[0][0], rot_mat[1][0]],
                       [rot_mat[0][1], rot_mat[1][1]]])
        t = np.array([rot_mat[0][2], rot_mat[1][2]])
        inv_t = -R @ t
        inv_rot = np.c_[R, inv_t]

        center_h = np.append(np.sum(rot_bbox, axis=0) / 2, 1.0)
        orig_center = np.array([center_h @ inv_rot[0], center_h @ inv_rot[1]])
        lm[:, :2] = rot_lm[:, :2] + orig_center + pad_bias

        bbox_lm = np.array([np.amin(lm[:, :2], axis=0), np.amax(lm[:, :2], axis=0)])
        wh_b = bbox_lm[1] - bbox_lm[0]
        bbox_lm += self._HAND_SHIFT * wh_b
        c_b = np.sum(bbox_lm, axis=0) / 2
        wh_b = bbox_lm[1] - bbox_lm[0]
        half_b = wh_b * self._HAND_ENLARGE / 2
        bbox_lm = np.array([c_b - half_b, c_b + half_b])

        return np.r_[
            bbox_lm.reshape(-1), lm.reshape(-1), rot_lm_w.reshape(-1),
            handedness[0][0], conf_val,
        ]


# ------------------------------------------------------------------
# Pipeline stage
# ------------------------------------------------------------------

class HandLandmarkerStage(Stage):
    """Detect hand landmarks using OpenCV DNN with MediaPipe ONNX models.

    Drop-in replacement for the previous mediapipe Tasks-based stage.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("hand_landmarker", config)
        self._palm_det: _PalmDetector | None = None
        self._hand_est: _HandPoseEstimator | None = None
        self._num_hands: int = 1

    def setup(self) -> None:
        palm_path = self.config.get(
            "palm_model_path", "models/palm_detection_mediapipe_2023feb.onnx",
        )
        hand_path = self.config.get(
            "hand_model_path", "models/handpose_estimation_mediapipe_2023feb.onnx",
        )
        _ensure_model(palm_path, _PALM_DET_URL)
        _ensure_model(hand_path, _HAND_POSE_URL)

        self._palm_det = _PalmDetector(
            palm_path,
            score_threshold=self.config.get("min_detection_confidence", 0.5),
        )
        self._hand_est = _HandPoseEstimator(
            hand_path,
            conf_threshold=self.config.get("min_tracking_confidence", 0.5),
        )
        self._num_hands = self.config.get("num_hands", 1)
        self._log.info("HandLandmarker ready (palm=%s, hand=%s)", palm_path, hand_path)

    def process(self, frame: np.ndarray) -> FrameResult:
        if frame is None:
            return FrameResult()

        bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        palms = self._palm_det.detect(bgr)
        hands: list[HandResult] = []

        for i, palm in enumerate(palms):
            if i >= self._num_hands:
                break
            result = self._hand_est.estimate(bgr, palm)
            if result is None:
                continue

            screen_lm = result[4:67].reshape(21, 3)
            pts: list[tuple[float, float, float]] = [
                (
                    float(screen_lm[j, 0] / w),
                    float(screen_lm[j, 1] / h),
                    float(screen_lm[j, 2] / max(w, h)),
                )
                for j in range(21)
            ]
            handedness_str = "Right" if result[130] > 0.5 else "Left"

            hands.append(HandResult(
                landmarks=pts,
                handedness=handedness_str,
                score=float(result[131]),
            ))

        return FrameResult(hands=hands, frame=frame)

    def cleanup(self) -> None:
        self._log.info("HandLandmarker closed.")


def _ensure_model(path: str, url: str) -> None:
    p = pathlib.Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s …", p.name)
    urllib.request.urlretrieve(url, str(p))
    logger.info("Download complete: %s", p)
