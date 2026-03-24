from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.inference.hand_landmarker import FrameResult, HandResult
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass
class GestureResult:
    """Recognised gesture for a single frame."""

    gesture: str | None
    confidence: float
    action: str | None
    handedness: str
    frame: np.ndarray | None = None
    landmarks: list[tuple[float, float, float]] | None = None


def _is_finger_up(
    landmarks: list[tuple[float, float, float]],
    tip: int,
    pip_: int,
) -> bool:
    """A finger is 'up' when its tip is above (lower y) its PIP joint."""
    return landmarks[tip][1] < landmarks[pip_][1]


def _is_thumb_up(
    landmarks: list[tuple[float, float, float]],
    handedness: str,
) -> bool:
    """Thumb is 'out' when its tip is further from the palm centre than its IP.

    For a right hand (mirrored in camera) the thumb extends to the left
    (lower x), and vice-versa for a left hand.
    """
    tip_x = landmarks[THUMB_TIP][0]
    ip_x = landmarks[THUMB_IP][0]
    if handedness == "Right":
        return tip_x < ip_x
    return tip_x > ip_x


def _finger_states(hand: HandResult) -> tuple[bool, bool, bool, bool, bool]:
    """Return (thumb, index, middle, ring, pinky) up/down booleans."""
    lm = hand.landmarks
    return (
        _is_thumb_up(lm, hand.handedness),
        _is_finger_up(lm, INDEX_TIP, INDEX_PIP),
        _is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        _is_finger_up(lm, RING_TIP, RING_PIP),
        _is_finger_up(lm, PINKY_TIP, PINKY_PIP),
    )


_GESTURE_TABLE: dict[tuple[bool, bool, bool, bool, bool], str] = {
    (False, False, False, False, False): "fist",
    (True, True, True, True, True): "open_palm",
    (False, True, True, False, False): "peace",
    (True, False, False, False, False): "thumbs_up",
    (False, True, False, False, False): "point",
    (False, False, False, False, True): "pinky",
    (True, True, False, False, True): "rock",
}


class GestureClassifierStage(Stage):
    """Rule-based gesture classifier operating on HandLandmarker output."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("gesture_classifier", config)
        self._action_map: dict[str, str] = {}

    def setup(self) -> None:
        for entry in self.config.get("gestures", []):
            name = entry.get("name")
            action = entry.get("action")
            if name and action:
                self._action_map[name] = action
        self._log.info("Gesture-action map: %s", self._action_map)

    def process(self, frame_result: FrameResult) -> GestureResult | None:
        if frame_result is None:
            return None

        if not frame_result.hands:
            return GestureResult(
                gesture=None, confidence=0.0, action=None,
                handedness="", frame=frame_result.frame,
            )

        hand = frame_result.hands[0]
        states = _finger_states(hand)
        gesture = _GESTURE_TABLE.get(states)
        action = self._action_map.get(gesture) if gesture else None

        return GestureResult(
            gesture=gesture,
            confidence=hand.score,
            action=action,
            handedness=hand.handedness,
            frame=frame_result.frame,
            landmarks=hand.landmarks,
        )


def classify_hand(
    hand: HandResult, action_map: dict[str, str] | None = None
) -> GestureResult:
    """Standalone helper (useful in tests)."""
    states = _finger_states(hand)
    gesture = _GESTURE_TABLE.get(states)
    action = (action_map or {}).get(gesture) if gesture else None
    return GestureResult(
        gesture=gesture,
        confidence=hand.score,
        action=action,
        handedness=hand.handedness,
        landmarks=hand.landmarks,
    )
