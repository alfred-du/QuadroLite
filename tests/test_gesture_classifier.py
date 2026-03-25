"""Unit tests for the rule-based gesture classifier."""
from __future__ import annotations

import pytest

from src.inference.gesture_classifier import (
    GestureResult,
    classify_hand,
    _finger_states,
)
from src.inference.hand_landmarker import HandResult

# ---------------------------------------------------------------------------
# Helpers — build synthetic 21-landmark hands
# ---------------------------------------------------------------------------

# Landmark y: lower value = higher in image = finger extended
_DOWN = 0.8  # finger curled (tip below PIP)
_UP = 0.2    # finger extended (tip above PIP)
_PIP_Y = 0.5  # reference PIP y for all fingers

# INDEX_MCP (landmark 5) is used as the distance reference for thumb detection.
_INDEX_MCP_POS = (0.5, 0.5, 0.0)


def _make_landmarks(
    thumb_up: bool,
    index_up: bool,
    middle_up: bool,
    ring_up: bool,
    pinky_up: bool,
) -> list[tuple[float, float, float]]:
    """Build a minimal 21-point landmark list with controllable finger states.

    Thumb detection is distance-based (no handedness dependency):
    extended  → THUMB_TIP far from INDEX_MCP, farther than THUMB_IP
    curled    → THUMB_TIP close to INDEX_MCP, closer than THUMB_IP
    """
    lm = [(0.5, 0.5, 0.0)] * 21

    lm[5] = _INDEX_MCP_POS  # INDEX_MCP — distance reference for thumb

    # THUMB_IP sits at a moderate distance from INDEX_MCP
    lm[3] = (0.35, 0.5, 0.0)  # THUMB_IP
    if thumb_up:
        lm[4] = (0.15, 0.5, 0.0)  # THUMB_TIP far from INDEX_MCP → extended
    else:
        lm[4] = (0.42, 0.5, 0.0)  # THUMB_TIP closer than THUMB_IP → curled

    # Index
    lm[6] = (0.5, _PIP_Y, 0.0)        # INDEX_PIP
    lm[8] = (0.5, _UP if index_up else _DOWN, 0.0)

    # Middle
    lm[10] = (0.5, _PIP_Y, 0.0)       # MIDDLE_PIP
    lm[12] = (0.5, _UP if middle_up else _DOWN, 0.0)

    # Ring
    lm[14] = (0.5, _PIP_Y, 0.0)       # RING_PIP
    lm[16] = (0.5, _UP if ring_up else _DOWN, 0.0)

    # Pinky
    lm[18] = (0.5, _PIP_Y, 0.0)       # PINKY_PIP
    lm[20] = (0.5, _UP if pinky_up else _DOWN, 0.0)

    return lm


def _hand(
    thumb: bool,
    index: bool,
    middle: bool,
    ring: bool,
    pinky: bool,
    handedness: str = "Right",
) -> HandResult:
    return HandResult(
        landmarks=_make_landmarks(thumb, index, middle, ring, pinky),
        handedness=handedness,
        score=0.95,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "fist": "stop",
    "open_palm": "forward",
}


class TestFingerStates:
    def test_all_down(self) -> None:
        h = _hand(False, False, False, False, False)
        assert _finger_states(h) == (False, False, False, False, False)

    def test_all_up(self) -> None:
        h = _hand(True, True, True, True, True)
        assert _finger_states(h) == (True, True, True, True, True)

    def test_partial_combo(self) -> None:
        h = _hand(False, True, True, False, False)
        assert _finger_states(h) == (False, True, True, False, False)


class TestClassifyHand:
    def test_fist(self) -> None:
        r = classify_hand(_hand(False, False, False, False, False), ACTION_MAP)
        assert r.gesture == "fist"
        assert r.action == "stop"

    def test_open_palm(self) -> None:
        r = classify_hand(_hand(True, True, True, True, True), ACTION_MAP)
        assert r.gesture == "open_palm"
        assert r.action == "forward"

    def test_unknown_combo_returns_none(self) -> None:
        r = classify_hand(_hand(True, True, True, True, False), ACTION_MAP)
        assert r.gesture is None
        assert r.action is None

    def test_thumb_detection_is_handedness_independent(self) -> None:
        """Both Right and Left hands should detect the same thumb state
        since detection is distance-based, not x-direction-based."""
        r_right = classify_hand(_hand(True, True, True, True, True, "Right"), ACTION_MAP)
        r_left = classify_hand(_hand(True, True, True, True, True, "Left"), ACTION_MAP)
        assert r_right.gesture == "open_palm"
        assert r_left.gesture == "open_palm"

    def test_confidence_propagates(self) -> None:
        h = _hand(False, False, False, False, False)
        r = classify_hand(h)
        assert r.confidence == pytest.approx(0.95)
