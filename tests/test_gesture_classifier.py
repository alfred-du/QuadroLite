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

# Landmark x positions for thumb logic
_THUMB_IP_X = 0.5


def _make_landmarks(
    thumb_up: bool,
    index_up: bool,
    middle_up: bool,
    ring_up: bool,
    pinky_up: bool,
    handedness: str = "Right",
) -> list[tuple[float, float, float]]:
    """Build a minimal 21-point landmark list with controllable finger states."""
    lm = [(0.5, 0.5, 0.0)] * 21

    # Thumb: tip x relative to IP x determines up/down.
    # Right hand: thumb up => tip_x < ip_x
    if handedness == "Right":
        thumb_tip_x = 0.3 if thumb_up else 0.7
    else:
        thumb_tip_x = 0.7 if thumb_up else 0.3

    lm[3] = (_THUMB_IP_X, 0.5, 0.0)   # THUMB_IP
    lm[4] = (thumb_tip_x, 0.5, 0.0)   # THUMB_TIP

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
        landmarks=_make_landmarks(thumb, index, middle, ring, pinky, handedness),
        handedness=handedness,
        score=0.95,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "fist": "forward",
    "open_palm": "stop",
    "peace": "wave",
    "thumbs_up": "nod",
    "point": "look",
}


class TestFingerStates:
    def test_all_down(self) -> None:
        h = _hand(False, False, False, False, False)
        assert _finger_states(h) == (False, False, False, False, False)

    def test_all_up(self) -> None:
        h = _hand(True, True, True, True, True)
        assert _finger_states(h) == (True, True, True, True, True)

    def test_peace(self) -> None:
        h = _hand(False, True, True, False, False)
        assert _finger_states(h) == (False, True, True, False, False)


class TestClassifyHand:
    def test_fist(self) -> None:
        r = classify_hand(_hand(False, False, False, False, False), ACTION_MAP)
        assert r.gesture == "fist"
        assert r.action == "forward"

    def test_open_palm(self) -> None:
        r = classify_hand(_hand(True, True, True, True, True), ACTION_MAP)
        assert r.gesture == "open_palm"
        assert r.action == "stop"

    def test_peace(self) -> None:
        r = classify_hand(_hand(False, True, True, False, False), ACTION_MAP)
        assert r.gesture == "peace"
        assert r.action == "wave"

    def test_thumbs_up(self) -> None:
        r = classify_hand(_hand(True, False, False, False, False), ACTION_MAP)
        assert r.gesture == "thumbs_up"
        assert r.action == "nod"

    def test_point(self) -> None:
        r = classify_hand(_hand(False, True, False, False, False), ACTION_MAP)
        assert r.gesture == "point"
        assert r.action == "look"

    def test_unknown_combo(self) -> None:
        r = classify_hand(_hand(True, True, True, True, False), ACTION_MAP)
        assert r.gesture is None
        assert r.action is None

    def test_left_hand_thumb(self) -> None:
        r = classify_hand(
            _hand(True, False, False, False, False, handedness="Left"),
            ACTION_MAP,
        )
        assert r.gesture == "thumbs_up"

    def test_confidence_propagates(self) -> None:
        h = _hand(False, False, False, False, False)
        r = classify_hand(h)
        assert r.confidence == pytest.approx(0.95)
