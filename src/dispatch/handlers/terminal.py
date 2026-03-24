from __future__ import annotations

import logging
import time

from src.dispatch.dispatcher import ActionHandler
from src.inference.gesture_classifier import GestureResult

logger = logging.getLogger(__name__)


class TerminalHandler(ActionHandler):
    """Prints recognised gestures to stdout.

    Includes basic throttling so the terminal isn't flooded when the same
    gesture is held continuously.
    """

    def __init__(self, cooldown: float = 0.5) -> None:
        self._cooldown = cooldown
        self._last_gesture: str | None = None
        self._last_time: float = 0.0

    def handle(self, result: GestureResult) -> None:
        now = time.monotonic()
        same = result.gesture == self._last_gesture
        if same and (now - self._last_time) < self._cooldown:
            return

        self._last_gesture = result.gesture
        self._last_time = now

        if result.gesture is None:
            return

        action_str = f" -> {result.action}" if result.action else ""
        logger.info(
            "[%s] gesture=%s  conf=%.2f%s",
            result.handedness,
            result.gesture,
            result.confidence,
            action_str,
        )
