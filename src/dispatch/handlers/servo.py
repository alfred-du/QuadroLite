from __future__ import annotations

import logging
from typing import Any

from src.dispatch.dispatcher import ActionHandler
from src.inference.gesture_classifier import GestureResult

logger = logging.getLogger(__name__)


class ServoHandler(ActionHandler):
    """Placeholder for future PWM servo control.

    Maps gesture actions to servo movements. Requires ``RPi.GPIO`` or
    ``pigpio`` at runtime on the Raspberry Pi.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._pin_pan: int = self._config.get("pin_pan", 12)
        self._pin_tilt: int = self._config.get("pin_tilt", 13)

    def handle(self, result: GestureResult) -> None:
        if result.action is None:
            return
        logger.debug(
            "ServoHandler: action=%s (pan=%d, tilt=%d) -- NOT IMPLEMENTED",
            result.action,
            self._pin_pan,
            self._pin_tilt,
        )

    def cleanup(self) -> None:
        logger.debug("ServoHandler cleanup (no-op).")
