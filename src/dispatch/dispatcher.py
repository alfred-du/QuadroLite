from __future__ import annotations

import abc
import logging
from typing import Any

from src.inference.gesture_classifier import GestureResult
from src.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class ActionHandler(abc.ABC):
    """Base class for action handlers (terminal, servo, etc.)."""

    @abc.abstractmethod
    def handle(self, result: GestureResult) -> None: ...

    def cleanup(self) -> None:
        pass


class DispatchStage(Stage):
    """Routes ``GestureResult`` objects to the active action handler."""

    def __init__(
        self,
        handler: ActionHandler,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("dispatch", config)
        self._handler = handler

    def process(self, result: GestureResult) -> GestureResult:
        if result is None:
            return None  # type: ignore[return-value]
        self._handler.handle(result)
        return result

    def cleanup(self) -> None:
        self._handler.cleanup()
