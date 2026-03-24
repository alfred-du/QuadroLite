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
    """Routes ``GestureResult`` objects to one or more action handlers."""

    def __init__(
        self,
        handler: ActionHandler | None = None,
        config: dict[str, Any] | None = None,
        handlers: list[ActionHandler] | None = None,
    ) -> None:
        super().__init__("dispatch", config)
        self._handlers: list[ActionHandler] = list(handlers or [])
        if handler is not None:
            self._handlers.append(handler)

    def process(self, result: GestureResult) -> GestureResult:
        if result is None:
            return None  # type: ignore[return-value]
        for h in self._handlers:
            h.handle(result)
        return result

    def cleanup(self) -> None:
        for h in self._handlers:
            h.cleanup()
