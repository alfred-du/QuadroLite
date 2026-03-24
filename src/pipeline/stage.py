from __future__ import annotations

import abc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Stage(abc.ABC):
    """Base class for every pipeline stage.

    Lifecycle:
        setup()   -> called once before the pipeline loop starts
        process() -> called for each item; returns the transformed item
        cleanup() -> called once after the pipeline loop ends (or on error)
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._log = logging.getLogger(f"{__name__}.{name}")

    def setup(self) -> None:
        """Allocate resources. Override in subclasses."""

    @abc.abstractmethod
    def process(self, item: Any) -> Any:
        """Transform *item* and return the result.

        Return ``None`` to signal that the item should be dropped.
        """

    def cleanup(self) -> None:
        """Release resources. Override in subclasses."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
