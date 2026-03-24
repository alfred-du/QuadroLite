"""Integration smoke tests for pipeline plumbing (no camera required)."""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.pipeline.stage import Stage
from src.pipeline.orchestrator import Orchestrator, _StaleDropQueue
from src.utils.fps import FPSCounter


# ---------------------------------------------------------------------------
# StaleDropQueue
# ---------------------------------------------------------------------------

class TestStaleDropQueue:
    def test_basic_put_get(self) -> None:
        q = _StaleDropQueue(maxsize=2)
        q.put("a")
        q.put("b")
        assert q.get(timeout=1) == "a"
        assert q.get(timeout=1) == "b"

    def test_overflow_drops_oldest(self) -> None:
        q = _StaleDropQueue(maxsize=1)
        q.put("old")
        q.put("new")
        assert q.get(timeout=1) == "new"

    def test_get_timeout_raises(self) -> None:
        import queue
        q = _StaleDropQueue(maxsize=1)
        with pytest.raises(queue.Empty):
            q.get(timeout=0.05)


# ---------------------------------------------------------------------------
# FPSCounter
# ---------------------------------------------------------------------------

class TestFPSCounter:
    def test_zero_before_ticks(self) -> None:
        c = FPSCounter()
        assert c.fps == 0.0

    def test_single_tick(self) -> None:
        c = FPSCounter()
        c.tick()
        assert c.fps == 0.0

    def test_multiple_ticks(self) -> None:
        c = FPSCounter(window=5)
        for _ in range(5):
            c.tick()
            time.sleep(0.01)
        assert c.fps > 0


# ---------------------------------------------------------------------------
# Stage contract
# ---------------------------------------------------------------------------

class _DoubleStage(Stage):
    def process(self, item: Any) -> Any:
        return item * 2


class TestStage:
    def test_process(self) -> None:
        s = _DoubleStage("double")
        assert s.process(5) == 10

    def test_name(self) -> None:
        s = _DoubleStage("double")
        assert s.name == "double"


# ---------------------------------------------------------------------------
# Orchestrator (mini integration)
# ---------------------------------------------------------------------------

class _SourceStage(Stage):
    """Produces integers 0, 1, 2, …"""

    def __init__(self) -> None:
        super().__init__("source")
        self._counter = 0

    def process(self, _item: Any = None) -> int:
        val = self._counter
        self._counter += 1
        time.sleep(0.02)
        return val


class _CollectorStage(Stage):
    """Collects items into a shared list for assertions."""

    def __init__(self, bucket: list[Any]) -> None:
        super().__init__("collector")
        self._bucket = bucket

    def process(self, item: Any) -> Any:
        self._bucket.append(item)
        return item


class TestOrchestrator:
    def test_items_flow_through(self) -> None:
        bucket: list[int] = []
        source = _SourceStage()
        double = _DoubleStage("double")
        collector = _CollectorStage(bucket)

        orch = Orchestrator(
            segments=[[source], [double], [collector]],
            queue_size=2,
            health_interval=60,
        )

        runner = threading.Thread(target=orch.start, daemon=True)
        runner.start()
        time.sleep(0.5)
        orch.stop()
        runner.join(timeout=3)

        assert len(bucket) > 0
        for val in bucket:
            assert val % 2 == 0  # all items were doubled
