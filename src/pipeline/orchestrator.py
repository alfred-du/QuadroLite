from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from typing import Any, Callable, Sequence

from src.pipeline.stage import Stage
from src.utils.fps import FPSCounter
from src.utils.health import log_health

logger = logging.getLogger(__name__)


class _StaleDropQueue:
    """Bounded queue that silently drops the oldest item when full."""

    def __init__(self, maxsize: int = 2) -> None:
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)

    def put(self, item: Any) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def get(self, timeout: float = 0.1) -> Any:
        return self._q.get(timeout=timeout)


class Orchestrator:
    """Wires pipeline stages together with threads and queues.

    Stages are grouped into *segments*; each segment runs on its own thread
    and processes items sequentially through its stages.  Segments are
    connected by bounded queues with a stale-frame drop policy.
    """

    def __init__(
        self,
        segments: Sequence[Sequence[Stage]],
        queue_size: int = 2,
        health_interval: float = 5.0,
    ) -> None:
        self._segments = segments
        self._queue_size = queue_size
        self._health_interval = health_interval

        self._queues: list[_StaleDropQueue] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._fps = FPSCounter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Set up stages, spawn threads, and block until stopped."""
        self._install_signal_handlers()
        self._setup_stages()
        self._build_queues()
        self._spawn_threads()
        logger.info("Pipeline started (%d segments)", len(self._segments))
        try:
            self._monitor_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        """Signal all threads to finish and join them."""
        if self._stop.is_set():
            return
        logger.info("Stopping pipeline …")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._cleanup_stages()
        logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %s", signal.Signals(signum).name)
        self._stop.set()

    def _setup_stages(self) -> None:
        for segment in self._segments:
            for stage in segment:
                stage.setup()
                logger.debug("Set up %s", stage)

    def _cleanup_stages(self) -> None:
        for segment in self._segments:
            for stage in segment:
                try:
                    stage.cleanup()
                except Exception:
                    logger.exception("Error cleaning up %s", stage)

    def _build_queues(self) -> None:
        # One queue between each pair of adjacent segments.
        self._queues = [
            _StaleDropQueue(self._queue_size)
            for _ in range(len(self._segments) - 1)
        ]

    def _spawn_threads(self) -> None:
        for idx, segment in enumerate(self._segments):
            in_q = self._queues[idx - 1] if idx > 0 else None
            out_q = self._queues[idx] if idx < len(self._queues) else None
            is_last = idx == len(self._segments) - 1
            t = threading.Thread(
                target=self._segment_loop,
                args=(segment, in_q, out_q, is_last),
                name=f"seg-{idx}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def _segment_loop(
        self,
        stages: Sequence[Stage],
        in_q: _StaleDropQueue | None,
        out_q: _StaleDropQueue | None,
        is_last: bool,
    ) -> None:
        while not self._stop.is_set():
            # --- get input ---
            if in_q is not None:
                try:
                    item = in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
            else:
                item = None  # first segment generates its own data

            # --- run stages ---
            for stage in stages:
                try:
                    item = stage.process(item)
                except Exception:
                    logger.exception("Error in %s", stage)
                    item = None
                if item is None:
                    break

            if item is None:
                continue

            # --- push output ---
            if out_q is not None:
                out_q.put(item)

            if is_last:
                self._fps.tick()

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._health_interval)
            if not self._stop.is_set():
                log_health(self._fps.fps)
