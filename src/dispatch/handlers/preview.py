"""MJPEG HTTP preview handler.

Serves the camera feed with landmark overlay at ``http://<pi-ip>:<port>/``.
Open this URL in a browser on any device on the same network.
"""
from __future__ import annotations

import logging
import threading
import time
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import cv2 as cv
import numpy as np

from src.dispatch.dispatcher import ActionHandler
from src.inference.gesture_classifier import GestureResult

logger = logging.getLogger(__name__)

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _draw_overlay(frame: np.ndarray, result: GestureResult) -> np.ndarray:
    """Draw landmarks, connections, and gesture label on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    if result.landmarks:
        pts = [(int(x * w), int(y * h)) for x, y, _ in result.landmarks]
        for a, b in _HAND_CONNECTIONS:
            cv.line(out, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv.circle(out, pt, 4, (0, 0, 255), -1)

    label_parts: list[str] = []
    if result.gesture:
        label_parts.append(result.gesture)
    if result.action:
        label_parts.append(f"-> {result.action}")
    if label_parts:
        label = " ".join(label_parts)
        cv.putText(out, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv.LINE_AA)

    return out


class _StreamHandler(BaseHTTPRequestHandler):
    """Serves a single MJPEG stream."""

    preview: PreviewHandler  # set by partial/factory

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = self.preview.get_jpeg()
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress per-request logs


class PreviewHandler(ActionHandler):
    """Dispatch handler that serves an MJPEG stream over HTTP."""

    def __init__(self, port: int = 8080) -> None:
        self._port = port
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._start_server()

    def _start_server(self) -> None:
        handler_cls = type(
            "_BoundStreamHandler",
            (_StreamHandler,),
            {"preview": self},
        )
        self._server = HTTPServer(("0.0.0.0", self._port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        logger.info("Preview server started on http://0.0.0.0:%d/", self._port)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def handle(self, result: GestureResult) -> None:
        if result.frame is None:
            return
        frame = _draw_overlay(result.frame, result)
        _, buf = cv.imencode(".jpg", cv.cvtColor(frame, cv.COLOR_RGB2BGR),
                              [cv.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self._jpeg = buf.tobytes()

    def cleanup(self) -> None:
        if self._server:
            self._server.shutdown()
            logger.info("Preview server stopped.")
