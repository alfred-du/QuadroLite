"""QuadroLite -- hand gesture CV pipeline entry point.

Usage:
    python -m src.main [--config CONFIG_PATH]
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any

import yaml

from src.capture.camera import CaptureStage
from src.dispatch.dispatcher import ActionHandler, DispatchStage
from src.dispatch.handlers.preview import PreviewHandler
from src.dispatch.handlers.servo import ServoHandler
from src.dispatch.handlers.terminal import TerminalHandler
from src.inference.gesture_classifier import GestureClassifierStage
from src.inference.hand_landmarker import HandLandmarkerStage
from src.pipeline.orchestrator import Orchestrator
from src.processing.preprocessor import PreprocessStage

DEFAULT_CONFIG = pathlib.Path("config/pipeline.yaml")

logger = logging.getLogger("quadrolite")


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_handlers(cfg: dict[str, Any]) -> list[ActionHandler]:
    dispatch_cfg = cfg.get("dispatch", {})
    handler_name = dispatch_cfg.get("handler", "terminal")
    handlers: list[ActionHandler] = []

    if handler_name == "servo":
        handlers.append(ServoHandler(config=dispatch_cfg.get("servo", {})))
    else:
        handlers.append(TerminalHandler())

    preview_cfg = dispatch_cfg.get("preview", {})
    if preview_cfg.get("enabled", False):
        handlers.append(PreviewHandler(port=preview_cfg.get("port", 8080)))

    return handlers


def main(config_path: pathlib.Path = DEFAULT_CONFIG) -> None:
    cfg = _load_config(config_path)

    cam_cfg = cfg.get("camera", {})
    pipe_cfg = cfg.get("pipeline", {})
    hl_cfg = cfg.get("hand_landmarker", {})
    gc_cfg = cfg.get("gesture_classifier", {})

    capture = CaptureStage(config=cam_cfg)
    preprocess = PreprocessStage(config=cam_cfg)
    hand_landmarker = HandLandmarkerStage(config=hl_cfg)
    gesture_classifier = GestureClassifierStage(config=gc_cfg)
    dispatch = DispatchStage(handlers=_build_handlers(cfg))

    # Segment 1 (capture thread): camera -> preprocess
    # Segment 2 (inference thread): hand landmarker -> gesture classifier
    # Segment 3 (dispatch thread): dispatch handler
    segments = [
        [capture, preprocess],
        [hand_landmarker, gesture_classifier],
        [dispatch],
    ]

    orchestrator = Orchestrator(
        segments=segments,
        queue_size=pipe_cfg.get("queue_size", 2),
    )
    orchestrator.start()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    _configure_logging()
    parser = argparse.ArgumentParser(description="QuadroLite CV pipeline")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help="Path to pipeline YAML config",
    )
    args = parser.parse_args()
    main(config_path=args.config)
