"""ROS 2 node wrapper for the QuadroLite gesture pipeline.

Subscribes to a camera Image topic and publishes gesture results.
Requires: rclpy, sensor_msgs, std_msgs, cv_bridge  (available in a
sourced ROS 2 Jazzy environment on Ubuntu 24.04).

Usage (after sourcing ROS 2 and building/installing the workspace):
    ros2 run quadrolite gesture_node
    ros2 run quadrolite gesture_node --ros-args -p config_path:=config/pipeline.yaml

Or standalone:
    python3 -m src.ros2_node --ros-args -p config_path:=config/pipeline.yaml

The node can also run with its own OpenCV camera (no external camera
topic required) by setting the ``use_internal_camera`` parameter to true.
"""
from __future__ import annotations

import pathlib
from typing import Any

import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from src.inference.hand_landmarker import HandLandmarkerStage, FrameResult
from src.inference.gesture_classifier import GestureClassifierStage, GestureResult
from src.processing.preprocessor import PreprocessStage

DEFAULT_CONFIG = pathlib.Path("config/pipeline.yaml")


class GestureNode(Node):
    """Receives camera frames, runs gesture detection, publishes results."""

    def __init__(self) -> None:
        super().__init__("quadrolite_gesture")

        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("use_internal_camera", False)

        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        cfg = self._load_config(pathlib.Path(config_path))

        cam_cfg = cfg.get("camera", {})
        hl_cfg = cfg.get("hand_landmarker", {})
        gc_cfg = cfg.get("gesture_classifier", {})

        self._preprocess = PreprocessStage(config=cam_cfg)
        self._preprocess.setup()

        self._landmarker = HandLandmarkerStage(config=hl_cfg)
        self._landmarker.setup()

        self._classifier = GestureClassifierStage(config=gc_cfg)
        self._classifier.setup()

        self._bridge = CvBridge()

        self._gesture_pub = self.create_publisher(String, "~/gesture", 10)

        use_internal = (
            self.get_parameter("use_internal_camera")
            .get_parameter_value()
            .bool_value
        )

        if use_internal:
            self._start_internal_camera(cam_cfg)
        else:
            camera_topic = (
                self.get_parameter("camera_topic")
                .get_parameter_value()
                .string_value
            )
            self._cam_sub = self.create_subscription(
                Image, camera_topic, self._on_image, 1,
            )
            self.get_logger().info(f"Subscribed to {camera_topic}")

    # ------------------------------------------------------------------
    # Internal camera mode (no external ROS camera node needed)
    # ------------------------------------------------------------------

    def _start_internal_camera(self, cam_cfg: dict[str, Any]) -> None:
        from src.capture.camera import CaptureStage

        self._capture = CaptureStage(config={**cam_cfg, "backend": "opencv"})
        self._capture.setup()

        fps = cam_cfg.get("fps", 30)
        period = 1.0 / fps
        self._timer = self.create_timer(period, self._timer_callback)
        self.get_logger().info("Using internal OpenCV camera (no topic subscription)")

    def _timer_callback(self) -> None:
        try:
            frame = self._capture.process()
        except RuntimeError:
            return
        self._run_pipeline(frame)

    # ------------------------------------------------------------------
    # Topic-driven mode
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._run_pipeline(frame)

    # ------------------------------------------------------------------
    # Shared inference path
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame: np.ndarray) -> None:
        frame = self._preprocess.process(frame)
        if frame is None:
            return

        frame_result: FrameResult = self._landmarker.process(frame)
        gesture_result: GestureResult | None = self._classifier.process(frame_result)

        if gesture_result is None or gesture_result.gesture is None:
            return

        msg = String()
        msg.data = (
            f'{{"gesture": "{gesture_result.gesture}", '
            f'"action": "{gesture_result.action}", '
            f'"confidence": {gesture_result.confidence:.3f}, '
            f'"handedness": "{gesture_result.handedness}"}}'
        )
        self._gesture_pub.publish(msg)

        self.get_logger().info(
            f"[{gesture_result.handedness}] {gesture_result.gesture}"
            f" -> {gesture_result.action}  conf={gesture_result.confidence:.2f}",
            throttle_duration_sec=0.5,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: pathlib.Path) -> dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def destroy_node(self) -> None:
        if hasattr(self, "_capture"):
            self._capture.cleanup()
        self._landmarker.cleanup()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
