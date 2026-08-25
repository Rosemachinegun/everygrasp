#!/usr/bin/env python3
"""通信层：把 FlowPose 结果发布到 ROS2 TF 和 RViz marker。

这里只负责 ROS2 桥接，不做感知推理、坐标规划或任务决策。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grasp_core.perception.flowpose_pipeline import FlowPoseObject, FlowPoseResult  # noqa: E402


@dataclass(frozen=True)
class ObjectPose:
    label: str
    frame_id: str
    pose: np.ndarray
    size: np.ndarray | None = None
    score: float | None = None


class FlowPoseRosBridge:
    """Continuously republishes latest FlowPose objects as ROS2 TF + markers."""

    def __init__(
        self,
        *,
        node_name: str = "flowpose_ros_bridge",
        parent_frame_id: str = "camera_rgb_link",
        base_frame_id: str = "base_link",
        tf_topic: str = "/tf",
        marker_topic: str = "/flowpose/grasp_markers",
        target_pose_topic: str = "/flowpose/target_pose_base",
        target_point_topic: str = "/flowpose/target_point_base",
        target_poses_topic: str = "/flowpose/target_poses_base",
        publish_rate_hz: float = 10.0,
        pregrasp_distance_m: float = 0.10,
        lift_distance_m: float = 0.08,
        approach_axis: str = "z",
        approach_sign: float = -1.0,
        base_to_camera: np.ndarray | None = None,
        start_background_spin: bool = True,
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
            from rclpy.node import Node
            from rclpy.duration import Duration
            from rclpy.time import Time
            from tf2_ros import Buffer, TransformListener
            from tf2_ros import StaticTransformBroadcaster
            from tf2_msgs.msg import TFMessage
            from visualization_msgs.msg import MarkerArray
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python packages are not importable. Source your ROS2 workspace "
                "before enabling FlowPose ROS publishing."
            ) from exc

        self.rclpy = rclpy
        self.Node = Node
        self.Duration = Duration
        self.Time = Time
        self.PointStamped = PointStamped
        self.PoseArray = PoseArray
        self.PoseStamped = PoseStamped
        self.StaticTransformBroadcaster = StaticTransformBroadcaster
        self.TFMessage = TFMessage
        self.MarkerArray = MarkerArray
        self.parent_frame_id = parent_frame_id
        self.base_frame_id = base_frame_id
        self.base_to_camera = (
            np.asarray(base_to_camera, dtype=np.float64)
            if base_to_camera is not None
            else None
        )
        self.pregrasp_distance_m = pregrasp_distance_m
        self.lift_distance_m = lift_distance_m
        self.approach_axis = approach_axis
        self.approach_sign = float(approach_sign)

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.tf_pub = self.node.create_publisher(TFMessage, tf_topic, 10)
        self.marker_pub = self.node.create_publisher(MarkerArray, marker_topic, 10)
        self.target_pose_pub = self.node.create_publisher(PoseStamped, target_pose_topic, 10)
        self.target_point_pub = self.node.create_publisher(PointStamped, target_point_topic, 10)
        self.target_poses_pub = self.node.create_publisher(PoseArray, target_poses_topic, 10)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self._lock = threading.Lock()
        self._objects: list[ObjectPose] = []
        self._last_tf_warn_monotonic = 0.0
        self._timer = self.node.create_timer(
            1.0 / max(float(publish_rate_hz), 0.1), self._publish_latest
        )
        self._publish_static_camera_transform()
        self._spin_thread: threading.Thread | None = None
        if start_background_spin:
            self._spin_thread = threading.Thread(
                target=self.rclpy.spin, args=(self.node,), daemon=True
            )
            self._spin_thread.start()

    def update_result(self, result: FlowPoseResult) -> None:
        self.update_objects(result.objects)

    def update_objects(self, objects: Iterable[FlowPoseObject | ObjectPose]) -> None:
        object_list: list[FlowPoseObject | ObjectPose] = list(objects)
        frame_ids = make_child_frame_ids(
            obj.frame_id if isinstance(obj, ObjectPose) else obj.name
            for obj in object_list
        )
        converted: list[ObjectPose] = []
        for obj, frame_id in zip(object_list, frame_ids, strict=False):
            if isinstance(obj, ObjectPose):
                pose = obj.pose
                label = obj.label
                size = obj.size
                score = obj.score
            else:
                pose = np.asarray(obj.pose, dtype=np.float64)
                label = obj.name
                size = np.asarray(obj.size, dtype=np.float64) if obj.size else None
                score = obj.score
            if pose.shape != (4, 4):
                self.node.get_logger().warning(
                    f"Skip {label}: expected 4x4 pose, got {pose.shape}"
                )
                continue
            converted.append(
                ObjectPose(
                    label=label,
                    frame_id=frame_id,
                    pose=pose,
                    size=size,
                    score=score,
                )
            )
        with self._lock:
            self._objects = converted
        if converted:
            frames = ", ".join(obj.frame_id for obj in converted)
            self.node.get_logger().info(f"Updated FlowPose TF objects: {frames}")

    def update_pose_all(
        self,
        labels: Iterable[str],
        pose_all: Iterable[Any],
        length_all: Iterable[Any] | None = None,
    ) -> None:
        sizes = list(length_all) if length_all is not None else []
        objects: list[ObjectPose] = []
        for index, (label, pose) in enumerate(zip(labels, pose_all, strict=False)):
            size = np.asarray(sizes[index], dtype=np.float64) if index < len(sizes) else None
            objects.append(
                ObjectPose(
                    label=str(label),
                    frame_id=str(label),
                    pose=np.asarray(pose, dtype=np.float64),
                    size=size,
                )
            )
        self.update_objects(objects)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        self._timer.cancel()
        self.node.destroy_node()
        self.rclpy.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

    def _publish_latest(self) -> None:
        self._publish_static_camera_transform()
        with self._lock:
            objects = list(self._objects)
        if not objects:
            return
        now = self.node.get_clock().now().to_msg()
        transforms = [
            make_transform_stamped(
                obj.pose, self.parent_frame_id, obj.frame_id, now
            )
            for obj in objects
        ]
        self.tf_pub.publish(self.TFMessage(transforms=transforms))

        marker_array = self._make_grasp_markers(objects, now)
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
        self._publish_base_target_topics(objects, now)

    def _publish_static_camera_transform(self) -> None:
        if self.base_to_camera is None:
            return
        stamp = self.node.get_clock().now().to_msg()
        self.static_tf_broadcaster.sendTransform(
            make_transform_stamped(
                self.base_to_camera,
                self.base_frame_id,
                self.parent_frame_id,
                stamp,
            )
        )

    def _publish_base_target_topics(self, objects: list[ObjectPose], stamp: Any) -> None:
        base_to_camera = self._base_to_camera_matrix()
        if base_to_camera is None:
            return
        base_poses = [(obj, base_to_camera @ obj.pose) for obj in objects]
        pose_array = self.PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.base_frame_id
        for _, pose in base_poses:
            pose_array.poses.append(make_pose_msg(pose))
        self.target_poses_pub.publish(pose_array)

        selected, selected_pose = base_poses[0]
        pose_stamped = self.PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = self.base_frame_id
        pose_stamped.pose = make_pose_msg(selected_pose)
        self.target_pose_pub.publish(pose_stamped)

        point_stamped = self.PointStamped()
        point_stamped.header.stamp = stamp
        point_stamped.header.frame_id = self.base_frame_id
        point_stamped.point.x = float(selected_pose[0, 3])
        point_stamped.point.y = float(selected_pose[1, 3])
        point_stamped.point.z = float(selected_pose[2, 3])
        self.target_point_pub.publish(point_stamped)

    def _make_grasp_markers(self, objects: list[ObjectPose], stamp: Any) -> Any:
        from visualization_msgs.msg import Marker

        marker_array = self.MarkerArray()
        base_to_camera = self._base_to_camera_matrix()
        if base_to_camera is None:
            return marker_array

        marker_id = 0
        lifetime = self.Duration(seconds=1.0).to_msg()
        for object_index, obj in enumerate(objects):
            pose_base = base_to_camera @ obj.pose
            grasp = pose_base[:3, 3]
            approach_dir = self._approach_direction(pose_base)
            pregrasp = grasp - approach_dir * self.pregrasp_distance_m
            lift = grasp + np.array([0.0, 0.0, self.lift_distance_m], dtype=np.float64)
            points = [pregrasp, grasp, lift]

            color = marker_color(object_index)
            marker_id = add_sphere_markers(
                marker_array,
                marker_id,
                stamp,
                self.base_frame_id,
                obj.frame_id,
                points,
                color,
                lifetime,
            )
            marker_id = add_path_marker(
                marker_array,
                marker_id,
                stamp,
                self.base_frame_id,
                obj.frame_id,
                points,
                color,
                lifetime,
            )
            marker_id = add_axes_markers(
                marker_array,
                marker_id,
                stamp,
                self.base_frame_id,
                obj.frame_id,
                pose_base,
                lifetime,
            )
        return marker_array

    def _base_to_camera_matrix(self) -> np.ndarray | None:
        if self.base_to_camera is not None:
            return self.base_to_camera
        return self._lookup_matrix(self.base_frame_id, self.parent_frame_id)

    def _lookup_matrix(self, target_frame: str, source_frame: str) -> np.ndarray | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                self.Time(),
                timeout=self.Duration(seconds=0.02),
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_tf_warn_monotonic > 2.0:
                self._last_tf_warn_monotonic = now
                self.node.get_logger().warn(
                    f"Waiting for TF {target_frame} <- {source_frame}: {exc}"
                )
            return None
        return transform_to_matrix(transform.transform)

    def _approach_direction(self, pose_base: np.ndarray) -> np.ndarray:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        index = axis_to_index.get(self.approach_axis.lower(), 2)
        direction = pose_base[:3, index] * self.approach_sign
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return direction / norm


def make_child_frame_ids(labels: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    frame_ids: list[str] = []
    for label in labels:
        base = normalize_label(label)
        counts[base] = counts.get(base, 0) + 1
        frame_ids.append(f"{base}_{counts[base]}")
    return frame_ids


def normalize_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    text = re.sub(r"_\d+$", "", text)
    return text or "object"


def make_transform_stamped(
    pose: np.ndarray,
    parent_frame_id: str,
    child_frame_id: str,
    stamp: Any,
) -> Any:
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent_frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = float(pose[0, 3])
    transform.transform.translation.y = float(pose[1, 3])
    transform.transform.translation.z = float(pose[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(pose[:3, :3])
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


def make_pose_msg(pose_matrix: np.ndarray) -> Any:
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(pose_matrix[0, 3])
    pose.position.y = float(pose_matrix[1, 3])
    pose.position.z = float(pose_matrix[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(pose_matrix[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return tuple(float(value) for value in quat)


def transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    matrix = quaternion_to_matrix(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    matrix[0, 3] = float(translation.x)
    matrix[1, 3] = float(translation.y)
    matrix[2, 3] = float(translation.z)
    return matrix


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def marker_color(index: int) -> tuple[float, float, float, float]:
    palette = [
        (0.1, 0.6, 1.0, 0.95),
        (1.0, 0.58, 0.05, 0.95),
        (0.0, 0.75, 0.45, 0.95),
        (0.92, 0.22, 0.14, 0.95),
        (0.78, 0.38, 0.70, 0.95),
    ]
    return palette[index % len(palette)]


def set_marker_header(marker: Any, stamp: Any, frame_id: str, ns: str, marker_id: int) -> None:
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id


def set_color(marker: Any, color: tuple[float, float, float, float]) -> None:
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])


def add_sphere_markers(
    marker_array: Any,
    marker_id: int,
    stamp: Any,
    frame_id: str,
    name: str,
    points: list[np.ndarray],
    color: tuple[float, float, float, float],
    lifetime: Any,
) -> int:
    from visualization_msgs.msg import Marker

    for point_index, point in enumerate(points):
        marker = Marker()
        set_marker_header(marker, stamp, frame_id, f"{name}_points", marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = float(point[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025 if point_index == 1 else 0.018
        marker.scale.y = marker.scale.x
        marker.scale.z = marker.scale.x
        marker.lifetime = lifetime
        set_color(marker, color)
        marker_array.markers.append(marker)
        marker_id += 1
    return marker_id


def add_path_marker(
    marker_array: Any,
    marker_id: int,
    stamp: Any,
    frame_id: str,
    name: str,
    points: list[np.ndarray],
    color: tuple[float, float, float, float],
    lifetime: Any,
) -> int:
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    marker = Marker()
    set_marker_header(marker, stamp, frame_id, f"{name}_path", marker_id)
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.006
    marker.lifetime = lifetime
    set_color(marker, color)
    for point in points:
        msg_point = Point()
        msg_point.x = float(point[0])
        msg_point.y = float(point[1])
        msg_point.z = float(point[2])
        marker.points.append(msg_point)
    marker_array.markers.append(marker)
    return marker_id + 1


def add_axes_markers(
    marker_array: Any,
    marker_id: int,
    stamp: Any,
    frame_id: str,
    name: str,
    pose: np.ndarray,
    lifetime: Any,
) -> int:
    colors = [
        (1.0, 0.05, 0.05, 0.95),
        (0.05, 0.85, 0.05, 0.95),
        (0.1, 0.35, 1.0, 0.95),
    ]
    origin = pose[:3, 3]
    for axis_index, color in enumerate(colors):
        marker_id = add_arrow_marker(
            marker_array,
            marker_id,
            stamp,
            frame_id,
            f"{name}_axis",
            origin,
            origin + pose[:3, axis_index] * 0.055,
            color,
            lifetime,
        )
    return marker_id


def add_arrow_marker(
    marker_array: Any,
    marker_id: int,
    stamp: Any,
    frame_id: str,
    ns: str,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
    lifetime: Any,
) -> int:
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    marker = Marker()
    set_marker_header(marker, stamp, frame_id, ns, marker_id)
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.008
    marker.scale.y = 0.016
    marker.scale.z = 0.016
    marker.lifetime = lifetime
    set_color(marker, color)
    for value in (start, end):
        point = Point()
        point.x = float(value[0])
        point.y = float(value[1])
        point.z = float(value[2])
        marker.points.append(point)
    marker_array.markers.append(marker)
    return marker_id + 1


def load_flowpose_objects(path: Path) -> list[FlowPoseObject]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects") or []
    result: list[FlowPoseObject] = []
    for index, obj in enumerate(objects):
        result.append(
            FlowPoseObject(
                name=str(obj.get("name") or f"object_{index + 1}"),
                obj_id=list(obj.get("obj_id") or [index + 1, index + 1]),
                pose=obj["pose"],
                size=list(obj.get("size") or []),
                score=obj.get("score"),
            )
        )
    if result:
        return result

    labels = payload.get("labels") or []
    pose_all = payload.get("pose_all") or []
    length_all = payload.get("length_all") or []
    frame_ids = make_child_frame_ids(labels or [f"object_{i + 1}" for i in range(len(pose_all))])
    for index, pose in enumerate(pose_all):
        result.append(
            FlowPoseObject(
                name=frame_ids[index],
                obj_id=[index + 1, index + 1],
                pose=pose,
                size=length_all[index] if index < len(length_all) else [],
                score=None,
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Republish saved FlowPose results as ROS2 TF and grasp markers."
    )
    parser.add_argument("--flowpose-json", required=True, type=Path)
    parser.add_argument("--parent-frame-id", default="camera_rgb_link")
    parser.add_argument("--base-frame-id", default="base_link")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--marker-topic", default="/flowpose/grasp_markers")
    parser.add_argument("--publish-rate-hz", type=float, default=10.0)
    parser.add_argument("--pregrasp-distance-m", type=float, default=0.10)
    parser.add_argument("--lift-distance-m", type=float, default=0.08)
    parser.add_argument("--approach-axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--approach-sign", type=float, default=-1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge = FlowPoseRosBridge(
        parent_frame_id=args.parent_frame_id,
        base_frame_id=args.base_frame_id,
        tf_topic=args.tf_topic,
        marker_topic=args.marker_topic,
        publish_rate_hz=args.publish_rate_hz,
        pregrasp_distance_m=args.pregrasp_distance_m,
        lift_distance_m=args.lift_distance_m,
        approach_axis=args.approach_axis,
        approach_sign=args.approach_sign,
        start_background_spin=False,
    )
    bridge.update_objects(load_flowpose_objects(args.flowpose_json))
    try:
        bridge.rclpy.spin(bridge.node)
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
