#!/usr/bin/env python3
"""任务层：抓取成功后的固定区域放置动作。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.communication.request_ik_publisher import (
    RequestIkTargetPublisher,
    publish_home_request_ik_target,
    publish_request_ik_path,
)
from grasp_core.core.pose_math import (
    PoseWaypoint,
    checked_position,
    ik_wrist_orientation_quat,
    normalize_object_type,
    quaternion_to_rotation_matrix,
    rotation_matrix_from_zyx_euler_deg,
)
from grasp_core.core.robot_target_pose import matrix_to_quaternion

FIXED_PUT_RIGHT_XYZ = (0.54, -0.30, 0.826)
# FIXED_PUT_RIGHT_XYZ = (0.54, -0.30, 0.776)
FIXED_PUT_OBJECT_RIGHT_XYZ = {
    "yellow_cube": (0.40, -0.40, 0.86),
    "yellow_duck": (0.37, -0.33, 0.80),
    "blue_cube": (0.35, -0.35, 0.83),
}


@dataclass(frozen=True)
class FixedPutResult:
    ok: bool
    status: str


def humanlike_put_waypoints(
    publisher: RequestIkTargetPublisher,
    hand: str,
    target_position: np.ndarray,
    target_orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> list[PoseWaypoint]:
    """Build a gentle segmented cubic-spline place path.

    The motion layer turns these sparse waypoints into cubic Hermite segments.
    Keeping this planner in put.py limits the behavior change to grasp -> place.
    """

    end_position = checked_position(target_position)
    remembered = publisher.remembered_target(hand)
    if remembered is None:
        return [(end_position.copy(), target_orientation)]

    start_position, _start_orientation = remembered
    start_position = checked_position(start_position)
    delta = end_position - start_position
    distance_m = float(np.linalg.norm(delta))
    if distance_m < 1e-4:
        return [(end_position.copy(), target_orientation)]

    max_endpoint_z = max(float(start_position[2]), float(end_position[2]))
    lift_m = min(max(0.18 * distance_m, 0.06), 0.14)
    safe_z_m = max(
        float(getattr(args, "home_safe_z_m", 0.95)),
        max_endpoint_z + 0.04,
    )
    arc_z = max(max_endpoint_z + lift_m, safe_z_m)

    place_orientation = put_outward_z_axis_orientation(hand, target_orientation)

    mid1 = start_position + 0.25 * delta
    mid2 = start_position + 0.65 * delta

    mid1[2] = start_position[2] + 0.65 * (arc_z - start_position[2])
    pre_place_lift_m = min(max(0.10 * distance_m, 0.06), 0.10)
    pre_place_z = min(
        float(end_position[2]) + pre_place_lift_m,
        max(arc_z - 0.02, float(end_position[2]) + 0.04),
    )
    mid2[2] = min(arc_z, max(arc_z - 0.03, pre_place_z + 0.02))

    pre_place = end_position.copy()
    pre_place[2] = pre_place_z

    waypoints = [
        (mid1, place_orientation),
        (mid2, place_orientation),
        (pre_place, place_orientation),
        (end_position.copy(), place_orientation),
    ]
    return compact_waypoints(waypoints)


def put_outward_z_axis_orientation(
    hand: str,
    base_orientation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    hand_sign = 1.0 if hand == "left" else -1.0
    local_z_yaw = rotation_matrix_from_zyx_euler_deg(yaw_deg=hand_sign * 20.0)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_rotation_matrix(base_orientation) @ local_z_yaw
    return matrix_to_quaternion(pose)


def compact_waypoints(waypoints: list[PoseWaypoint]) -> list[PoseWaypoint]:
    compacted: list[PoseWaypoint] = []
    last_position: np.ndarray | None = None
    for position, orientation in waypoints:
        checked = checked_position(position).copy()
        if (
            last_position is not None
            and np.linalg.norm(checked - last_position) < 1e-4
        ):
            continue
        compacted.append((checked, orientation))
        last_position = checked
    return compacted


def fixed_put_xyz_for_hand(
    hand: str,
    object_type: str | None = None,
) -> tuple[float, float, float]:
    hand_name = normalize_hand(hand)
    if object_type:
        object_name = normalize_object_type(object_type)
        if object_name in FIXED_PUT_OBJECT_RIGHT_XYZ:
            return mirror_right_xyz_for_hand(
                FIXED_PUT_OBJECT_RIGHT_XYZ[object_name],
                hand_name,
            )

    return mirror_right_xyz_for_hand(FIXED_PUT_RIGHT_XYZ, hand_name)


def normalize_hand(hand: str) -> str:
    return "left" if str(hand).strip().lower() == "left" else "right"


def mirror_right_xyz_for_hand(
    right_xyz: tuple[float, float, float],
    hand: str,
) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in right_xyz)
    if hand == "left":
        return x, abs(y), z
    return x, -abs(y), z


def publisher_stop_requested(publisher: RequestIkTargetPublisher) -> bool:
    stop_requested = getattr(publisher, "stop_requested", None)
    if callable(stop_requested):
        return bool(stop_requested())

    client = getattr(publisher, "client", None)
    ok = getattr(client, "ok", None)
    if callable(ok):
        return not bool(ok())

    return False


def execute_fixed_put_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool,
    object_type: str | None = None,
    keep_put_pose: bool = False,
) -> FixedPutResult:
    """Place and release, optionally keeping the released put pose.

    The caller must pass grasp_confirmed=True from a completed gripper grip result.
    Without that explicit confirmation this function refuses to publish any put target.
    When keep_put_pose=True, no Home target is published and the publisher's
    remembered target remains the put pose for the next grasp.
    """
    hand = normalize_hand(hand)
    if not bool(grasp_confirmed):
        status = f"{hand} put blocked: gripper has not confirmed a successful grasp"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    if publisher is None:
        status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    if publisher_stop_requested(publisher):
        status = f"STOPPED by B before {hand} put target publishing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    put_target_hold_sec = max(float(getattr(args, "put_target_hold_sec", 0.05)), 0.0)
    put_home_hold_sec = max(float(getattr(args, "put_home_hold_sec", 0.05)), 0.0)
    position = np.asarray(fixed_put_xyz_for_hand(hand, object_type), dtype=np.float64)
    orientation = ik_wrist_orientation_quat(args, hand=hand)
    waypoints = humanlike_put_waypoints(publisher, hand, position, orientation, args)
    count = publish_request_ik_path(
        publisher,
        hand,
        waypoints,
        args,
        final_hold_sec=put_target_hold_sec,
        terminal_slowdown=True,
    )
    if publisher_stop_requested(publisher):
        status = f"STOPPED by B during {hand} put target publishing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    print(
        "[put] fixed put target reached "
        f"hand={hand} object={normalize_object_type(object_type) if object_type else 'default'} "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m "
        f"waypoints={len(waypoints)} count={count} hold={put_target_hold_sec:.2f}s",
        flush=True,
    )

    release_status = send_gripper_signal("release", args, hand=hand)
    if "ERR " in release_status or "failed exit_code=" in release_status:
        status = f"{hand} put release failed: {release_status}"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)
    else:
        status = f"{hand} put release ok"
        print(f"[put] {status}", flush=True)

    if bool(keep_put_pose):
        keep_status = f"{hand} kept at put pose; next grasp starts here"
        print(f"[put] {keep_status}", flush=True)
        return FixedPutResult(True, f"{status}; {keep_status}")

    if publisher_stop_requested(publisher):
        stop_status = f"STOPPED by B before {hand} home target publishing"
        print(f"[put] {stop_status}", flush=True)
        return FixedPutResult(False, f"{status}; {stop_status}")

    home_status = publish_home_request_ik_target(
        publisher,
        hand,
        position_for_home(hand, args),
        args,
        final_hold_sec=put_home_hold_sec,
    )
    return FixedPutResult(True, f"{status}; {home_status}")


def publish_fixed_put_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool = False,
    object_type: str | None = None,
    keep_put_pose: bool = False,
) -> bool:
    """Boolean one-call wrapper for fixed put.

    True means the put target was reached and release succeeded. With
    keep_put_pose=True, Home is deliberately skipped; otherwise Home must also succeed.
    """
    return execute_fixed_put_after_grasp(
        publisher,
        hand,
        args,
        grasp_confirmed=grasp_confirmed,
        object_type=object_type,
        keep_put_pose=keep_put_pose,
    ).ok


def position_for_home(hand: str, args: argparse.Namespace) -> tuple[float, float, float]:
    if normalize_hand(hand) == "left":
        return args.left_home_xyz
    return args.right_home_xyz
