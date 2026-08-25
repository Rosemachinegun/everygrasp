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
    publish_request_ik_target,
)
from grasp_core.core.pose_math import ik_wrist_orientation_quat, normalize_object_type

FIXED_PUT_RIGHT_XYZ = (0.54, -0.30, 0.826)
#FIXED_PUT_RIGHT_XYZ = (0.54, -0.30, 0.776)
FIXED_PUT_LEFT_XYZ = (0.559, 0.350, 0.772)
FIXED_PUT_OBJECT_XYZ = {
    "yellow_cube": (0.50, -0.35, 0.86),
    "blue_cube": (0.35, -0.35, 0.83),
}


@dataclass(frozen=True)
class FixedPutResult:
    ok: bool
    status: str


def fixed_put_xyz_for_hand(
    hand: str,
    object_type: str | None = None,
) -> tuple[float, float, float]:
    if object_type:
        object_name = normalize_object_type(object_type)
        if object_name in FIXED_PUT_OBJECT_XYZ:
            return FIXED_PUT_OBJECT_XYZ[object_name]

    hand_name = str(hand).strip().lower()
    if hand_name == "left":
        return FIXED_PUT_LEFT_XYZ
    return FIXED_PUT_RIGHT_XYZ


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
) -> FixedPutResult:
    """Return ok=True only when the full place-release-home sequence completes.

    The caller must pass grasp_confirmed=True from a completed gripper grip result.
    Without that explicit confirmation this function refuses to publish any put target.
    """
    hand = "left" if str(hand).strip().lower() == "left" else "right"
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

    position = np.asarray(fixed_put_xyz_for_hand(hand, object_type), dtype=np.float64)
    orientation = ik_wrist_orientation_quat(args)
    count = publish_request_ik_target(
        publisher,
        hand,
        position,
        orientation,
        args,
    )
    if publisher_stop_requested(publisher):
        status = f"STOPPED by B during {hand} put target publishing"
        print(f"[put] {status}", flush=True)
        return FixedPutResult(False, status)

    print(
        "[put] fixed put target reached "
        f"hand={hand} object={normalize_object_type(object_type) if object_type else 'default'} "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m "
        f"count={count}",
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

    if publisher_stop_requested(publisher):
        stop_status = f"STOPPED by B before {hand} home target publishing"
        print(f"[put] {stop_status}", flush=True)
        return FixedPutResult(False, f"{status}; {stop_status}")

    home_status = publish_home_request_ik_target(
        publisher,
        hand,
        position_for_home(hand, args),
        args,
    )
    return FixedPutResult(True, f"{status}; {home_status}")


def publish_fixed_put_after_grasp(
    publisher: RequestIkTargetPublisher | None,
    hand: str,
    args: argparse.Namespace,
    *,
    grasp_confirmed: bool = False,
    object_type: str | None = None,
) -> bool:
    """Boolean one-call wrapper for fixed put.

    True means the put target was published, the gripper release succeeded,
    and the home request was sent. False means nothing was placed or a step failed.
    """
    return execute_fixed_put_after_grasp(
        publisher,
        hand,
        args,
        grasp_confirmed=grasp_confirmed,
        object_type=object_type,
    ).ok


def position_for_home(hand: str, args: argparse.Namespace) -> tuple[float, float, float]:
    if str(hand).strip().lower() == "left":
        return args.left_home_xyz
    return args.right_home_xyz
