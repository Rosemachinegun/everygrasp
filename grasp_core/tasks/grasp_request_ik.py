#!/usr/bin/env python3
"""任务层：把最新感知目标转换成 request_ik 抓取路径并触发夹爪动作。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Callable

import numpy as np

from grasp_core.core.robot_target_pose import TargetObjectPose, matrix_to_quaternion
from grasp_core.communication.gripper_signal import send_gripper_signal
from grasp_core.planning.grasp_pose import make_gripper_target_pose
from grasp_core.core.pose_math import (
    PickTemplateWaypoint,
    PoseWaypoint,
    checked_position,
    format_quat,
    format_xyz,
    home_position_for_hand,
    ik_wrist_orientation_quat,
    log_grasp_pose_plan,
    normalize_quaternion,
    pose_from_position_quaternion,
    select_ik_hand,
)
from grasp_core.config.request_ik_config import DEFAULT_GRIP_SETTLE_SEC
from grasp_core.communication.request_ik_publisher import (
    RequestIkTargetPublisher,
    save_request_ik_grasp_path_artifacts,
    publish_request_ik_path,
    publish_request_ik_target,
)
from grasp_core.planning.tool_pick_templates import (
    build_pick_template_waypoints,
    pick_template_for_target,
)
WaypointCallback = Callable[
    [RequestIkTargetPublisher, str, np.ndarray, tuple[float, float, float, float]],
    int,
]
GripConfirmedCallback = Callable[[str, str], None]


class GripFailedMinLimit(RuntimeError):
    """Raised when the gripper closes to its minimum limit without grasping."""


class GripCommandFailed(RuntimeError):
    """Raised when a gripper command fails before completing normally."""


def publish_latest_request_ik_target(
    publisher: RequestIkTargetPublisher | None,
    targets: list[TargetObjectPose],
    pick_templates: dict[str, dict[str, list[PickTemplateWaypoint]]],
    args: argparse.Namespace,
) -> str:
    if publisher is None:
        status = "request_ik_tester publisher unavailable; check ROS2 sourcing"
        print(f"[request_ik_tester] {status}", flush=True)
        return status
    if not targets:
        status = "No FlowPose target to publish: press F and wait for result"
        print(f"[request_ik_tester] {status}", flush=True)
        return status

    index = min(max(int(args.ik_target_index), 0), len(targets) - 1)
    target = targets[index]
    hand = select_ik_hand(target.base_xyz, args.ik_hand)
    grip_result: dict[str, Any] = {"confirmed": False, "hand": None, "status": ""}
    start_position = home_position_for_hand(hand, args)
    start_orientation = ik_wrist_orientation_quat(args)

    relative_pick_waypoints = pick_template_for_target(target, hand, pick_templates)
    print(
        "[request_ik_tester] grasp request "
        f"label={target.label!r} hand={hand} "
        f"tool_template={'hit' if relative_pick_waypoints is not None else 'miss'} "
        f"ik_target_stage={args.ik_target_stage}",
        flush=True,
    )
    used_pick_template = False
    template: np.ndarray | None = None
    fallback_reason: str | None = None
    grasp_path_artifacts = None
    if relative_pick_waypoints is not None:
        try:
            pick_waypoints = build_pick_template_waypoints(
                target, relative_pick_waypoints, args
            )
            print(
                "[tool_template] using YAML xyz only; YAML quaternions ignored, "
                "fixed orientation + downward tilt applied "
                f"base_quat={format_quat(start_orientation)} "
                f"tilt={args.ik_downward_tilt_deg:.2f}deg/"
                f"{args.ik_downward_tilt_axis}+y={args.ik_downward_tilt_y_deg:.2f}deg/"
                f"{args.ik_downward_tilt_frame}",
                flush=True,
            )
            grip_waypoint_index = pick_grip_waypoint_index(pick_waypoints)
            pose_waypoints = strip_gripper_states(pick_waypoints)
            grip_callbacks = make_grip_waypoint_callbacks(
                grip_waypoint_index,
                pick_waypoints,
                args,
                on_grip_confirmed=lambda grip_hand, grip_status: grip_result.update(
                    confirmed=True,
                    hand=grip_hand,
                    status=grip_status,
                ),
            )
            if grip_waypoint_index is not None:
                grip_position, _grip_orientation, grip_state = pick_waypoints[
                    grip_waypoint_index
                ]
                print(
                    "[tool_template] selected grip waypoint "
                    f"index={grip_waypoint_index} "
                    f"xyz=({grip_position[0]:.4f}, {grip_position[1]:.4f}, {grip_position[2]:.4f}) "
                    f"gripper_state={float(grip_state):.1f}",
                    flush=True,
                )
            if grip_waypoint_index is None:
                count = publish_request_ik_path(
                    publisher,
                    hand,
                    pose_waypoints,
                    args,
                    start_position_xyz=start_position,
                    start_orientation_xyzw=start_orientation,
                )
            else:
                print(
                    "[tool_template] executing pick path in explicit phases: "
                    f"approach_until_grip_index={grip_waypoint_index}, "
                    f"total_waypoints={len(pose_waypoints)}",
                    flush=True,
                )
                count = publish_request_ik_path(
                    publisher,
                    hand,
                    pose_waypoints[: grip_waypoint_index + 1],
                    args,
                    start_position_xyz=start_position,
                    start_orientation_xyzw=start_orientation,
                )
                grasp_path_artifacts = save_request_ik_grasp_path_artifacts(
                    publisher, target, hand, args
                )
                grip_position, grip_orientation = pose_waypoints[grip_waypoint_index]
                count += grip_callbacks[grip_waypoint_index](
                    publisher,
                    hand,
                    grip_position,
                    grip_orientation,
                )
                remaining_waypoints = pose_waypoints[grip_waypoint_index + 1 :]
                if remaining_waypoints:
                    count += publish_request_ik_path(
                        publisher,
                        hand,
                        remaining_waypoints,
                        args,
                    )
            position, orientation = pose_waypoints[-1]
            used_pick_template = True
            gripper_pose = pose_from_position_quaternion(position, orientation)
        except GripFailedMinLimit as exc:
            status = f"GRIP_FAILED_MIN_LIMIT hand={hand}: {exc}"
            print(f"[tool_template] {status}", flush=True)
            return status
        except GripCommandFailed as exc:
            status = f"GRIP_COMMAND_FAILED hand={hand}: {exc}"
            print(f"[tool_template] {status}", flush=True)
            return status
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"pick template failed: {exc}"
            print(
                f"[tool_template] {fallback_reason}; using computed target", flush=True
            )
    if not used_pick_template:
        try:
            gripper_pose, template, grasp_fallback_reason = make_gripper_target_pose(
                target, args
            )
            fallback_reason = fallback_reason or grasp_fallback_reason
            position = gripper_pose[:3, 3].copy()
            orientation = matrix_to_quaternion(gripper_pose)
            count = publish_request_ik_target(
                publisher,
                hand,
                position,
                orientation,
                args,
                start_position_xyz=start_position,
                start_orientation_xyzw=start_orientation,
            )
            print(
                "[computed_grasp] target reached; sending grip command "
                f"hand={hand} xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})",
                flush=True,
            )
            count += execute_grip_at_pose(
                publisher,
                hand,
                position,
                orientation,
                args,
                on_grip_confirmed=lambda grip_hand, grip_status: grip_result.update(
                    confirmed=True,
                    hand=grip_hand,
                    status=grip_status,
                ),
            )
        except GripFailedMinLimit as exc:
            status = f"GRIP_FAILED_MIN_LIMIT hand={hand}: {exc}"
            print(f"[computed_grasp] {status}", flush=True)
            return status
        except GripCommandFailed as exc:
            status = f"GRIP_COMMAND_FAILED hand={hand}: {exc}"
            print(f"[computed_grasp] {status}", flush=True)
            return status

    log_grasp_pose_plan(target, gripper_pose, template, fallback_reason)
    qx, qy, qz, qw = orientation
    topic = args.left_target_topic if hand == "left" else args.right_target_topic
    status = (
        f"Published {hand} request_ik_tester target "
        f"xyz=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})m"
        f"{' pick-template' if used_pick_template else ''}"
        f"{' fallback' if fallback_reason else ''}"
    )
    if bool(grip_result["confirmed"]):
        status += f" | grasp_confirmed=True hand={grip_result['hand']}"
    print(
        "[request_ik_tester] sent "
        f"{target.frame_id}_{args.ik_target_stage}: hand={hand} topic={topic} "
        f"frame={args.ik_frame_id} count={count} "
        f"position=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m "
        f"orientation_xyzw=({qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}) "
        f"pick_template={used_pick_template} "
        f"template_raw_pose={used_pick_template} "
        f"use_flowpose_rotation={bool(args.use_flowpose_grasp_rotation)} "
        f"tcp_offset={format_xyz(np.asarray(args.ik_grasp_tcp_offset_m, dtype=np.float64))} "
        f"tilt={args.ik_downward_tilt_deg:.2f}deg/"
        f"{args.ik_downward_tilt_axis}+y={args.ik_downward_tilt_y_deg:.2f}deg/"
        f"{args.ik_downward_tilt_frame}",
        flush=True,
    )
    artifacts = grasp_path_artifacts or save_request_ik_grasp_path_artifacts(
        publisher, target, hand, args
    )
    if artifacts is not None:
        status += f" | grasp_path_csv={artifacts.csv_path}"
        if artifacts.plot_path is not None:
            status += f" | grasp_path_plot={artifacts.plot_path}"
    return status


def strip_gripper_states(waypoints: list[PickTemplateWaypoint]) -> list[PoseWaypoint]:
    return [(position, orientation) for position, orientation, _ in waypoints]


def pick_grip_waypoint_index(waypoints: list[PickTemplateWaypoint]) -> int | None:
    if not waypoints:
        return None
    candidates = [
        index
        for index, (_position, _orientation, gripper_state) in enumerate(waypoints)
        if float(gripper_state) >= 0.5
    ]
    if not candidates:
        fallback_index = min(
            range(len(waypoints)),
            key=lambda index: float(waypoints[index][0][2]),
        )
        position = checked_position(waypoints[fallback_index][0])
        print(
            "[tool_template] WARNING no gripper_state>=0.5 in pick template; "
            "defaulting grip trigger to lowest waypoint "
            f"index={fallback_index} "
            f"xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})",
            flush=True,
        )
        return fallback_index
    return min(candidates, key=lambda index: float(waypoints[index][0][2]))


def make_grip_waypoint_callbacks(
    grip_waypoint_index: int | None,
    waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    *,
    on_grip_confirmed: GripConfirmedCallback | None = None,
) -> dict[int, WaypointCallback]:
    if grip_waypoint_index is None:
        return {}
    grip_position = checked_position(waypoints[grip_waypoint_index][0])

    def callback(
        publisher: RequestIkTargetPublisher,
        hand: str,
        position: np.ndarray,
        orientation: tuple[float, float, float, float],
    ) -> int:
        print(
            "[tool_template] gripper_state=1.0; lowest pick waypoint reached, sending grip command like L key "
            f"xyz=({grip_position[0]:.4f}, {grip_position[1]:.4f}, {grip_position[2]:.4f})",
            flush=True,
        )
        return execute_grip_at_pose(
            publisher,
            hand,
            position,
            orientation,
            args,
            on_grip_confirmed=on_grip_confirmed,
        )

    return {grip_waypoint_index: callback}


def execute_grip_at_pose(
    publisher: RequestIkTargetPublisher,
    hand: str,
    position: np.ndarray,
    orientation: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    on_grip_confirmed: GripConfirmedCallback | None = None,
) -> int:
    settle_sec = max(
        float(getattr(args, "grip_settle_sec", DEFAULT_GRIP_SETTLE_SEC)), 0.0
    )
    pre_grip_hold_sec = max(settle_sec, 0.05)
    print(
        "[grip] holding target before close "
        f"hand={hand} pre_hold={pre_grip_hold_sec:.2f}s "
        f"post_hold={settle_sec:.2f}s",
        flush=True,
    )
    count = 0
    if publisher.uses_trajectory_command(hand):
        count += publisher.client.hold_pose(
            hand,
            checked_position(position),
            normalize_quaternion(orientation),
            duration_sec=pre_grip_hold_sec,
            publish_rate_hz=publisher.publish_rate_hz,
        )
    else:
        count += publisher.hold_target(hand, position, orientation, pre_grip_hold_sec)
    gripper_status = send_gripper_signal("grip", args, hand=hand)
    if "GRASP_FAILED_MIN_LIMIT" in gripper_status:
        raise GripFailedMinLimit(gripper_status)
    if "ERR " in gripper_status or "failed exit_code=" in gripper_status:
        raise GripCommandFailed(gripper_status)
    if on_grip_confirmed is not None:
        on_grip_confirmed(hand, gripper_status)
    return count + publisher.hold_target(hand, position, orientation, settle_sec)
