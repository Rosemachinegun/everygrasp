#!/usr/bin/env python3
"""Side-first grasp policy for dual-arm request_ik targets.

Left arm stays on base_link Y+ and approaches inward. Right arm mirrors this on
base_link Y-.  Wrist orientation stays close to the configured natural pose; a
FlowPose/reference orientation is only accepted when it does not require a large
flip from that natural pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_core.core.pose_math import (
    PickTemplateWaypoint,
    PoseWaypoint,
    ik_wrist_orientation_quat,
    normalize_quaternion,
    pose_from_position_quaternion,
    quaternion_angle_rad,
    quaternion_to_rotation_matrix,
    rotation_matrix_from_zyx_euler_deg,
)
from grasp_core.core.robot_target_pose import matrix_to_quaternion


@dataclass(frozen=True)
class SideApproachPolicyResult:
    """Adjusted waypoints plus short metadata for logging."""

    waypoints: list[PickTemplateWaypoint]
    inserted_side_waypoint: bool
    side_sign: float
    side_y: float
    min_abs_y: float
    orientation_source: str


def side_sign_for_hand(hand: str) -> float:
    if hand == "left":
        return 1.0
    if hand == "right":
        return -1.0
    raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")


def policy_enabled(args) -> bool:
    return bool(getattr(args, "use_side_approach_grasp_policy", False))


def natural_wrist_orientation_quat(
    hand: str,
    args,
    *,
    reference_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return a safe wrist orientation close to the configured natural pose."""

    natural_quat = ik_wrist_orientation_quat(args)

    if reference_orientation_xyzw is not None:
        reference_quat = normalize_quaternion(reference_orientation_xyzw)
        max_deviation_deg = safe_wrist_max_deviation_deg(args)
        reference_angle_deg = np.rad2deg(
            quaternion_angle_rad(natural_quat, reference_quat)
        )
        if reference_angle_deg <= max_deviation_deg:
            return reference_quat

    outward_bias_deg = side_wrist_outward_bias_deg(args)
    if abs(outward_bias_deg) < 1e-8:
        return natural_quat

    biased_pose = np.eye(4, dtype=np.float64)
    biased_pose[:3, :3] = quaternion_to_rotation_matrix(natural_quat)
    biased_pose[:3, :3] = (
        rotation_matrix_from_zyx_euler_deg(
            yaw_deg=side_sign_for_hand(hand) * outward_bias_deg
        )
        @ biased_pose[:3, :3]
    )
    biased_quat = normalize_quaternion(matrix_to_quaternion(biased_pose))
    if (
        np.rad2deg(quaternion_angle_rad(natural_quat, biased_quat))
        <= safe_wrist_max_deviation_deg(args)
    ):
        return biased_quat
    return natural_quat


def wrist_orientation_source(
    args,
    *,
    reference_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> str:
    if reference_orientation_xyzw is None:
        return "natural+bias"
    natural_quat = ik_wrist_orientation_quat(args)
    reference_quat = normalize_quaternion(reference_orientation_xyzw)
    angle_deg = np.rad2deg(quaternion_angle_rad(natural_quat, reference_quat))
    if angle_deg <= safe_wrist_max_deviation_deg(args):
        return f"flowpose/reference({angle_deg:.1f}deg)"
    return f"natural+bias(reference_rejected={angle_deg:.1f}deg)"


def apply_side_approach_to_pick_waypoints(
    waypoints: list[PickTemplateWaypoint],
    *,
    hand: str,
    args,
    reference_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> SideApproachPolicyResult | None:
    if not policy_enabled(args) or not waypoints:
        return None

    side_sign = side_sign_for_hand(hand)
    min_abs_y = min_abs_centerline_clearance(args)
    approach_offset = side_approach_offset(args)
    side_orientation = natural_wrist_orientation_quat(
        hand,
        args,
        reference_orientation_xyzw=reference_orientation_xyzw,
    )
    orientation_source = wrist_orientation_source(
        args,
        reference_orientation_xyzw=reference_orientation_xyzw,
    )

    adjusted: list[PickTemplateWaypoint] = []
    for position, _orientation, gripper_state in waypoints:
        adjusted.append(
            (
                enforce_same_side(position, side_sign, min_abs_y),
                side_orientation,
                float(gripper_state),
            )
        )

    grip_index = grip_waypoint_index(adjusted)
    grip_position = adjusted[grip_index][0]
    side_y = outside_y(grip_position[1], side_sign, min_abs_y, approach_offset)

    first_position = adjusted[0][0]
    side_position = first_position.copy()
    side_position[1] = side_y
    side_position[2] = max(
        float(first_position[2]),
        float(grip_position[2]) + side_approach_lift(args),
    )

    inserted_side_waypoint = not positions_close(side_position, first_position)
    if inserted_side_waypoint:
        adjusted.insert(0, (side_position, side_orientation, 0.0))

    return SideApproachPolicyResult(
        waypoints=adjusted,
        inserted_side_waypoint=inserted_side_waypoint,
        side_sign=side_sign,
        side_y=float(side_y),
        min_abs_y=float(min_abs_y),
        orientation_source=orientation_source,
    )


def side_approach_pose_waypoints(
    *,
    position: np.ndarray,
    hand: str,
    args,
    reference_orientation_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[list[PoseWaypoint], SideApproachPolicyResult] | None:
    if not policy_enabled(args):
        return None

    side_sign = side_sign_for_hand(hand)
    min_abs_y = min_abs_centerline_clearance(args)
    approach_offset = side_approach_offset(args)
    orientation = natural_wrist_orientation_quat(
        hand,
        args,
        reference_orientation_xyzw=reference_orientation_xyzw,
    )
    orientation_source = wrist_orientation_source(
        args,
        reference_orientation_xyzw=reference_orientation_xyzw,
    )

    final_position = enforce_same_side(position, side_sign, min_abs_y)
    side_position = final_position.copy()
    side_position[1] = outside_y(
        final_position[1],
        side_sign,
        min_abs_y,
        approach_offset,
    )
    side_position[2] = max(
        float(side_position[2]),
        float(final_position[2]) + side_approach_lift(args),
    )

    pick_waypoints: list[PickTemplateWaypoint] = [
        (side_position, orientation, 0.0),
        (final_position, orientation, 0.0),
    ]
    return (
        [(waypoint[0], waypoint[1]) for waypoint in pick_waypoints],
        SideApproachPolicyResult(
            waypoints=pick_waypoints,
            inserted_side_waypoint=True,
            side_sign=side_sign,
            side_y=float(side_position[1]),
            min_abs_y=float(min_abs_y),
            orientation_source=orientation_source,
        ),
    )


def apply_side_approach_to_pose(
    pose: np.ndarray,
    *,
    hand: str,
    args,
) -> tuple[np.ndarray, SideApproachPolicyResult] | None:
    if not policy_enabled(args):
        return None

    position = np.asarray(pose, dtype=np.float64)[:3, 3]
    result = side_approach_pose_waypoints(
        position=position,
        hand=hand,
        args=args,
        reference_orientation_xyzw=matrix_to_quaternion(pose),
    )
    if result is None:
        return None

    waypoints, metadata = result
    final_position, final_orientation = waypoints[-1]
    adjusted_pose = pose_from_position_quaternion(final_position, final_orientation)
    return adjusted_pose, metadata


def grip_waypoint_index(waypoints: list[PickTemplateWaypoint]) -> int:
    grip_candidates = [
        index
        for index, (_position, _orientation, gripper_state) in enumerate(waypoints)
        if float(gripper_state) >= 0.5
    ]
    candidates = grip_candidates or list(range(len(waypoints)))
    return min(candidates, key=lambda index: float(waypoints[index][0][2]))


def enforce_same_side(
    position_xyz: np.ndarray,
    side_sign: float,
    min_abs_y: float,
) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=np.float64).copy()
    signed_y = float(position[1]) * float(side_sign)
    if signed_y < float(min_abs_y):
        position[1] = float(side_sign) * float(min_abs_y)
    return position


def outside_y(
    y_value: float,
    side_sign: float,
    min_abs_y: float,
    approach_offset: float,
) -> float:
    signed_y = max(float(y_value) * float(side_sign), float(min_abs_y))
    return float(side_sign) * (signed_y + max(float(approach_offset), 0.0))


def min_abs_centerline_clearance(args) -> float:
    return max(float(getattr(args, "side_approach_min_abs_y_m", 0.035)), 0.0)


def side_approach_offset(args) -> float:
    return max(float(getattr(args, "side_approach_offset_y_m", 0.12)), 0.0)


def side_approach_lift(args) -> float:
    return max(float(getattr(args, "side_approach_lift_m", 0.06)), 0.0)


def side_wrist_outward_bias_deg(args) -> float:
    return max(float(getattr(args, "side_approach_wrist_outward_bias_deg", 8.0)), 0.0)


def safe_wrist_max_deviation_deg(args) -> float:
    return max(float(getattr(args, "side_approach_max_wrist_deviation_deg", 25.0)), 0.0)


def positions_close(a: np.ndarray, b: np.ndarray) -> bool:
    distance = np.linalg.norm(
        np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    )
    return bool(float(distance) < 1e-6)


def pose_with_natural_wrist(
    position: np.ndarray,
    *,
    hand: str,
    args,
) -> np.ndarray:
    return pose_from_position_quaternion(
        enforce_same_side(
            position,
            side_sign_for_hand(hand),
            min_abs_centerline_clearance(args),
        ),
        natural_wrist_orientation_quat(hand, args),
    )


def rotation_from_natural_wrist(hand: str, args) -> np.ndarray:
    return quaternion_to_rotation_matrix(natural_wrist_orientation_quat(hand, args))
