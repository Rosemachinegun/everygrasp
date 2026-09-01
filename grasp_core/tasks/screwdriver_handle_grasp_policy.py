#!/usr/bin/env python3
"""Side-aware grasp policy for screwdriver-handle style long objects.

Coordinate conventions
----------------------
FlowPose:
    local X = physical long axis of the screwdriver handle.

Grasp-policy frame:
    local Y = physical long axis
    local Z = world +Z
    local X = Y x Z

The sign of policy +Y is selected according to the active arm:
    left  -> +Y tends toward base/world +Y
    right -> +Y tends toward base/world -Y

The resulting policy frame is always right-handed:
    X x Y = Z

Gripper TCP convention:
    local X = gripper closing axis
    local Z = gripper approach axis
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from grasp_core.core.pose_math import (
    PickTemplateWaypoint,
    PoseWaypoint,
    apply_pregrasp_offset,
    checked_position,
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_orientation_rotation,
    make_downward_tilt_rotation,
    normalize_object_type,
    validate_pose_matrix,
)
from grasp_core.core.robot_target_pose import (
    TargetObjectPose,
    matrix_to_quaternion,
)


LONG_OBJECT_KEYWORDS = (
    "screwdriver_handle",
    "pen",
)
HAND_SIDE_EPS = 1e-8

# Prevent ±180° orientation switching when the long axis is almost parallel
# to world X and therefore has almost no left/right Y component.
LONG_AXIS_SIDE_EPS = 0.05

WORLD_X_AXIS = np.asarray(
    [1.0, 0.0, 0.0],
    dtype=np.float64,
)
WORLD_Y_AXIS = np.asarray(
    [0.0, 1.0, 0.0],
    dtype=np.float64,
)
WORLD_Z_AXIS = np.asarray(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)

# request_ik TCP convention:
#
# local X = closing axis
# local Z = approach axis
GRIPPER_CLOSING_AXIS_INDEX = 0
GRIPPER_APPROACH_AXIS_INDEX = 2

# FlowPose screwdriver convention:
#
# local X = physical long axis
FLOWPOSE_LONG_AXIS_INDEX = 0


@dataclass(frozen=True)
class ScrewdriverHandleGraspPolicyResult:
    """Adjusted screwdriver-handle grasp plan."""

    pose: np.ndarray

    # Reconstructed Z-up policy frame:
    #
    # X = lateral direction
    # Y = long axis
    # Z = world up
    object_pose: np.ndarray

    # +1 left, -1 right
    side_sign: float

    # Policy X axis, used as desired gripper closing direction.
    side_axis: np.ndarray

    # Policy Y axis, physical screwdriver long axis.
    long_axis: np.ndarray

    yaw_deg: float
    tilt_y_deg: float


def is_screwdriver_handle_object(object_type: str) -> bool:
    """Return True for supported screwdriver/pen style long objects."""

    normalized_type = normalize_object_type(object_type)

    return any(
        keyword in normalized_type
        for keyword in LONG_OBJECT_KEYWORDS
    )

def make_screwdriver_handle_gripper_pose(
    target: TargetObjectPose,
    args: argparse.Namespace,
    *,
    hand: str,
) -> tuple[
    np.ndarray,
    ScrewdriverHandleGraspPolicyResult,
] | None:
    """Construct the gripper target pose for a screwdriver handle."""

    if not is_screwdriver_handle_object(target.label):
        return None

    side_sign = side_sign_for_hand(hand)
    if side_sign == 0.0:
        return None

    object_pose = screwdriver_handle_z_up_object_pose(
        target.base_pose,
        hand=hand,
    )

    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        return None

    # After screwdriver_handle_z_up_object_pose():
    #
    # object_pose[:, 0] = lateral axis
    # object_pose[:, 1] = physical long axis
    # object_pose[:, 2] = world +Z
    side_axis = object_pose[:3, 0].copy()
    long_axis = object_pose[:3, 1].copy()

    gripper_pose = np.eye(4, dtype=np.float64)
    gripper_pose[:3, 3] = object_pose[:3, 3]

    orientation, yaw_deg, tilt_y_deg = (
        screwdriver_handle_wrist_orientation(
            args,
            hand=hand,
            side_axis=side_axis,
        )
    )

    gripper_pose[:3, :3] = orientation

    gripper_pose[:3, 3] += np.asarray(
        args.ik_grasp_tcp_offset_m,
        dtype=np.float64,
    )

    if args.ik_target_stage == "pregrasp":
        gripper_pose = apply_pregrasp_offset(
            gripper_pose,
            args,
        )

    return (
        gripper_pose,
        ScrewdriverHandleGraspPolicyResult(
            pose=gripper_pose,
            object_pose=object_pose,
            side_sign=side_sign,
            side_axis=side_axis,
            long_axis=long_axis,
            yaw_deg=yaw_deg,
            tilt_y_deg=tilt_y_deg,
        ),
    )


def build_screwdriver_handle_pick_waypoints(
    target: TargetObjectPose,
    relative_waypoints: list[PickTemplateWaypoint],
    args: argparse.Namespace,
    *,
    hand: str,
) -> list[PickTemplateWaypoint] | None:
    """Build screwdriver-handle pick waypoints.

    Relative XYZ positions are interpreted in the reconstructed policy frame:

        X = lateral direction
        Y = physical long axis
        Z = world up

    Therefore:
        relative Y -> offset along the screwdriver long axis
        relative X -> lateral offset across the screwdriver
        relative Z -> vertical offset
    """

    if not is_screwdriver_handle_object(target.label):
        return None

    if not relative_waypoints:
        return []

    side_sign = side_sign_for_hand(hand)
    if side_sign == 0.0:
        return None

    object_pose = screwdriver_handle_z_up_object_pose(
        target.base_pose,
        hand=hand,
    )

    fallback_reason = validate_pose_matrix(object_pose)
    if fallback_reason is not None:
        raise ValueError(
            "invalid screwdriver_handle pose: "
            f"{fallback_reason}"
        )

    side_axis = object_pose[:3, 0].copy()

    gripper_pose = np.eye(4, dtype=np.float64)

    gripper_pose[:3, :3] = (
        screwdriver_handle_wrist_orientation(
            args,
            hand=hand,
            side_axis=side_axis,
        )[0]
    )

    orientation = matrix_to_quaternion(
        gripper_pose,
    )

    waypoints: list[PickTemplateWaypoint] = []

    for (
        relative_xyz,
        _relative_quat,
        gripper_value,
    ) in relative_waypoints:

        relative_point = np.ones(
            4,
            dtype=np.float64,
        )

        relative_point[:3] = checked_position(
            relative_xyz,
        )

        position = (
            object_pose @ relative_point
        )[:3]

        waypoints.append(
            (
                position.copy(),
                orientation,
                float(gripper_value),
            )
        )

    return waypoints


def screwdriver_handle_pose_waypoints(
    target: TargetObjectPose,
    args: argparse.Namespace,
    *,
    hand: str,
) -> tuple[
    list[PoseWaypoint],
    ScrewdriverHandleGraspPolicyResult,
] | None:
    """Return the single IK target waypoint for the screwdriver policy."""

    result = make_screwdriver_handle_gripper_pose(
        target,
        args,
        hand=hand,
    )

    if result is None:
        return None

    pose, metadata = result

    return (
        [
            (
                pose[:3, 3].copy(),
                matrix_to_quaternion(pose),
            )
        ],
        metadata,
    )


def screwdriver_handle_z_up_object_pose(
    object_pose: np.ndarray,
    *,
    hand: str,
) -> np.ndarray:
    """Build a Z-up right-handed frame while preserving FlowPose X as long axis.

    Convention:
        X = physical long axis
        Z = world +Z
        Y = Z x X

    X may be flipped by 180 degrees so that:
        left  -> +X tends toward world +Y
        right -> +X tends toward world -Y

    The frame always remains right-handed.
    """

    pose = np.asarray(object_pose, dtype=np.float64).copy()

    side_sign = side_sign_for_hand(hand)
    if side_sign == 0.0:
        raise ValueError(f"unsupported hand: {hand!r}")

    # FlowPose local X is the physical long axis.
    x_axis = horizontal_unit_vector(
        pose[:3, FLOWPOSE_LONG_AXIS_INDEX]
    )
    if x_axis is None:
        raise ValueError(
            "screwdriver long axis has no valid horizontal component"
        )

    # Choose +X / -X according to active arm side.
    if abs(float(x_axis[1])) >= LONG_AXIS_SIDE_EPS:
        if float(x_axis[1]) * side_sign < 0.0:
            x_axis = -x_axis

    z_axis = WORLD_Z_AXIS.copy()

    # Right-handed frame:
    #
    # X × Y = Z
    # therefore Y = Z × X
    y_axis = normalized(
        np.cross(z_axis, x_axis)
    )

    # Re-orthogonalize X.
    x_axis = normalized(
        np.cross(y_axis, z_axis)
    )

    pose[:3, :3] = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return pose

def screwdriver_handle_long_axis(
    object_pose: np.ndarray,
    *,
    hand: str,
) -> np.ndarray:
    """Return signed horizontal physical long axis.

    FlowPose local X is treated as the screwdriver physical long axis.

    The returned direction follows the grasp-policy convention:
        left  -> +Y side
        right -> -Y side
    """

    pose = np.asarray(
        object_pose,
        dtype=np.float64,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            f"object_pose must be 4x4, got {pose.shape}"
        )

    side_sign = side_sign_for_hand(hand)

    if side_sign == 0.0:
        raise ValueError(
            f"unsupported hand: {hand!r}"
        )

    raw_axis = pose[
        :3,
        FLOWPOSE_LONG_AXIS_INDEX,
    ]

    long_axis = horizontal_unit_vector(
        raw_axis,
    )

    if long_axis is None:
        raise ValueError(
            "screwdriver long axis has no valid "
            "horizontal component"
        )

    return choose_long_axis_sign_for_hand(
        long_axis,
        side_sign,
    )


def choose_long_axis_sign_for_hand(
    long_axis: np.ndarray,
    side_sign: float,
) -> np.ndarray:
    """Choose ±long_axis so +Y points toward the active arm side.

    left:
        desired world-Y sign > 0

    right:
        desired world-Y sign < 0

    A deadband is used when the long axis is nearly parallel to world X.
    In that region, the original FlowPose sign is retained to avoid
    unstable 180-degree flips caused by tiny Y-axis noise.
    """

    axis = normalized(
        np.asarray(
            long_axis,
            dtype=np.float64,
        )
    )

    y_component = float(axis[1])

    # If the screwdriver is almost parallel to world X, neither ±axis
    # clearly belongs to the left or right side.
    #
    # Keep FlowPose's original sign instead of flipping according to noise.
    if abs(y_component) < LONG_AXIS_SIDE_EPS:
        return axis.copy()

    if y_component * side_sign < 0.0:
        return -axis

    return axis.copy()


def screwdriver_handle_lateral_axis(
    object_pose: np.ndarray,
) -> np.ndarray:
    """Return policy local X, the lateral/closing direction.

    screwdriver_handle_z_up_object_pose() already defines:

        X = Y x Z

    so X must not be flipped independently, otherwise the frame would
    become left-handed.
    """

    pose = np.asarray(
        object_pose,
        dtype=np.float64,
    )

    return normalized(
        pose[:3, 0]
    )


def screwdriver_handle_wrist_orientation(
    args: argparse.Namespace,
    *,
    hand: str,
    side_axis: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Construct screwdriver wrist orientation.

    The gripper local X axis is aligned with the policy lateral direction,
    while the shared Y-axis downward tilt is preserved.
    """

    base_rotation = ik_orientation_rotation(
        args,
    )

    yaw_deg = screwdriver_handle_yaw_deg(
        base_rotation,
        side_axis,
        reference_yaw_deg=(
            ik_downward_tilt_deg_for_hand(
                args,
                hand,
            )
        ),
    )

    tilt_y_deg = (
        ik_downward_tilt_y_deg_for_hand(
            args,
            hand,
        )
    )

    tilt_rotation = (
        make_downward_tilt_rotation(
            yaw_deg,
            axis="z",
            y_deg=tilt_y_deg,
        )
    )

    if (
        getattr(
            args,
            "ik_downward_tilt_frame",
            "local",
        )
        == "base"
    ):
        orientation = (
            tilt_rotation @ base_rotation
        )
    else:
        orientation = (
            base_rotation @ tilt_rotation
        )

    return (
        orientation,
        yaw_deg,
        tilt_y_deg,
    )


def screwdriver_handle_yaw_deg(
    base_rotation: np.ndarray,
    side_axis: np.ndarray,
    *,
    reference_yaw_deg: float = 0.0,
) -> float:
    """Compute yaw that aligns gripper local X with the lateral axis.

    Because local X is the gripper closing axis, this chooses a closing
    direction perpendicular to the screwdriver long axis.

    Two orientations separated by 180 degrees describe the same parallel
    closing-axis line.  The orientation closest to the hand-specific
    reference yaw is selected.
    """

    desired_closing = normalized(
        side_axis,
    )

    desired_in_base_wrist = (
        np.asarray(
            base_rotation,
            dtype=np.float64,
        ).T
        @ desired_closing
    )

    desired_horizontal = (
        horizontal_unit_vector(
            desired_in_base_wrist,
        )
    )

    if desired_horizontal is None:
        desired_horizontal = (
            WORLD_X_AXIS.copy()
        )

    # local X should point toward desired_horizontal.
    yaw_rad = np.arctan2(
        desired_horizontal[1],
        desired_horizontal[0],
    )

    yaw_deg = float(
        np.rad2deg(
            yaw_rad,
        )
    )

    return closest_parallel_axis_yaw_deg(
        yaw_deg,
        reference_yaw_deg,
    )


def closest_parallel_axis_yaw_deg(
    yaw_deg: float,
    reference_yaw_deg: float,
) -> float:
    """Choose yaw or yaw+180 closest to the preferred wrist orientation."""

    candidates = (
        normalize_angle_deg(
            yaw_deg,
        ),
        normalize_angle_deg(
            yaw_deg + 180.0,
        ),
    )

    return min(
        candidates,
        key=lambda candidate: (
            angular_distance_deg(
                candidate,
                reference_yaw_deg,
            ),
            abs(candidate),
        ),
    )


def normalize_angle_deg(
    angle_deg: float,
) -> float:
    """Normalize angle to [-180, 180)."""

    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def angular_distance_deg(
    a_deg: float,
    b_deg: float,
) -> float:
    """Shortest absolute angular distance in degrees."""

    return abs(
        normalize_angle_deg(
            float(a_deg)
            - float(b_deg)
        )
    )


def closing_axis_from_orientation(
    rotation: np.ndarray,
) -> np.ndarray:
    """Return gripper local X expressed in base frame."""

    return np.asarray(
        rotation,
        dtype=np.float64,
    )[:, GRIPPER_CLOSING_AXIS_INDEX]


def approach_axis_from_orientation(
    rotation: np.ndarray,
) -> np.ndarray:
    """Return gripper local Z expressed in base frame."""

    return np.asarray(
        rotation,
        dtype=np.float64,
    )[:, GRIPPER_APPROACH_AXIS_INDEX]


def side_sign_for_hand(
    hand: str,
) -> float:
    """Return body's Y-side sign for a hand.

    left  -> +1
    right -> -1
    """

    hand_name = (
        str(hand)
        .strip()
        .lower()
    )

    if hand_name == "left":
        return 1.0

    if hand_name == "right":
        return -1.0

    return 0.0


def horizontal_unit_vector(
    vector: np.ndarray,
) -> np.ndarray | None:
    """Project a vector onto world XY and normalize it."""

    projected = np.asarray(
        vector,
        dtype=np.float64,
    ).copy()

    projected[2] = 0.0

    norm = float(
        np.linalg.norm(
            projected,
        )
    )

    if norm < HAND_SIDE_EPS:
        return None

    return projected / norm


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    """Return normalized vector."""

    array = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(
            array,
        )
    )

    if norm < HAND_SIDE_EPS:
        raise ValueError(
            "cannot normalize near-zero vector"
        )

    return array / norm