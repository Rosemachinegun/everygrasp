#!/usr/bin/env python3
"""运动层：唯一负责笛卡尔 waypoint 插值和轨迹步长计算的模块。

这里不发布 ROS、不控制夹爪、不读取配置文件，也不做抓取任务决策。
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse

import numpy as np

from grasp_core.core.pose_math import (
    PoseWaypoint,
    checked_position,
    normalize_quaternion,
    path_segment_steps,
    slerp_quaternion,
)
from grasp_core.config.request_ik_config import (
    DEFAULT_TARGET_PUBLISH_RATE_HZ,
    DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
    DEFAULT_TARGET_TRAJECTORY_SPEED_MPS,
    DEFAULT_TARGET_TRAJECTORY_STEP_DEG,
    DEFAULT_TARGET_TRAJECTORY_STEP_M,
)


@dataclass(frozen=True)
class TrajectoryPlan:
    """Interpolated Cartesian trajectory plus metadata for logging/plotting."""

    raw_waypoints: list[PoseWaypoint]
    samples: list[PoseWaypoint]
    segment_steps: list[int]


def plan_pose_path(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    waypoints: list[PoseWaypoint],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> TrajectoryPlan:
    """Interpolate a Cartesian path without stopping at each intermediate point."""

    current_position = checked_position(start_position).copy()
    current_orientation = normalize_quaternion(start_orientation)
    raw_waypoints: list[PoseWaypoint] = [(current_position.copy(), current_orientation)]
    segment_steps: list[int] = []

    for end_position_raw, end_orientation_raw in waypoints:
        end_position = checked_position(end_position_raw)
        end_orientation = normalize_quaternion(end_orientation_raw)
        steps = path_segment_steps(
            current_position,
            current_orientation,
            end_position,
            end_orientation,
            max_step_m=max_step_m,
            max_step_deg=max_step_deg,
            min_steps=min_steps,
        )
        raw_waypoints.append((end_position.copy(), end_orientation))
        segment_steps.append(steps)

        current_position = end_position
        current_orientation = end_orientation

    segment_steps = refine_segment_steps_for_limits(
        raw_waypoints,
        segment_steps,
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
    )
    samples = interpolate_pose_samples(raw_waypoints, segment_steps)
    return TrajectoryPlan(
        raw_waypoints=raw_waypoints,
        samples=samples,
        segment_steps=segment_steps,
    )


def interpolate_pose_samples(
    raw_waypoints: list[PoseWaypoint],
    segment_steps: list[int],
) -> list[PoseWaypoint]:
    if len(raw_waypoints) < 2:
        return []

    positions = [checked_position(position).copy() for position, _ in raw_waypoints]
    orientations = [
        normalize_quaternion(orientation) for _, orientation in raw_waypoints
    ]
    tangents = position_tangents(positions, stop_at_endpoints=True)
    samples: list[PoseWaypoint] = []

    for segment_index, steps in enumerate(segment_steps):
        start_position = positions[segment_index]
        end_position = positions[segment_index + 1]
        start_tangent = tangents[segment_index]
        end_tangent = tangents[segment_index + 1]
        start_orientation = orientations[segment_index]
        end_orientation = orientations[segment_index + 1]
        for step in range(1, int(steps) + 1):
            alpha = float(step) / float(steps)
            position = cubic_hermite_position(
                start_position,
                end_position,
                start_tangent,
                end_tangent,
                alpha,
            )
            orientation = slerp_quaternion(
                start_orientation,
                end_orientation,
                alpha,
            )
            samples.append((checked_position(position).copy(), orientation))
    return samples


def position_tangents(
    positions: list[np.ndarray],
    *,
    stop_at_endpoints: bool = False,
) -> list[np.ndarray]:
    """Return clamped waypoint tangents so the target glides through waypoints."""

    count = len(positions)
    if count == 0:
        return []
    if count == 1:
        return [np.zeros(3, dtype=np.float64)]

    tangents: list[np.ndarray] = []
    for index, position in enumerate(positions):
        if stop_at_endpoints and index in {0, count - 1}:
            tangent = np.zeros(3, dtype=np.float64)
            max_length = 0.0
        elif index == 0:
            tangent = positions[1] - position
            max_length = float(np.linalg.norm(tangent))
        elif index == count - 1:
            tangent = position - positions[index - 1]
            max_length = float(np.linalg.norm(tangent))
        else:
            previous_delta = position - positions[index - 1]
            next_delta = positions[index + 1] - position
            tangent = 0.5 * (previous_delta + next_delta)
            max_length = 0.5 * min(
                float(np.linalg.norm(previous_delta)),
                float(np.linalg.norm(next_delta)),
            )
        tangents.append(clamp_vector_length(tangent, max_length))
    return tangents


def refine_segment_steps_for_limits(
    raw_waypoints: list[PoseWaypoint],
    segment_steps: list[int],
    *,
    max_step_m: float,
    max_step_deg: float,
    max_iterations: int = 6,
) -> list[int]:
    """Increase segment sample counts until the actual curve respects limits."""

    if len(raw_waypoints) < 2:
        return []

    positions = [checked_position(position).copy() for position, _ in raw_waypoints]
    orientations = [
        normalize_quaternion(orientation) for _, orientation in raw_waypoints
    ]
    tangents = position_tangents(positions, stop_at_endpoints=True)
    refined_steps = [max(int(steps), 1) for steps in segment_steps]
    linear_limit = max(float(max_step_m), 1e-4)
    angular_limit = np.deg2rad(max(float(max_step_deg), 0.1))

    for _ in range(max_iterations):
        changed = False
        for segment_index, steps in enumerate(refined_steps):
            max_linear_step = 0.0
            max_angular_step = 0.0
            previous_position = positions[segment_index]
            previous_orientation = orientations[segment_index]
            for step in range(1, steps + 1):
                alpha = float(step) / float(steps)
                position = cubic_hermite_position(
                    positions[segment_index],
                    positions[segment_index + 1],
                    tangents[segment_index],
                    tangents[segment_index + 1],
                    alpha,
                )
                orientation = slerp_quaternion(
                    orientations[segment_index],
                    orientations[segment_index + 1],
                    alpha,
                )
                max_linear_step = max(
                    max_linear_step,
                    float(np.linalg.norm(position - previous_position)),
                )
                max_angular_step = max(
                    max_angular_step,
                    quaternion_step_rad(previous_orientation, orientation),
                )
                previous_position = position
                previous_orientation = orientation

            ratio = max(
                max_linear_step / linear_limit,
                max_angular_step / angular_limit,
            )
            if ratio > 1.0 + 1e-6:
                refined_steps[segment_index] = max(
                    steps + 1,
                    int(np.ceil(float(steps) * ratio * 1.05)),
                )
                changed = True
        if not changed:
            break

    return refined_steps


def quaternion_step_rad(
    start_xyzw: tuple[float, float, float, float],
    end_xyzw: tuple[float, float, float, float],
) -> float:
    q0 = np.asarray(normalize_quaternion(start_xyzw), dtype=np.float64)
    q1 = np.asarray(normalize_quaternion(end_xyzw), dtype=np.float64)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def clamp_vector_length(vector: np.ndarray, max_length: float) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-9 or max_length <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if length <= max_length:
        return np.asarray(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) * (float(max_length) / length)


def cubic_hermite_position(
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    alpha: float,
) -> np.ndarray:
    t = float(np.clip(alpha, 0.0, 1.0))
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * start_position
        + h10 * start_tangent
        + h01 * end_position
        + h11 * end_tangent
    )


def plan_pose_target(
    start_position: np.ndarray,
    start_orientation: tuple[float, float, float, float],
    end_position: np.ndarray,
    end_orientation: tuple[float, float, float, float],
    *,
    max_step_m: float,
    max_step_deg: float,
    min_steps: int = 1,
) -> TrajectoryPlan:
    """Interpolate a single target as a one-segment path."""

    return plan_pose_path(
        start_position,
        start_orientation,
        [(checked_position(end_position), normalize_quaternion(end_orientation))],
        max_step_m=max_step_m,
        max_step_deg=max_step_deg,
        min_steps=min_steps,
    )


def effective_trajectory_step_limits(args: argparse.Namespace) -> tuple[float, float]:
    """Combine configured geometric step limits with speed/rate limits."""

    publish_rate_hz = max(
        float(getattr(args, "target_publish_rate_hz", DEFAULT_TARGET_PUBLISH_RATE_HZ)),
        0.1,
    )
    configured_step_m = max(
        float(
            getattr(args, "target_trajectory_step_m", DEFAULT_TARGET_TRAJECTORY_STEP_M)
        ),
        1e-4,
    )
    configured_step_deg = max(
        float(
            getattr(
                args, "target_trajectory_step_deg", DEFAULT_TARGET_TRAJECTORY_STEP_DEG
            )
        ),
        0.1,
    )
    speed_mps = max(
        float(
            getattr(
                args, "target_trajectory_speed_mps", DEFAULT_TARGET_TRAJECTORY_SPEED_MPS
            )
        ),
        1e-4,
    )
    angular_speed_dps = max(
        float(
            getattr(
                args,
                "target_trajectory_angular_speed_dps",
                DEFAULT_TARGET_TRAJECTORY_ANGULAR_SPEED_DPS,
            )
        ),
        0.1,
    )
    speed_step_m = speed_mps / publish_rate_hz
    angular_speed_step_deg = angular_speed_dps / publish_rate_hz
    return min(configured_step_m, speed_step_m), min(
        configured_step_deg, angular_speed_step_deg
    )
