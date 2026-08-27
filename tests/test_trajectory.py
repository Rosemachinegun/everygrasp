import pytest
import numpy as np

from grasp_core.communication.request_ik_publisher import terminal_sample_periods
from grasp_core.motion.trajectory import plan_pose_path, position_tangents


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def test_position_tangents_preserve_path_shape_at_endpoints() -> None:
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.2, 0.1, 0.0]),
    ]

    tangents = position_tangents(positions)

    np.testing.assert_allclose(tangents[0], positions[1] - positions[0])
    np.testing.assert_allclose(tangents[-1], positions[-1] - positions[-2])
    assert np.linalg.norm(tangents[1]) > 0.0


def test_plan_pose_path_keeps_final_sample_on_target() -> None:
    target = np.array([0.2, 0.1, 0.0])
    plan = plan_pose_path(
        np.array([0.0, 0.0, 0.0]),
        IDENTITY_QUAT,
        [
            (np.array([0.1, 0.0, 0.0]), IDENTITY_QUAT),
            (target, IDENTITY_QUAT),
        ],
        max_step_m=0.02,
        max_step_deg=5.0,
        min_steps=5,
    )

    assert len(plan.samples) == sum(plan.segment_steps)
    np.testing.assert_allclose(plan.samples[-1][0], target)


def test_terminal_sample_periods_only_changes_timing_tail() -> None:
    periods = terminal_sample_periods(
        10,
        0.01,
        enabled=True,
        tail_count=4,
        max_scale=3.0,
    )

    assert periods[:6] == pytest.approx([0.01] * 6)
    assert periods[-1] == pytest.approx(0.03)
    assert periods[-4:] == sorted(periods[-4:])


def test_terminal_sample_periods_disabled_is_uniform() -> None:
    assert terminal_sample_periods(5, 0.01, enabled=False) == pytest.approx(
        [0.01] * 5
    )
