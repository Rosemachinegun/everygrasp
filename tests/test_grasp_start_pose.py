from argparse import Namespace

import numpy as np

from grasp_core.tasks.grasp_request_ik import resolve_grasp_start_pose


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


class FakePublisher:
    def __init__(self, remembered=None) -> None:
        self.remembered = remembered

    def remembered_target(self, hand: str):
        return self.remembered


def start_args() -> Namespace:
    return Namespace(
        left_home_xyz=(0.25, 0.25, 0.81),
        right_home_xyz=(0.25, -0.25, 0.81),
        ik_orientation_quat=IDENTITY_QUAT,
        ik_downward_tilt_deg=0.0,
        ik_downward_tilt_left_deg=None,
        ik_downward_tilt_right_deg=None,
        ik_downward_tilt_axis="y",
        ik_downward_tilt_y_deg=0.0,
        ik_downward_tilt_y_left_deg=None,
        ik_downward_tilt_y_right_deg=None,
        ik_downward_tilt_frame="local",
    )


def test_next_grasp_starts_at_remembered_put_pose() -> None:
    put_position = np.array([0.54, -0.30, 0.826])
    position, orientation, source = resolve_grasp_start_pose(
        FakePublisher((put_position, IDENTITY_QUAT)),
        "right",
        start_args(),
    )

    np.testing.assert_allclose(position, put_position)
    assert orientation == IDENTITY_QUAT
    assert source == "last_published_target"


def test_first_grasp_falls_back_to_home() -> None:
    position, orientation, source = resolve_grasp_start_pose(
        FakePublisher(),
        "right",
        start_args(),
    )

    np.testing.assert_allclose(position, np.array([0.25, -0.25, 0.81]))
    assert orientation == IDENTITY_QUAT
    assert source == "home"
