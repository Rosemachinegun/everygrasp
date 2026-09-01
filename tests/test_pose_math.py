from argparse import Namespace

import numpy as np

from grasp_core.core.pose_math import (
    ik_downward_tilt_deg_for_hand,
    ik_downward_tilt_y_deg_for_hand,
    ik_wrist_orientation_quat,
    quaternion_to_rotation_matrix,
)


def test_wrist_orientation_is_mirrored_between_hands() -> None:
    args = Namespace(
        ik_orientation_quat=(0.0, 0.0, 0.0, 1.0),
        ik_downward_tilt_deg=0.0,
        ik_downward_tilt_left_deg=-45.0,
        ik_downward_tilt_right_deg=45.0,
        ik_downward_tilt_axis="z",
        ik_downward_tilt_y_deg=0.0,
        ik_downward_tilt_y_left_deg=45.0,
        ik_downward_tilt_y_right_deg=45.0,
        ik_downward_tilt_frame="local",
    )

    right_rotation = quaternion_to_rotation_matrix(
        ik_wrist_orientation_quat(args, hand="right")
    )
    left_rotation = quaternion_to_rotation_matrix(
        ik_wrist_orientation_quat(args, hand="left")
    )
    mirror_y = np.diag([1.0, -1.0, 1.0])

    np.testing.assert_allclose(left_rotation, mirror_y @ right_rotation @ mirror_y)


def test_global_tilt_values_remain_fallbacks() -> None:
    args = Namespace(
        ik_downward_tilt_deg=12.0,
        ik_downward_tilt_left_deg=None,
        ik_downward_tilt_right_deg=None,
        ik_downward_tilt_y_deg=34.0,
        ik_downward_tilt_y_left_deg=None,
        ik_downward_tilt_y_right_deg=None,
    )

    assert ik_downward_tilt_deg_for_hand(args, "left") == 12.0
    assert ik_downward_tilt_deg_for_hand(args, "right") == 12.0
    assert ik_downward_tilt_y_deg_for_hand(args, "left") == 34.0
    assert ik_downward_tilt_y_deg_for_hand(args, "right") == 34.0
