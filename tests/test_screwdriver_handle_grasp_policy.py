from argparse import Namespace

import numpy as np

from grasp_core.config.request_ik_config import DEFAULT_TOOL_TEMPLATE_PATH, load_tool_yaml
from grasp_core.core.pose_math import quaternion_to_rotation_matrix
from grasp_core.core.robot_target_pose import TargetObjectPose
from grasp_core.planning.grasp_pose import make_gripper_target_pose
from grasp_core.planning.tool_pick_templates import build_pick_template_waypoints
from grasp_core.tasks.screwdriver_handle_grasp_policy import (
    LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD,
    approach_axis_from_orientation,
    build_screwdriver_handle_pick_waypoints,
    closing_axis_from_orientation,
    gripper_closing_axis_index_for_object,
    gripper_closing_axis_index_from_name,
    make_screwdriver_handle_gripper_pose,
    screwdriver_handle_flowpose_long_axis_index,
    screwdriver_handle_long_axis,
    screwdriver_handle_long_axis_index,
    screwdriver_handle_z_up_object_pose,
)


def test_screwdriver_handle_keeps_y_tilt_and_closing_axis_is_lateral_left() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 1.0, 0.4]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(ik_downward_tilt_y_left_deg=45.0),
        hand="left",
    )

    assert result is not None
    gripper_pose, metadata = result
    object_pose = metadata.object_pose
    np.testing.assert_allclose(object_pose[:3, 2], [0.0, 0.0, 1.0])
    assert metadata.tilt_y_deg == 45.0
    assert abs(float(closing_axis_from_orientation(gripper_pose[:3, :3]) @ metadata.long_axis)) < 1e-8
    assert horizontal_norm(closing_axis_from_orientation(gripper_pose[:3, :3])) > 0.70
    assert float(approach_axis_from_orientation(gripper_pose[:3, :3]) @ object_pose[:3, 2]) > 0.70


def test_screwdriver_handle_pose_selects_right_halfspace_side() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 1.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(ik_downward_tilt_y_right_deg=45.0),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    object_pose = metadata.object_pose
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.tilt_y_deg == 45.0
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8
    assert horizontal_norm(closing_axis) > 0.70


def test_screwdriver_handle_right_hand_avoids_equivalent_180deg_wrist_flip() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    _gripper_pose, metadata = result
    assert abs(metadata.yaw_deg) <= 90.0


def test_screwdriver_handle_right_hand_aligns_local_y_closing_axis() -> None:
    target = make_screwdriver_target(long_axis=np.array([-1.0, 1.0, 0.0]))

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8
    assert abs(float(abs(closing_axis @ metadata.side_axis) - 1.0)) < 1e-8


def test_screwdriver_handle_template_waypoints_keep_y_tilt_ignore_yaml_quaternion() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))
    relative_waypoints = [
        (
            np.array([0.0, 0.0, 0.2]),
            (0.171010, 0.171010, -0.03015, 0.96984),
            0.0,
        ),
        (
            np.array([0.0, 0.0, 0.04]),
            (0.171010, 0.171010, -0.03015, 0.96984),
            1.0,
        ),
    ]

    waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_y_left_deg=45.0),
        hand="left",
    )

    assert waypoints is not None
    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    long_axis = screwdriver_handle_long_axis(target.base_pose, target.size)
    for position, orientation, _gripper_value in waypoints:
        rotation = quaternion_to_rotation_matrix(orientation)
        assert float(approach_axis_from_orientation(rotation) @ object_pose[:3, 2]) > 0.70
        assert (
            abs(float(closing_axis_from_orientation(rotation) @ long_axis))
            < 1e-8
        )
        assert position[2] > target.base_pose[2, 3]


def test_screwdriver_handle_waypoint_y_offset_is_hand_independent() -> None:
    target = make_screwdriver_target(long_axis=np.array([0.2, 1.0, 0.0]))
    relative_waypoints = [
        (
            np.array([0.0, -0.05, 0.03]),
            (0.0, 0.0, 0.0, 1.0),
            1.0,
        ),
    ]

    left_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_left_deg=-45.0),
        hand="left",
    )
    right_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_right_deg=45.0),
        hand="right",
    )

    assert left_waypoints is not None
    assert right_waypoints is not None
    np.testing.assert_allclose(
        left_waypoints[0][0],
        right_waypoints[0][0],
    )

    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    expected_position = (
        object_pose
        @ np.array([0.0, -0.05, 0.03, 1.0], dtype=np.float64)
    )[:3]
    np.testing.assert_allclose(
        left_waypoints[0][0],
        expected_position,
    )


def test_pen_prompt_shares_position_and_closing_axis_policy() -> None:
    screwdriver_target = make_screwdriver_target(
        long_axis=np.array([0.2, 1.0, 0.0]),
        label="yellow_screwdriver_handle",
    )
    pen_target = make_screwdriver_target(
        long_axis=np.array([0.2, 1.0, 0.0]),
        label="pen",
    )

    screwdriver_result = make_screwdriver_handle_gripper_pose(
        screwdriver_target,
        args(ik_downward_tilt_y_right_deg=45.0),
        hand="right",
    )
    pen_result = make_screwdriver_handle_gripper_pose(
        pen_target,
        args(ik_downward_tilt_y_right_deg=45.0),
        hand="right",
    )

    assert screwdriver_result is not None
    assert pen_result is not None
    screwdriver_pose, screwdriver_metadata = screwdriver_result
    pen_pose, pen_metadata = pen_result
    assert LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["screwdriver_handle"] == "y"
    assert LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["pen"] == "y"
    assert gripper_closing_axis_index_from_name("x") == 0
    assert gripper_closing_axis_index_from_name("y") == 1
    assert gripper_closing_axis_index_for_object("yellow_screwdriver_handle") == 1
    assert gripper_closing_axis_index_for_object("pen") == 1
    assert screwdriver_metadata.gripper_closing_axis_index == 1
    assert pen_metadata.gripper_closing_axis_index == 1
    np.testing.assert_allclose(pen_metadata.object_pose, screwdriver_metadata.object_pose)
    np.testing.assert_allclose(pen_metadata.side_axis, screwdriver_metadata.side_axis)
    np.testing.assert_allclose(pen_metadata.long_axis, screwdriver_metadata.long_axis)

    screwdriver_closing_axis = closing_axis_from_orientation(
        screwdriver_pose[:3, :3],
        screwdriver_metadata.gripper_closing_axis_index,
    )
    pen_closing_axis = closing_axis_from_orientation(
        pen_pose[:3, :3],
        pen_metadata.gripper_closing_axis_index,
    )
    assert abs(float(screwdriver_closing_axis @ screwdriver_metadata.long_axis)) < 1e-8
    assert abs(float(abs(screwdriver_closing_axis @ screwdriver_metadata.side_axis) - 1.0)) < 1e-8
    assert abs(float(pen_closing_axis @ pen_metadata.long_axis)) < 1e-8
    assert abs(float(abs(pen_closing_axis @ pen_metadata.side_axis) - 1.0)) < 1e-8


def test_pen_template_waypoints_match_yellow_screwdriver_handle() -> None:
    raw = load_tool_yaml(DEFAULT_TOOL_TEMPLATE_PATH)
    templates = raw["templates"]
    pen_entries = {entry["arm"]: entry for entry in templates["pen"]["pick"]}
    screwdriver_entries = {
        entry["arm"]: entry
        for entry in templates["yellow_screwdriver_handle"]["pick"]
    }

    for hand in ("left", "right"):
        pen_relative = np.asarray(pen_entries[hand]["pose_relative"], dtype=np.float64)
        screwdriver_relative = np.asarray(
            screwdriver_entries[hand]["pose_relative"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(pen_relative[:, 1:3], screwdriver_relative[:, 1:3])
        assert pen_entries[hand]["gripper_state"] == screwdriver_entries[hand]["gripper_state"]
        assert pen_entries[hand]["time"] == screwdriver_entries[hand]["time"]


def test_long_object_closing_axis_config_can_select_x_or_y() -> None:
    assert gripper_closing_axis_index_from_name("x") == 0
    assert gripper_closing_axis_index_from_name("y") == 1

    original_pen_axis = LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["pen"]
    try:
        LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["pen"] = "x"
        assert gripper_closing_axis_index_for_object("pen") == 0

        LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["pen"] = "y"
        assert gripper_closing_axis_index_for_object("pen") == 1
    finally:
        LONG_OBJECT_GRIPPER_CLOSING_AXIS_BY_KEYWORD["pen"] = original_pen_axis


def test_pen_template_waypoint_y_offset_is_hand_independent() -> None:
    target = make_screwdriver_target(
        long_axis=np.array([-0.4, 1.0, 0.0]),
        label="pen",
    )
    relative_waypoints = [
        (
            np.array([0.0, -0.05, 0.03]),
            (0.0, 0.0, 0.0, 1.0),
            1.0,
        ),
    ]

    left_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_left_deg=-45.0),
        hand="left",
    )
    right_waypoints = build_screwdriver_handle_pick_waypoints(
        target,
        relative_waypoints,
        args(ik_downward_tilt_right_deg=45.0),
        hand="right",
    )

    assert left_waypoints is not None
    assert right_waypoints is not None
    np.testing.assert_allclose(
        left_waypoints[0][0],
        right_waypoints[0][0],
    )


def test_screwdriver_handle_integrates_with_global_y_tilt_but_custom_z_yaw() -> None:
    target = make_screwdriver_target(long_axis=np.array([1.0, 0.0, 0.0]))

    gripper_pose, _template, fallback_reason = make_gripper_target_pose(
        target,
        args(
            ik_downward_tilt_left_deg=-45.0,
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_axis="z",
            ik_downward_tilt_y_left_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
        ),
        hand="left",
    )

    object_pose = screwdriver_handle_z_up_object_pose(target.base_pose, target.size)
    long_axis = screwdriver_handle_long_axis(target.base_pose, target.size)
    assert fallback_reason == "screwdriver_handle_z_yaw_policy"
    assert (
        float(approach_axis_from_orientation(gripper_pose[:3, :3]) @ object_pose[:3, 2])
        > 0.70
    )
    assert (
        abs(
            float(closing_axis_from_orientation(gripper_pose[:3, :3]) @ long_axis)
        )
        < 1e-8
    )


def test_screwdriver_handle_uses_z_up_local_y_as_long_axis() -> None:
    pose = np.array(
        [
            [0.9838460166918025, -0.1788843948818775, 0.006883945628626315, 0.1006392166018486],
            [-0.16174550929752052, -0.9047477087225659, -0.39404311157962474, -0.11112722009420395],
            [0.07671639760676369, 0.38656429844077866, -0.9190661768932799, 0.6505053639411926],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    size = np.array([0.0444459393620491, 0.020002959296107292, 0.10555826872587204])

    assert screwdriver_handle_long_axis_index(size) == 1
    long_axis = screwdriver_handle_long_axis(pose, size)

    z_up_pose = screwdriver_handle_z_up_object_pose(pose, size)
    expected = z_up_pose[:3, 1]
    np.testing.assert_allclose(long_axis, expected)


def test_screwdriver_handle_real_capture_closing_x_is_not_long_axis() -> None:
    pose = np.array(
        [
            [-0.3830627202987671, -0.9236584305763245, 0.010864862240850925, 0.05669158324599266],
            [-0.7809002995491028, 0.31752994656562805, -0.5379307270050049, -0.10034602135419846],
            [0.49341437220573425, -0.2145455926656723, -0.8429189920425415, 0.6472218632698059],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    size = np.array([0.0379534587264061, 0.02592730149626732, 0.09254536032676697])
    target = TargetObjectPose(
        label="red_screwdriver_handle",
        frame_id="red_screwdriver_handle_1",
        camera_pose=np.eye(4),
        base_pose=pose,
        size=size,
    )

    result = make_screwdriver_handle_gripper_pose(
        target,
        args(
            ik_downward_tilt_right_deg=45.0,
            ik_downward_tilt_y_right_deg=45.0,
            ik_downward_tilt_axis="z",
        ),
        hand="right",
    )

    assert result is not None
    gripper_pose, metadata = result
    closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
    assert metadata.raw_long_axis_index == 2
    assert metadata.policy_long_axis_index == 1
    assert abs(float(closing_axis @ metadata.long_axis)) < 1e-8


def test_screwdriver_handle_captures_close_perpendicular_to_dynamic_flowpose_long_axis() -> None:
    capture_objects = (
        (
            np.array(
                [
                    [0.1452111005783081, -0.9891809821128845, -0.020848996937274933, 0.20597906410694122],
                    [-0.8634114265441895, -0.11640176177024841, -0.4908883273601532, -0.14546464383602142],
                    [0.48315054178237915, 0.08928371220827103, -0.8709729909896851, 0.6647924780845642],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.10, 0.02, 0.03], dtype=np.float64),
        ),
        (
            np.array(
                [
                    [-0.40771448612213135, -0.9130849242210388, -0.006684210151433945, 0.17271284759044647],
                    [-0.8099409937858582, 0.36501896381378174, -0.4590824544429779, -0.027604399248957634],
                    [0.421621173620224, -0.1817607879638672, -0.888368546962738, 0.6149181723594666],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.03, 0.02, 0.10], dtype=np.float64),
        ),
    )

    for index, (pose, size) in enumerate(capture_objects):
        target = TargetObjectPose(
            label="yellow_screwdriver_handle",
            frame_id=f"yellow_screwdriver_handle_{index}",
            camera_pose=pose,
            base_pose=pose,
            size=size,
        )

        result = make_screwdriver_handle_gripper_pose(
            target,
            args(
                ik_downward_tilt_right_deg=45.0,
                ik_downward_tilt_y_right_deg=45.0,
                ik_downward_tilt_axis="z",
            ),
            hand="right",
        )

        assert result is not None
        gripper_pose, metadata = result
        raw_long_axis_index = screwdriver_handle_flowpose_long_axis_index(size)
        assert metadata.raw_long_axis_index == raw_long_axis_index
        assert metadata.policy_long_axis_index == 1
        raw_long_axis = pose[:3, raw_long_axis_index].copy()
        raw_long_axis[2] = 0.0
        raw_long_axis /= np.linalg.norm(raw_long_axis)
        np.testing.assert_allclose(
            np.abs(metadata.long_axis),
            np.abs(raw_long_axis),
        )
        closing_axis = closing_axis_from_orientation(gripper_pose[:3, :3])
        assert (
            abs(float(closing_axis @ metadata.object_pose[:3, 1]))
            < 1e-8
        )
        assert abs(float(abs(closing_axis @ metadata.object_pose[:3, 0]) - 1.0)) < 1e-8


def test_pen_capture_raw_x_and_raw_z_long_axes_both_normalize_to_policy_y() -> None:
    capture_objects = pen_160958_capture_objects()

    for label, pose, size in capture_objects:
        z_up_pose = screwdriver_handle_z_up_object_pose(pose, size)
        raw_long_axis_index = screwdriver_handle_flowpose_long_axis_index(size)
        raw_long_axis = horizontal_unit(pose[:3, raw_long_axis_index])

        assert screwdriver_handle_long_axis_index(size) == 1
        np.testing.assert_allclose(
            z_up_pose[:3, 1],
            raw_long_axis,
        )
        np.testing.assert_allclose(
            np.cross(z_up_pose[:3, 0], z_up_pose[:3, 1]),
            z_up_pose[:3, 2],
            atol=1e-8,
        )
        assert np.linalg.det(z_up_pose[:3, :3]) > 0.999999


def test_generic_template_path_is_unchanged() -> None:
    target = make_target("yellow_cube")
    relative_waypoints = [(np.array([0.0, 0.0, 0.1]), (0.0, 0.0, 0.0, 1.0), 0.0)]

    waypoints = build_pick_template_waypoints(
        target,
        relative_waypoints,
        args(
            ik_downward_tilt_left_deg=-45.0,
            ik_downward_tilt_axis="z",
            ik_downward_tilt_y_left_deg=45.0,
        ),
        hand="left",
    )

    rotation = quaternion_to_rotation_matrix(waypoints[0][1])
    assert not np.allclose(rotation[:3, 2], [0.0, 0.0, 1.0])


def args(**overrides) -> Namespace:
    values = {
        "ik_grasp_tcp_offset_m": (0.0, 0.0, 0.0),
        "ik_target_stage": "grasp",
        "ik_orientation_quat": (0.0, 0.0, 0.0, 1.0),
        "ik_downward_tilt_deg": 0.0,
        "ik_downward_tilt_left_deg": None,
        "ik_downward_tilt_right_deg": None,
        "ik_downward_tilt_axis": "y",
        "ik_downward_tilt_y_deg": 0.0,
        "ik_downward_tilt_y_left_deg": None,
        "ik_downward_tilt_y_right_deg": None,
        "ik_downward_tilt_frame": "local",
        "use_flowpose_grasp_rotation": True,
        "approach_axis": "z",
        "approach_sign": -1.0,
        "pregrasp_distance_m": 0.05,
        "ik_pregrasp_extra_offset_m": (0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return Namespace(**values)


def make_screwdriver_target(
    long_axis: np.ndarray,
    *,
    label: str = "red_screwdriver_handle",
) -> TargetObjectPose:
    pose = np.eye(4)
    x_axis = long_axis / np.linalg.norm(long_axis)
    y_axis = np.cross([0.0, 0.0, 1.0], x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = [0.2, 0.1, 0.8]
    return TargetObjectPose(
        label=label,
        frame_id=f"{label}_1",
        camera_pose=np.eye(4),
        base_pose=pose,
        size=np.array([0.12, 0.02, 0.02]),
    )


def horizontal_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)[:2]))


def horizontal_unit(vector: np.ndarray) -> np.ndarray:
    projected = np.asarray(vector, dtype=np.float64).copy()
    projected[2] = 0.0
    return projected / np.linalg.norm(projected)


def pen_160958_capture_objects() -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    return (
        (
            "pen_1",
            np.array(
                [
                    [-0.9908349514007568, -0.13507558405399323, 0.0008445863495580852, 0.02061368338763714],
                    [-0.1288204938173294, 0.94303297996521, -0.3067471981048584, -0.05671478435397148],
                    [0.040637608617544174, -0.30404457449913025, -0.9517906308174133, 0.6330291628837585],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.15033143758773804, 0.012042555958032608, 0.018913574516773224]),
        ),
        (
            "pen_2",
            np.array(
                [
                    [-0.8879576921463013, -0.458947092294693, 0.029979951679706573, 0.07844527065753937],
                    [-0.398145467042923, 0.7344159483909607, -0.5496487021446228, -0.16830554604530334],
                    [0.23024185001850128, -0.5000012516975403, -0.8348578214645386, 0.6767749190330505],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.04699477180838585, 0.022977419197559357, 0.15271708369255066]),
        ),
    )


def make_target(label: str) -> TargetObjectPose:
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.1, 0.8]
    return TargetObjectPose(
        label=label,
        frame_id=f"{label}_1",
        camera_pose=np.eye(4),
        base_pose=pose,
    )
