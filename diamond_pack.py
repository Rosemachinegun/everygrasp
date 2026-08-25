#!/usr/bin/env python3
"""One-window dashboard for dual cameras, tactile sensors, and force gripper.
    包含全部夹爪功能的单窗口dashboard样例
Controls:
    L: close gripper with force/current-limited grasp
    P: open gripper
    R: reset selected tactile sensors
    Q/ESC: quit

"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daimon_stuff.dm_gripper_cam_py.viewer_config import camera_specs_from_args  # noqa: E402
from daimon_stuff.dm_gripper_cam_py.dashboard import make_camera_dashboard  # noqa: E402
from daimon_stuff.dm_gripper_cam_py.worker import CameraWorker  # noqa: E402
from grasp_core.communication.gripper_signal import (  # noqa: E402
    DEFAULT_RECEIVER_PATH,
    send_gripper_signal,
    start_gripper_signal_receiver,
    stop_gripper_signal_receiver,
)
from daimon_stuff.dm_gripper_tac_py.tac import (  # noqa: E402
    SensorManager,
    disable_proxy_for_host,
    make_dashboard as make_tactile_dashboard,
    make_options as make_tactile_options,
)


WINDOW_NAME = "Diamond Pack Dashboard"
QUIT_KEYS = {ord("q"), 27}
KEY_GRIP = ord("l")
KEY_RELEASE = ord("p")
KEY_RESET_TACTILE = ord("r")

CAMERA_PANEL_SIZE = (480, 270)
TACTILE_PANEL_SIZE = (640, 480)
STATUS_HEIGHT = 56

DEFAULT_LEFT_GRIPPER_SERVER = "192.168.10.10:55551"
DEFAULT_RIGHT_GRIPPER_SERVER = "192.168.10.11:55551"


@dataclass(frozen=True)
class TactileSpec:
    """One tactile sensor's fixed connection tuple."""

    sensor_id: int
    remote_addr: str
    dev_id: str
    pc_host: str
    pc_port: int


TACTILE_SPECS: dict[int, TactileSpec] = {
    1: TactileSpec(
        sensor_id=1,
        remote_addr="192.168.10.11:50052",
        dev_id="2",
        pc_host="192.168.10.123",
        pc_port=60031,
    ),
    2: TactileSpec(
        sensor_id=2,
        remote_addr="192.168.10.11:50051",
        dev_id="0",
        pc_host="192.168.10.123",
        pc_port=60030,
    ),
    3: TactileSpec(
        sensor_id=3,
        remote_addr="192.168.10.10:50052",
        dev_id="2",
        pc_host="192.168.10.123",
        pc_port=60033,
    ),
    4: TactileSpec(
        sensor_id=4,
        remote_addr="192.168.10.10:50051",
        dev_id="0",
        pc_host="192.168.10.123",
        pc_port=60032,
    ),
}


def tactile_specs_from_selection(
    sensor_ids: list[int] | None = None,
    count: int | None = None,
) -> list[TactileSpec]:
    """Return tactile specs by user-facing ids 1..4.

    Examples:
        tactile_specs_from_selection([1, 3]) -> sensors 1 and 3
        tactile_specs_from_selection(count=2) -> sensors 1 and 2
        tactile_specs_from_selection() -> all four sensors
    """

    if sensor_ids:
        ids = sensor_ids
    elif count is not None:
        ids = list(sorted(TACTILE_SPECS))[: int(count)]
    else:
        ids = list(sorted(TACTILE_SPECS))

    specs: list[TactileSpec] = []
    seen: set[int] = set()
    for sensor_id in ids:
        if sensor_id in seen:
            continue
        seen.add(sensor_id)
        if sensor_id not in TACTILE_SPECS:
            valid = ", ".join(str(key) for key in sorted(TACTILE_SPECS))
            raise ValueError(f"Unknown tactile id {sensor_id}; valid ids: {valid}")
        specs.append(TACTILE_SPECS[sensor_id])
    return specs


def make_tactile_args(spec: TactileSpec, args: argparse.Namespace) -> argparse.Namespace:
    """Build the small args object expected by tactile helpers."""

    return argparse.Namespace(
        dev_id=spec.dev_id,
        backend=args.backend,
        remote_addr=spec.remote_addr,
        pc_host=spec.pc_host,
        pc_port=spec.pc_port,
        max_fps=args.max_fps,
        force=args.force,
    )


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(high, value))


def normalize_key(key: int) -> int:
    key &= 0xFF
    if ord("A") <= key <= ord("Z"):
        return key + 32
    return key


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_blank(size: tuple[int, int], text: str) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (18, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class TactileWorker:
    """Read tactile frames in the background and keep the newest dashboard data."""

    def __init__(self, spec: TactileSpec, args: argparse.Namespace) -> None:
        self.spec = spec
        self.args = args
        self.sensor: SensorManager | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_data: dict | None = None
        self.error: str | None = None
        self.opening = False
        self.frames_read = 0

    def open(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        with self.lock:
            self.opening = True
            self.error = None
        print(
            "[tactile] opening "
            f"id={self.spec.sensor_id} remote={self.args.remote_addr} "
            f"dev_id={self.args.dev_id} pc={self.args.pc_host}:{self.args.pc_port} "
            f"backend={self.args.backend}",
            flush=True,
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_sensor(self) -> None:
        disable_proxy_for_host(self.args.remote_addr)
        self.sensor = SensorManager(make_tactile_options(self.args))
        with self.lock:
            self.opening = False
            self.error = None
        print(f"[tactile] Tactile {self.spec.sensor_id} opened", flush=True)

    def _run(self) -> None:
        try:
            self._open_sensor()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.opening = False
                self.error = str(exc)
            print(
                f"[tactile] Tactile {self.spec.sensor_id} failed to open: {exc}",
                flush=True,
            )
            return

        while not self.stop_event.is_set():
            assert self.sensor is not None
            try:
                if str(self.args.backend).strip().lower() == "flux":
                    self.sensor.sensor.getEvents()
                if not self.sensor.update():
                    with self.lock:
                        self.error = "no new frame"
                    continue
                data = self.sensor.read()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.error = str(exc)
                time.sleep(0.2)
                continue
            with self.lock:
                self.latest_data = data
                self.error = None
                self.frames_read += 1

    def latest_dashboard(self) -> np.ndarray:
        with self.lock:
            data = dict(self.latest_data) if self.latest_data is not None else None
            error = self.error
            opening = self.opening
            frames_read = self.frames_read
        if data is None:
            if opening:
                text = f"Tactile {self.spec.sensor_id} opening"
            elif error:
                text = f"Tactile {self.spec.sensor_id} waiting: {error}"
            else:
                text = f"Tactile {self.spec.sensor_id} waiting"
            return make_blank(TACTILE_PANEL_SIZE, text)
        dashboard = make_tactile_dashboard(data)
        if error and frames_read == 0:
            return draw_label(dashboard, f"Tactile {self.spec.sensor_id}: {error}")
        return dashboard

    def reset(self) -> str:
        if self.opening:
            return f"Tactile {self.spec.sensor_id} is opening"
        if self.sensor is None:
            return f"Tactile {self.spec.sensor_id} is not open"
        try:
            self.sensor.reset()
        except Exception as exc:  # noqa: BLE001
            return f"Tactile {self.spec.sensor_id} reset failed: {exc}"
        return f"Tactile {self.spec.sensor_id} reset"

    def close(self) -> None:
        self.stop_event.set()
        if (
            self.thread is not None
            and self.thread.is_alive()
            and threading.current_thread() is not self.thread
        ):
            self.thread.join(timeout=1.0)
        if self.sensor is not None:
            self.sensor.close()


def make_tactile_grid(
    dashboards: list[tuple[TactileSpec, np.ndarray]],
    tile_size: tuple[int, int] = (480, 360),
) -> np.ndarray:
    """Arrange the selected tactile dashboards in a compact 1/2/3/4 grid."""

    if not dashboards:
        return make_blank(TACTILE_PANEL_SIZE, "No tactile sensors selected")

    tiles: list[np.ndarray] = []
    for spec, dashboard in dashboards:
        tile = cv2.resize(dashboard, tile_size, interpolation=cv2.INTER_AREA)
        tiles.append(draw_label(tile, f"Tactile {spec.sensor_id}"))

    columns = 1 if len(tiles) == 1 else 2
    rows: list[np.ndarray] = []
    blank = make_blank(tile_size, "")
    for row_start in range(0, len(tiles), columns):
        row_tiles = tiles[row_start : row_start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(blank.copy())
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


class DiamondPackApp:
    """Coordinates remote cameras, tactile visualization, and gripper commands."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.camera_workers = [
            CameraWorker(spec, args) for spec in camera_specs_from_args(args)
        ]
        self.tactile_specs = tactile_specs_from_selection(
            args.tactile_ids,
            args.tactile_count,
        )
        self.tactile_workers = [
            TactileWorker(spec, make_tactile_args(spec, args))
            for spec in self.tactile_specs
        ]
        self.gripper_receiver = None
        self.command_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gripper-command",
        )
        self.gripper_future: Future | None = None
        self.status = "Ready"

    def open(self) -> None:
        for worker in self.camera_workers:
            try:
                worker.open()
                worker.start(self.args.read_timeout)
            except Exception as exc:  # noqa: BLE001
                worker.error = str(exc)
                print(f"[camera] {worker.spec.name} failed to open: {exc}", flush=True)

        for worker in self.tactile_workers:
            worker.open()
            time.sleep(self.args.tactile_startup_stagger_sec)

        self.gripper_receiver = start_gripper_signal_receiver(self.args)
        ids = ",".join(str(spec.sensor_id) for spec in self.tactile_specs) or "none"
        self.status = f"Ready: tactile={ids}; L close, P open, R reset tactile, Q quit"

    def close(self) -> None:
        for worker in self.camera_workers:
            worker.close()
        for worker in self.tactile_workers:
            worker.close()
        stop_gripper_signal_receiver(self.gripper_receiver)
        self.command_executor.shutdown(wait=True, cancel_futures=True)
        cv2.destroyAllWindows()

    def _collect_gripper_result(self) -> None:
        if self.gripper_future is None or not self.gripper_future.done():
            return
        try:
            self.status = self.gripper_future.result()
        except Exception as exc:  # noqa: BLE001
            self.status = f"Gripper command failed: {exc}"
        self.gripper_future = None

    def send_gripper(self, command: str) -> None:
        if self.gripper_future is not None and not self.gripper_future.done():
            self.status = "Gripper command already running"
            return
        hand = str(getattr(self.args, "manual_gripper_hand", "right"))
        self.status = f"Gripper {command} running hand={hand}"
        self.gripper_future = self.command_executor.submit(
            send_gripper_signal,
            command,
            self.args,
            hand,
        )

    def render(self) -> np.ndarray:
        frames = []
        any_open = False
        for worker in self.camera_workers:
            if worker.cap.isOpened():
                any_open = True
            frames.append(worker.latest_frame())
            worker.maybe_print_stats(self.args.stats_interval)

        camera_panel = make_camera_dashboard(frames, CAMERA_PANEL_SIZE)
        camera_panel = draw_label(camera_panel, "Dual Camera")

        tactile_panel = make_tactile_grid(
            [
                (worker.spec, worker.latest_dashboard())
                for worker in self.tactile_workers
            ]
        )

        width = max(camera_panel.shape[1], tactile_panel.shape[1])
        camera_panel = resize_to_width(camera_panel, width)
        tactile_panel = resize_to_width(tactile_panel, width)

        status_panel = np.zeros((STATUS_HEIGHT, width, 3), dtype=np.uint8)
        camera_state = "camera:on" if any_open else "camera:waiting"
        status_text = (
            f"{self.status} | {camera_state} | "
            f"force close threshold={self.args.gripper_current_threshold}"
        )
        cv2.putText(
            status_panel,
            status_text[:150],
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([camera_panel, tactile_panel, status_panel])

    def handle_key(self, key: int) -> bool:
        key = normalize_key(key)
        if key in QUIT_KEYS:
            return False
        if key == KEY_GRIP:
            self.send_gripper("grip")
        elif key == KEY_RELEASE:
            self.send_gripper("release")
        elif key == KEY_RESET_TACTILE:
            results = [worker.reset() for worker in self.tactile_workers]
            self.status = "; ".join(results) if results else "No tactile sensors selected"
        return True

    def run(self) -> int:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            self._collect_gripper_result()
            cv2.imshow(WINDOW_NAME, self.render())
            if not self.handle_key(cv2.waitKey(1)):
                return 0


def add_gripper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gripper-server", default=DEFAULT_RIGHT_GRIPPER_SERVER)
    parser.add_argument(
        "--dual-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start/control left and right gripper receivers. "
            "Left uses --grip-signal-port, right uses port+1."
        ),
    )
    parser.add_argument("--left-gripper-server", default=DEFAULT_LEFT_GRIPPER_SERVER)
    parser.add_argument("--right-gripper-server", default=DEFAULT_RIGHT_GRIPPER_SERVER)
    parser.add_argument(
        "--manual-gripper-hand",
        choices=("left", "right"),
        default="right",
        help="Hand controlled by manual L/P gripper keys.",
    )
    parser.add_argument("--gripper-clamp-pos", type=int, default=-52525)
    parser.add_argument("--gripper-open-pos", type=int, default=-142525)
    parser.add_argument("--gripper-max-itinerary", type=int, default=90000)
    parser.add_argument("--gripper-speed-coe", type=int, default=3600)
    parser.add_argument("--gripper-calibration-tolerance", type=int, default=150)
    parser.add_argument("--gripper-connect-attempts", type=int, default=3)
    parser.add_argument("--gripper-connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--gripper-connect-retry-delay-sec", type=float, default=0.5)
    parser.add_argument(
        "--gripper-allow-homing-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow grip_signal_receiver.py to run SDK grip_init() if known "
            "calibration init fails. This performs a homing motion."
        ),
    )
    parser.add_argument("--gripper-min-pos", type=int, default=300)
    parser.add_argument("--gripper-max-pos", type=int, default=900)
    parser.add_argument("--gripper-grip-speed", type=int, default=60)
    parser.add_argument("--gripper-grip-torque", type=int, default=30)
    parser.add_argument("--gripper-hold-torque", type=int, default=10)
    parser.add_argument("--gripper-current-threshold", type=int, default=120)
    parser.add_argument("--gripper-poll-interval", type=float, default=0.05)
    parser.add_argument("--gripper-contact-grace", type=float, default=0.4)
    parser.add_argument("--gripper-progress-epsilon", type=int, default=2)
    parser.add_argument("--gripper-stall-samples", type=int, default=5)
    parser.add_argument("--gripper-timeout", type=float, default=20.0)
    parser.add_argument("--gripper-release-target", type=int, default=1000)
    parser.add_argument("--gripper-release-speed", type=int, default=40)
    parser.add_argument("--gripper-release-torque", type=int, default=20)
    parser.add_argument("--gripper-release-wait", type=float, default=0.5)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-window dual-camera + tactile + force gripper dashboard"
    )

    # Remote dual camera options mirror daimon_stuff/dm_gripper_cam_py.
    parser.add_argument("--left-host", default="192.168.10.10")
    parser.add_argument("--right-host", default="192.168.10.11")
    parser.add_argument("--port", type=int, default=50088)
    parser.add_argument("--codec", choices=("HEVC", "MJPG"), default="MJPG")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--read-timeout", type=float, default=0.5)
    parser.add_argument("--stats-interval", type=float, default=1.0)
    parser.add_argument("--client-ip", default="")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--left-udp-port", type=int, default=0)
    parser.add_argument("--right-udp-port", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--left-window-name", default="Left Camera")
    parser.add_argument("--right-window-name", default="Right Camera")

    # Tactile options: ids 1..4 map to TACTILE_SPECS above.
    parser.add_argument(
        "--tactile-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Tactile sensor ids to show. Valid ids are 1 2 3 4. Default: all.",
    )
    parser.add_argument(
        "--tactile-count",
        type=int,
        default=None,
        help="Show the first N tactile sensors by id. Ignored when --tactile-ids is set.",
    )
    parser.add_argument("--backend", default="Flux")
    parser.add_argument("--max-fps", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tactile-startup-stagger-sec",
        type=float,
        default=0.25,
        help="Delay between starting tactile workers to avoid overloading sensor servers.",
    )

    # Gripper receiver options mirror grasp_core/config/request_ik_config.py.
    parser.add_argument("--grip-signal-host", default="127.0.0.1")
    parser.add_argument("--grip-signal-port", type=int, default=55660)
    parser.add_argument("--grip-signal-timeout-sec", type=float, default=1.0)
    parser.add_argument(
        "--grip-signal-command-timeout-sec",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--grip-signal-auto-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--grip-signal-receiver-path",
        default=str(DEFAULT_RECEIVER_PATH),
    )
    parser.add_argument("--grip-signal-token", default=None)
    add_gripper_args(parser)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.tactile_count is not None:
        args.tactile_count = max(int(args.tactile_count), 0)
    args.backend = "Flux" if str(args.backend).strip().lower() == "flux" else args.backend
    args.tactile_startup_stagger_sec = max(
        float(args.tactile_startup_stagger_sec),
        0.0,
    )
    try:
        tactile_specs_from_selection(args.tactile_ids, args.tactile_count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.gripper_min_pos = int(clamp(args.gripper_min_pos, 0, 1000))
    args.gripper_max_pos = int(clamp(args.gripper_max_pos, 0, 1000))
    args.gripper_release_target = int(clamp(args.gripper_release_target, 0, 1000))
    args.gripper_grip_speed = int(clamp(args.gripper_grip_speed, 10, 100))
    args.gripper_release_speed = int(clamp(args.gripper_release_speed, 10, 100))
    args.gripper_grip_torque = int(clamp(args.gripper_grip_torque, 10, 100))
    args.gripper_hold_torque = int(clamp(args.gripper_hold_torque, 10, 100))
    args.gripper_release_torque = int(clamp(args.gripper_release_torque, 10, 100))
    args.dual_gripper = bool(args.dual_gripper)
    args.left_gripper_server = str(args.left_gripper_server)
    args.right_gripper_server = str(args.right_gripper_server)
    args.manual_gripper_hand = str(args.manual_gripper_hand)
    args.grip_signal_port = int(args.grip_signal_port)
    args.gripper_poll_interval = max(float(args.gripper_poll_interval), 0.02)
    args.gripper_contact_grace = max(float(args.gripper_contact_grace), 0.0)
    args.gripper_progress_epsilon = max(int(args.gripper_progress_epsilon), 0)
    args.gripper_stall_samples = max(int(args.gripper_stall_samples), 1)
    args.gripper_timeout = max(float(args.gripper_timeout), 0.1)
    args.gripper_release_wait = max(float(args.gripper_release_wait), 0.0)
    args.gripper_calibration_tolerance = max(
        int(args.gripper_calibration_tolerance),
        0,
    )
    args.gripper_allow_homing_fallback = bool(args.gripper_allow_homing_fallback)
    args.gripper_connect_attempts = max(int(args.gripper_connect_attempts), 1)
    args.gripper_connect_timeout_sec = max(
        float(args.gripper_connect_timeout_sec),
        0.1,
    )
    args.gripper_connect_retry_delay_sec = max(
        float(args.gripper_connect_retry_delay_sec),
        0.0,
    )
    if args.gripper_min_pos > args.gripper_max_pos:
        raise SystemExit("--gripper-min-pos must be <= --gripper-max-pos")
    return args


def main() -> int:
    args = normalize_args(build_argparser().parse_args())
    app = DiamondPackApp(args)
    try:
        app.open()
        return app.run()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
