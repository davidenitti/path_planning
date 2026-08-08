#!/usr/bin/env python3
"""Drive a Hybrid A* demo vehicle interactively with the keyboard.

Examples:
    python hybrid_astar_game.py
    python hybrid_astar_game.py --env walls --primitive_length 0.4

Controls:
    Up/W: move forward           Down/S: move backward
    Left/A: steer left           Right/D: steer right
    R: reset to the environment start pose
    Q/Escape: close the game

Pressing a drive key executes one fine-length primitive immediately; holding it
repeats movement after a short delay. Hold a steering key together with a drive
key to turn at the vehicle's maximum steering angle. Steering keys used alone
only turn the displayed front wheels, and releasing them returns the wheels to
straight. Opposing drive or steering keys cancel each other. A movement step is
blocked if any swept collision sample intersects an obstacle or the
safety-inflated world boundary.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional


import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, TimerBase
from matplotlib.patches import FancyArrowPatch, Polygon

from hybrid_astar_main import draw_environment, draw_goal_pose, draw_goal_region
from hybrid_astar_planner import (
    Environment,
    HybridAStar,
    Node,
    Vehicle,
    make_environment,
    vehicle_heading_arrow,
    vehicle_polygon,
    vehicle_tire_polygons,
    wrap,
)


@dataclass
class CarGame:
    """Hold interactive vehicle state and apply collision-checked controls."""

    planner: HybridAStar
    env: Environment
    pose: tuple[float, float, float]
    status: str = "Ready"
    steer: float = 0.0

    @classmethod
    def from_planner(cls, planner: HybridAStar, env: Environment) -> "CarGame":
        """Create a game positioned at the environment start pose.

        Args:
            planner: Planner providing the exact primitive collision checker.
            env: Environment providing start and goal poses.

        Returns:
            A game ready to receive driving commands.
        """
        return cls(planner=planner, env=env, pose=env.start)

    def move(self, direction: int, steer_index: int) -> bool:
        """Apply one fine primitive when all swept-path samples are collision-free.

        Args:
            direction: ``1`` for forward motion or ``-1`` for reverse motion.
            steer_index: Index into the planner's discrete steering values.

        Returns:
            ``True`` if the pose changed; ``False`` when the primitive was blocked.
        """
        node = Node(
            *self.pose,
            0.0,
            None,
            direction,
            steer_index,
            0.0,
        )
        endpoint = self.planner.check_primitive(
            node,
            direction,
            steer_index,
            self.planner._collision_distances,
        )
        if endpoint is None:
            self.status = "Blocked: the swept safety envelope would collide"
            return False
        self.pose = endpoint
        self.steer = float(self.planner.steers[steer_index])
        self.status = "Moved"
        return True

    def reset(self) -> None:
        """Return the vehicle to the environment start pose."""
        self.pose = self.env.start
        self.steer = 0.0
        self.status = "Reset to start"

    def set_steer(self, steer_index: int) -> None:
        """Set the displayed steering angle.

        Args:
            steer_index: Index into the planner's discrete steering values.
        """
        self.steer = float(self.planner.steers[steer_index])

    def goal_reached(self) -> bool:
        """Return whether the current pose meets the demo's goal tolerances.

        Returns:
            Whether position and heading errors both fall within planner tolerances.
        """
        x, y, yaw = self.pose
        gx, gy, gyaw = self.env.goal
        return (
            math.hypot(x - gx, y - gy) <= self.planner.position_tolerance
            and abs(wrap(yaw - gyaw)) <= self.planner.yaw_tolerance
        )


class GameView:
    """Render and control a ``CarGame`` using a Matplotlib window."""

    def __init__(
        self,
        game: CarGame,
        hold_delay_ms: int = 50,
        move_interval_ms: int = 100,
    ) -> None:
        """Create a reusable interactive scene view.

        Args:
            game: Game state to display and control.
            hold_delay_ms: Delay after the first step before held-key movement.
            move_interval_ms: Interval between repeated held-key steps.
        """
        self.game = game
        self.hold_delay_seconds = hold_delay_ms / 1000.0
        self.move_interval_seconds = move_interval_ms / 1000.0
        self.drive_pressed_at: dict[str, float] = {}
        self.last_motion_at: dict[str, float] = {}
        self.release_timers: dict[str, TimerBase] = {}
        self.figure, self.ax = plt.subplots(figsize=(12, 7))
        self.figure.canvas.manager.set_window_title("Hybrid A* keyboard driver")
        draw_environment(self.ax, game.env)
        draw_goal_region(self.ax, game.planner, game.env)
        draw_goal_pose(self.ax, game.planner, game.env)
        self.goal_pose = next(patch for patch in self.ax.patches if patch.get_label() == "Goal pose")
        self.goal_pose_default_facecolor = self.goal_pose.get_facecolor()
        self.goal_pose_default_alpha = self.goal_pose.get_alpha()

        # Matplotlib's Polygon constructor requires a two-dimensional vertex
        # array. These placeholders are replaced by ``update`` immediately.
        placeholder = [(0.0, 0.0)] * 4
        self.safety_body = Polygon(
            placeholder,
            closed=True,
            fill=False,
            linestyle=":",
            linewidth=2.2,
            alpha=0.0,
            zorder=5,
        )
        self.body = Polygon(placeholder, closed=True, fill=False, linewidth=2.0, zorder=6)
        self.tires = [
            Polygon(placeholder, closed=True, facecolor="black", edgecolor="black", zorder=7)
            for _ in range(4)
        ]
        self.heading = FancyArrowPatch(
            (0, 0),
            (0, 0),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2.0,
            color="tab:blue",
            zorder=7,
        )
        self.rear_axle = self.ax.scatter([], [], color="white", edgecolor="black", s=28, zorder=10)
        self.ax.add_patch(self.safety_body)
        self.ax.add_patch(self.body)
        for tire in self.tires:
            self.ax.add_patch(tire)
        self.ax.add_patch(self.heading)
        self.pressed_keys: set[str] = set()
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("draw_event", self.on_draw)
        self.motion_timer = self.figure.canvas.new_timer(interval=move_interval_ms)
        self.motion_timer.add_callback(self.on_motion_timer)
        self.motion_timer.start()
        self.update()

    def focus_canvas(self) -> None:
        """Give the plotting canvas keyboard focus when the backend supports it."""
        canvas = self.figure.canvas
        if hasattr(canvas, "get_tk_widget"):
            canvas.get_tk_widget().focus_set()
        elif hasattr(canvas, "setFocus"):
            canvas.setFocus()
        elif hasattr(canvas, "SetFocus"):
            canvas.SetFocus()

    def on_draw(self, _event: object) -> None:
        """Focus the canvas after its native window has been created."""
        self.focus_canvas()

    def on_click(self, _event: object) -> None:
        """Restore keyboard focus after a click in the game window."""
        self.focus_canvas()

    @staticmethod
    def normalized_key(event: KeyEvent) -> str:
        """Return one canonical control name from a Matplotlib key event.

        Args:
            event: Matplotlib keypress event.

        Returns:
            ``up``, ``down``, ``left``, or ``right`` for either arrow/WASD
            input, otherwise the normalized original key name.
        """
        key = event.key.lower().split("+")[-1] if event.key else ""
        return {
            "w": "up",
            "s": "down",
            "a": "left",
            "d": "right",
        }.get(key, key)

    def active_steer_index(self) -> int:
        """Return straight or maximum steering from the currently held keys.

        Returns:
            Index of the active planner steering value. Opposing steering keys
            cancel to straight.
        """
        steer_left = len(self.game.planner.steers) - 1
        steer_straight = len(self.game.planner.steers) // 2
        steer_right = 0
        if "left" in self.pressed_keys and "right" not in self.pressed_keys:
            return steer_left
        if "right" in self.pressed_keys and "left" not in self.pressed_keys:
            return steer_right
        return steer_straight

    def update_steering(self) -> None:
        """Apply held steering keys without translating the vehicle."""
        steer_index = self.active_steer_index()
        self.game.set_steer(steer_index)
        if steer_index == len(self.game.planner.steers) - 1:
            self.game.status = "Steering full left"
        elif steer_index == 0:
            self.game.status = "Steering full right"
        else:
            self.game.status = "Steering straight"

    def on_key_press(self, event: KeyEvent) -> None:
        """Record a held drive/steering key or apply a one-shot command.

        Args:
            event: Matplotlib keypress event.
        """
        key = self.normalized_key(event)
        if key in {"q", "escape"}:
            plt.close(self.figure)
            return
        if key == "r":
            self.pressed_keys.clear()
            self.drive_pressed_at.clear()
            self.last_motion_at.clear()
            for release_timer in self.release_timers.values():
                release_timer.stop()
            self.release_timers.clear()
            self.game.reset()
        elif key in {"up", "down", "left", "right"}:
            release_timer = self.release_timers.pop(key, None)
            if release_timer is not None:
                release_timer.stop()
            already_pressed = key in self.pressed_keys
            if already_pressed:
                return
            self.pressed_keys.add(key)
            if key in {"left", "right"}:
                self.update_steering()
            else:
                self.game.status = "Drive key held"
                now = time.monotonic()
                self.drive_pressed_at[key] = now
                self.last_motion_at[key] = now
                opposite = "down" if key == "up" else "up"
                if opposite not in self.pressed_keys:
                    direction = 1 if key == "up" else -1
                    self.game.move(direction, self.active_steer_index())
        else:
            return
        self.update()

    def on_key_release(self, event: KeyEvent) -> None:
        """Stop the released drive direction or restore the wheel angle.

        Args:
            event: Matplotlib key-release event.
        """
        key = self.normalized_key(event)
        if key not in {"up", "down", "left", "right"}:
            return
        release_timer = self.release_timers.pop(key, None)
        if release_timer is not None:
            release_timer.stop()
        release_timer = self.figure.canvas.new_timer(interval=20)
        release_timer.single_shot = True
        release_timer.add_callback(self.finish_key_release, key)
        self.release_timers[key] = release_timer
        release_timer.start()

    def finish_key_release(self, key: str) -> None:
        """Apply a delayed key release after filtering backend auto-repeat.

        Args:
            key: Canonical drive or steering key to release.

        Returns:
            None.
        """
        self.release_timers.pop(key, None)
        self.pressed_keys.discard(key)
        if key in {"up", "down"}:
            self.drive_pressed_at.pop(key, None)
            self.last_motion_at.pop(key, None)
        if key in {"left", "right"}:
            self.update_steering()
        elif not ({"up", "down"} & self.pressed_keys):
            self.game.status = "Stopped"
        self.update()

    def on_motion_timer(self) -> None:
        """Advance one primitive according to the currently held drive keys."""
        forward = "up" in self.pressed_keys
        backward = "down" in self.pressed_keys
        if forward == backward:
            return
        key = "up" if forward else "down"
        now = time.monotonic()
        if now - self.drive_pressed_at[key] < self.hold_delay_seconds:
            return
        if now - self.last_motion_at[key] + 1e-12 < self.move_interval_seconds:
            return
        direction = 1 if forward else -1
        self.game.move(direction, self.active_steer_index())
        self.last_motion_at[key] = now
        self.update()

    def update(self) -> None:
        """Refresh vehicle artists and the compact control/status message."""
        game = self.game
        x, y, yaw = game.pose
        vehicle = game.planner.vehicle
        self.safety_body.set_xy(vehicle_polygon(x, y, yaw, vehicle, game.planner.safety_margin))
        self.body.set_xy(vehicle_polygon(x, y, yaw, vehicle))
        for tire, points in zip(self.tires, vehicle_tire_polygons(x, y, yaw, game.steer, vehicle)):
            tire.set_xy(points)
        self.heading.set_positions(*vehicle_heading_arrow(x, y, yaw, vehicle))
        self.rear_axle.set_offsets([[x, y]])
        goal_reached = game.goal_reached()
        if goal_reached:
            self.goal_pose.set_fill(True)
            self.goal_pose.set_facecolor("tab:green")
            self.goal_pose.set_alpha(0.55)
        else:
            self.goal_pose.set_fill(False)
            self.goal_pose.set_facecolor(self.goal_pose_default_facecolor)
            self.goal_pose.set_alpha(self.goal_pose_default_alpha)
        goal_text = " Goal reached!" if goal_reached else ""
        self.ax.set_title(
            f"{game.env.title} - {game.status}{goal_text}\n"
            "Up/W forward, Down/S reverse; hold Left/A or Right/D while moving to steer; "
            "R reset, Q quit\n"
            f"Pose: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°) - click the plot if keys are ignored"
        )
        self.figure.canvas.draw_idle()


def positive_float(value: str) -> float:
    """Parse a finite positive CLI float.

    Args:
        value: Text supplied to argparse.

    Returns:
        Parsed positive value.

    Raises:
        ValueError: If conversion to float fails.
        argparse.ArgumentTypeError: If the value is not finite and positive.
    """
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def nonnegative_float(value: str) -> float:
    """Parse a finite non-negative CLI float.

    Args:
        value: Text supplied to argparse.

    Returns:
        Parsed non-negative value.

    Raises:
        ValueError: If conversion to float fails.
        argparse.ArgumentTypeError: If the value is not finite and non-negative.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def positive_int(value: str) -> int:
    """Parse a positive CLI integer.

    Args:
        value: Text supplied to argparse.

    Returns:
        Parsed positive integer.

    Raises:
        ValueError: If conversion to int fails.
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_parser() -> argparse.ArgumentParser:
    """Create the game CLI using the demo's corresponding default values.

    Returns:
        Configured command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=("walls", "maze", "parking", "parking2", "parking2_hard", "parking3", "parking4"),
        default="parking",
        help="Demo environment to drive in.",
    )
    parser.add_argument(
        "--primitive_length",
        type=positive_float,
        default=0.2,
        help="Distance moved per movement step [m].",
    )
    parser.add_argument(
        "--safety_margin", type=nonnegative_float, default=0.20, help="Clearance around the vehicle [m]."
    )
    parser.add_argument(
        "--position_tolerance", type=positive_float, default=0.2, help="Goal position tolerance [m]."
    )
    parser.add_argument(
        "--yaw_tolerance_deg", type=positive_float, default=1.5, help="Goal yaw tolerance [deg]."
    )
    parser.add_argument(
        "--collision_check_step",
        type=positive_float,
        default=0.05,
        help="Swept collision sample spacing [m].",
    )
    parser.add_argument(
        "--integration_step",
        type=positive_float,
        default=0.1,
        help="Planner reconstruction sample spacing [m].",
    )
    parser.add_argument(
        "--xy_resolution", type=positive_float, default=0.15, help="Demo planner x/y resolution [m]."
    )
    parser.add_argument(
        "--yaw_resolution_deg", type=positive_float, default=1.0, help="Demo planner yaw resolution [deg]."
    )
    parser.add_argument(
        "--hold_delay_ms",
        type=positive_int,
        default=150,
        help="Delay before a held drive key starts repeating [ms].",
    )
    parser.add_argument(
        "--move_interval_ms",
        type=positive_int,
        default=100,
        help="Time between repeated movement steps while holding a drive key [ms].",
    )
    return parser


def create_game(args: argparse.Namespace) -> CarGame:
    """Build a game with the demo's vehicle, environment, and collision checker.

    Args:
        args: Parsed game command-line arguments.

    Returns:
        Interactive game state at the selected environment's start pose.
    """
    planner_options = {
        "xy_resolution": args.xy_resolution,
        "yaw_resolution": math.radians(args.yaw_resolution_deg),
        "primitive_length": args.primitive_length,
        "position_tolerance": args.position_tolerance,
        "yaw_tolerance": math.radians(args.yaw_tolerance_deg),
        "reverse_multiplier": 1.0,
        "gear_change_penalty": 0.0,
        "steering_change_penalty": 0.0,
    }
    env = make_environment(args.env, planner_options)
    planner = HybridAStar(
        environment=env,
        vehicle=Vehicle(),
        safety_margin=args.safety_margin,
        integration_step=args.integration_step,
        collision_check_step=args.collision_check_step,
    )
    return CarGame.from_planner(planner, env)


def main(args: Optional[argparse.Namespace] = None) -> CarGame:
    """Start the keyboard-driving window.

    Args:
        args: Optional pre-parsed arguments; CLI arguments are parsed when omitted.

    Returns:
        The game state after its window closes.
    """
    if args is None:
        args = make_parser().parse_args()
    game = create_game(args)
    view = GameView(
        game,
        hold_delay_ms=args.hold_delay_ms,
        move_interval_ms=args.move_interval_ms,
    )
    plt.show()
    del view
    return game


if __name__ == "__main__":
    main()
