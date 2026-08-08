import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.backend_bases import KeyEvent

from hybrid_astar_game import CarGame, GameView, create_game, make_parser
from hybrid_astar_planner import Environment, HybridAStar, Obstacle, Vehicle, make_environment


def test_game_defaults_match_the_demo_defaults() -> None:
    args = make_parser().parse_args([])
    game = create_game(args)

    assert game.env.name == "parking"
    assert game.planner.primitive_length == 0.2
    assert game.planner.safety_margin == 0.2
    assert game.planner.position_tolerance == 0.2
    assert game.planner.yaw_tolerance == math.radians(1.5)
    assert game.planner.collision_check_step == 0.05


def test_standalone_html_maze_matches_python_environment() -> None:
    html_path = Path(__file__).parents[1] / "hybrid_astar_game_standalone.html"
    html = html_path.read_text(encoding="utf-8")
    source_match = re.search(
        r"const MAZE_SOURCE_BOXES = Object\.freeze\((\[.*?\])\);",
        html,
        flags=re.DOTALL,
    )

    assert source_match is not None
    source_boxes = json.loads(source_match.group(1))
    constants: dict[str, float] = {}
    for name in (
        "MAZE_SVG_SCALE",
        "MAZE_FRAME_LEFT",
        "MAZE_FRAME_RIGHT",
        "MAZE_FRAME_TOP",
        "MAZE_FRAME_BOTTOM",
    ):
        constant_match = re.search(rf"const {name} = ([0-9.]+);", html)
        assert constant_match is not None
        constants[name] = float(constant_match.group(1))
    scale = constants["MAZE_SVG_SCALE"]
    frame_left = constants["MAZE_FRAME_LEFT"]
    frame_bottom = constants["MAZE_FRAME_BOTTOM"]
    html_obstacles = tuple(
        Obstacle(
            (xmin - frame_left) * scale,
            (xmax - frame_left) * scale,
            (frame_bottom - ymax) * scale,
            (frame_bottom - ymin) * scale,
        )
        for xmin, xmax, ymin, ymax in source_boxes
    )
    python_maze = make_environment("maze", {})

    assert html_obstacles == python_maze.obstacles
    assert math.isclose(
        (constants["MAZE_FRAME_RIGHT"] - frame_left) * scale,
        python_maze.width,
    )
    assert math.isclose(
        (frame_bottom - constants["MAZE_FRAME_TOP"]) * scale,
        python_maze.height,
    )
    assert "title: \"Classic branching maze environment\"" in html
    assert "width: (MAZE_FRAME_RIGHT - MAZE_FRAME_LEFT) * MAZE_SVG_SCALE" in html
    assert "height: (MAZE_FRAME_BOTTOM - MAZE_FRAME_TOP) * MAZE_SVG_SCALE" in html
    assert "start: [(46.540 - MAZE_FRAME_LEFT) * MAZE_SVG_SCALE, 51.5, -90 * DEG]" in html
    assert "goal: [(431.147 - MAZE_FRAME_LEFT) * MAZE_SVG_SCALE, 6.0, -90 * DEG]" in html


def test_game_keeps_pose_when_swept_primitive_collides() -> None:
    env = Environment(
        name="blocked",
        title="Blocked",
        width=20.0,
        height=12.0,
        obstacles=(Obstacle(7.5, 8.0, 0.0, 12.0),),
        start=(4.0, 6.0, 0.0),
        goal=(12.0, 6.0, 0.0),
        planner={
            "xy_resolution": 0.15,
            "yaw_resolution": math.radians(1.0),
            "primitive_length": 0.2,
            "position_tolerance": 0.2,
            "yaw_tolerance": math.radians(1.5),
            "reverse_multiplier": 1.0,
            "gear_change_penalty": 0.0,
            "steering_change_penalty": 0.0,
        },
    )
    planner = HybridAStar(env, Vehicle(), 0.0, 0.1, 0.05)
    game = CarGame.from_planner(planner, env)
    straight = len(planner.steers) // 2

    assert not game.move(1, straight)
    assert game.pose == env.start
    assert game.status.startswith("Blocked")


def test_backward_uses_held_right_steering() -> None:
    game = create_game(make_parser().parse_args([]))
    view = GameView(game)
    initial_pose = game.pose

    for key in ("right", "down"):
        view.on_key_press(KeyEvent("key_press_event", view.figure.canvas, key=key))

    assert game.pose[0] < initial_pose[0]
    assert game.pose[2] > initial_pose[2]
    assert game.steer == -game.planner.vehicle.max_steer
    view.figure.canvas.manager.destroy()
