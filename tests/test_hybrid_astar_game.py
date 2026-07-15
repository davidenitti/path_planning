import math

import matplotlib

matplotlib.use("Agg")

from matplotlib.backend_bases import KeyEvent

from hybrid_astar_demo import Environment, HybridAStar, Obstacle, Vehicle
from hybrid_astar_game import CarGame, GameView, create_game, make_parser


def test_game_defaults_match_the_demo_defaults() -> None:
    args = make_parser().parse_args([])
    game = create_game(args)

    assert game.env.name == "parking"
    assert game.planner.fine_primitive_length == 0.2
    assert game.planner.safety_margin == 0.2
    assert game.planner.position_tolerance == 0.2
    assert game.planner.yaw_tolerance == math.radians(1.5)
    assert game.planner.collision_check_step == 0.05


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

