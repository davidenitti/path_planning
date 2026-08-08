import ast
import heapq
import inspect
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import hybrid_astar_main as demo_module
import hybrid_astar_planner as planner_module
import hybrid_astar_two_queues as two_queue_module
from hybrid_astar_planner import (
    Environment,
    HybridAStar,
    Node,
    Obstacle,
    Vehicle,
    _sample_collision_free_primitive,
    arc_pose,
    make_environment,
    sample_distances,
)
from hybrid_astar_two_queues import TwoQueueHybridAStar


def make_planner(
    goal: tuple[float, float, float] = (8.0, 5.0, 0.0),
    *,
    integration_step: float = 0.1,
    collision_check_step: float = 0.05,
    heuristic_mode: str = "default",
    state_key_mode: str = "pose_control",
    heuristic_weight: float = 1.0,
    coarse_heuristic_weight: float | None = None,
    gear_change_penalty: float = 0.0,
    steering_change_penalty: float = 0.0,
    primitive_length: float = 0.5,
    position_tolerance: float = 0.3,
    yaw_tolerance: float = math.radians(2.0),
    use_two_queues: bool = False,
    coarse_primitive_mult: int = 4,
    queue_beta: float = 1.5,
    origin_priority_factor: float = 2.0,
    obstacles: tuple[Obstacle, ...] = (),
) -> HybridAStar:
    environment = Environment(
        name="test",
        title="Test environment",
        width=30.0,
        height=20.0,
        obstacles=obstacles,
        start=(5.0, 5.0, 0.0),
        goal=goal,
        planner={
            "xy_resolution": 0.25,
            "yaw_resolution": math.radians(5.0),
            "primitive_length": primitive_length,
            "position_tolerance": position_tolerance,
            "yaw_tolerance": yaw_tolerance,
            "reverse_multiplier": 1.65,
            "gear_change_penalty": gear_change_penalty,
            "steering_change_penalty": steering_change_penalty,
        },
    )
    common_kwargs = dict(
        environment=environment,
        vehicle=Vehicle(),
        safety_margin=0.0,
        integration_step=integration_step,
        collision_check_step=collision_check_step,
        heuristic_mode=heuristic_mode,
        state_key_mode=state_key_mode,
        heuristic_weight=heuristic_weight,
    )
    if use_two_queues:
        planner = TwoQueueHybridAStar(
            **common_kwargs,
            coarse_heuristic_weight=coarse_heuristic_weight,
            coarse_primitive_mult=coarse_primitive_mult,
            queue_beta=queue_beta,
            origin_priority_factor=origin_priority_factor,
        )
    else:
        planner = HybridAStar(**common_kwargs)
    planner.goal = goal
    return planner


def test_maze_environment_is_available_with_a_navigable_point_robot_route() -> None:
    environment = make_environment(
        "maze",
        {
            "xy_resolution": 0.25,
            "yaw_resolution": math.radians(5.0),
            "primitive_length": 0.5,
            "position_tolerance": 0.3,
            "yaw_tolerance": math.radians(2.0),
            "reverse_multiplier": 1.65,
            "gear_change_penalty": 0.0,
            "steering_change_penalty": 0.0,
        },
    )

    assert environment.name == "maze"
    assert environment.width == pytest.approx(109.26225)
    assert environment.height == pytest.approx(56.8035)
    assert environment.start == pytest.approx((6.5325, 51.5, math.radians(-90.0)))
    assert environment.goal == pytest.approx((102.68425, 6.0, math.radians(-90.0)))
    assert len(environment.obstacles) == 32
    assert min(obstacle.xmin for obstacle in environment.obstacles) == pytest.approx(0.0)
    assert max(obstacle.xmax for obstacle in environment.obstacles) == pytest.approx(
        environment.width
    )
    assert min(obstacle.ymin for obstacle in environment.obstacles) == pytest.approx(0.0)
    assert max(obstacle.ymax for obstacle in environment.obstacles) == pytest.approx(
        environment.height
    )
    first_wall = environment.obstacles[0]
    assert (first_wall.xmin, first_wall.xmax, first_wall.ymin, first_wall.ymax) == pytest.approx(
        (0.0, 2.5, 0.0, environment.height)
    )
    right_wall = environment.obstacles[5]
    assert (right_wall.ymin, right_wall.ymax) == pytest.approx((0.0, environment.height))
    last_wall = environment.obstacles[-1]
    assert (last_wall.xmin, last_wall.xmax, last_wall.ymin, last_wall.ymax) == pytest.approx(
        (64.13925, 66.63925, 1.25, 12.7255)
    )

    planner = HybridAStar(environment, Vehicle(), 0.2, 0.1, 0.05)
    assert not planner.collides(*environment.start)
    assert not planner.collides(*environment.goal)
    assert planner.collides(-0.1, environment.height / 2.0, 0.0)
    assert planner.collides(environment.width + 0.1, environment.height / 2.0, 0.0)

    from hybrid_astar_corridor import CoarsePathCorridor

    corridor = CoarsePathCorridor(
        environment.width,
        environment.height,
        tuple(
            (obstacle.xmin, obstacle.xmax, obstacle.ymin, obstacle.ymax)
            for obstacle in environment.obstacles
        ),
        coarse_resolution=1.0,
        corridor_width=1.0,
        obstacle_clearance=1.1,
    )
    path = corridor.build(environment.start[:2], environment.goal[:2])

    assert len(path) > 100
    assert corridor.path_length > 100.0


def make_node(
    x: float,
    cost: float,
    *,
    parent: Node | None = None,
    y: float = 5.0,
    yaw: float = 0.0,
    direction: int = 1,
    steer_index: int = 2,
    primitive_length: float | None = None,
) -> Node:
    if primitive_length is None:
        primitive_length = 0.0 if parent is None else 0.5
    return Node(
        x=x,
        y=y,
        yaw=yaw,
        cost=cost,
        parent=parent,
        direction=direction,
        steer_index=steer_index,
        primitive_length=primitive_length,
    )


def chain_from_terminal(terminal: Node) -> list[Node]:
    chain: list[Node] = []
    node: Node | None = terminal
    while node is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()
    return chain


def grouped_open_pushes(
    captured_entries: list[planner_module.OpenEntry],
) -> Iterator[tuple[planner_module.OpenEntry, planner_module.OpenEntry]]:
    by_serial: dict[int, list[planner_module.OpenEntry]] = defaultdict(list)
    for entry in captured_entries:
        by_serial[entry[3]].append(entry)
    for serial, entries in by_serial.items():
        if serial == 0:
            continue
        assert len(entries) == 2
        yield entries[0], entries[1]


def test_node_is_compact_identity_based_and_stores_edge_length() -> None:
    first = make_node(5.0, 0.0)
    second = make_node(5.0, 0.0)

    assert not hasattr(first, "__dict__")
    assert not hasattr(first, "segment")
    assert first.primitive_length == 0.0
    assert first == first
    assert first != second


def test_collision_kernel_remains_numba_jitted_with_disk_cache() -> None:
    assert hasattr(_sample_collision_free_primitive, "py_func")
    assert _sample_collision_free_primitive._cache is not None


def test_empty_collision_sampling_returns_the_input_pose() -> None:
    pose = (4.25, 6.75, -0.4)
    result = _sample_collision_free_primitive(
        *pose,
        1,
        0.2,
        2.6,
        np.empty(0, dtype=float),
        2.2,
        0.9,
        1.2,
        100.0,
        100.0,
        np.empty((0, 4), dtype=float),
    )

    assert result[0] is True
    np.testing.assert_allclose(result[1:], pose, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("queue_name", ["fine", "coarse"])
def test_collision_kernel_endpoint_matches_exact_arc(
    direction: int,
    queue_name: str,
) -> None:
    planner = make_planner(use_two_queues=True, coarse_primitive_mult=4)
    start = make_node(8.0, 0.0, y=9.0, yaw=0.35)
    if queue_name == "fine":
        length = planner.primitive_length
        collision_distances = planner._collision_distances
        steer_indices = planner.steer_indices
    else:
        length = planner.coarse_primitive_length
        collision_distances = planner._coarse_collision_distances
        steer_indices = planner.coarse_steer_indices

    for steer_index in steer_indices:
        endpoint = planner.check_primitive(
            start,
            direction,
            steer_index,
            collision_distances,
        )
        assert endpoint is not None
        expected = arc_pose(
            start.x,
            start.y,
            start.yaw,
            direction * length,
            float(planner.steers[steer_index]),
            planner.vehicle.wheelbase,
        )
        np.testing.assert_allclose(endpoint, expected, atol=1e-12, rtol=0.0)


def test_reconstruct_regenerates_mixed_length_samples_and_controls() -> None:
    planner = make_planner(
        integration_step=0.18,
        use_two_queues=True,
        coarse_primitive_mult=2,
    )
    root = make_node(6.0, 0.0, y=7.0, yaw=0.2, steer_index=2)

    first_steer_index = 4
    first_pose = arc_pose(
        root.x,
        root.y,
        root.yaw,
        planner.primitive_length,
        float(planner.steers[first_steer_index]),
        planner.vehicle.wheelbase,
    )
    first = Node(
        *first_pose,
        planner.primitive_length,
        root,
        1,
        first_steer_index,
        planner.primitive_length,
    )

    second_steer_index = 0
    second_pose = arc_pose(
        first.x,
        first.y,
        first.yaw,
        -planner.coarse_primitive_length,
        float(planner.steers[second_steer_index]),
        planner.vehicle.wheelbase,
    )
    second = Node(
        *second_pose,
        first.cost + planner.coarse_primitive_length * planner.reverse_multiplier,
        first,
        -1,
        second_steer_index,
        planner.coarse_primitive_length,
    )

    path, directions, steers = planner.reconstruct(second)

    expected_parts = [np.asarray([[root.x, root.y, root.yaw]], dtype=float)]
    expected_directions = [root.direction]
    expected_steers = [float(planner.steers[root.steer_index])]
    for parent, child in ((root, first), (first, second)):
        distances = planner._integration_distances_by_length[child.primitive_length]
        steer = float(planner.steers[child.steer_index])
        segment = np.asarray(
            [
                arc_pose(
                    parent.x,
                    parent.y,
                    parent.yaw,
                    child.direction * float(distance),
                    steer,
                    planner.vehicle.wheelbase,
                )
                for distance in distances
            ],
            dtype=float,
        )
        segment[-1] = (child.x, child.y, child.yaw)
        expected_parts.append(segment)
        expected_directions.extend([child.direction] * len(segment))
        expected_steers.extend([steer] * len(segment))

    np.testing.assert_allclose(path, np.concatenate(expected_parts), atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(directions, np.asarray(expected_directions))
    np.testing.assert_allclose(steers, np.asarray(expected_steers))


def test_terminal_remains_reconstructable_after_planner_reuse() -> None:
    planner = make_planner(goal=(8.0, 5.0, 0.0))
    first_path, first_directions, first_steers, first_terminal = planner.plan(
        planner.environment.start,
        (8.0, 5.0, 0.0),
        max_expansions=1_000,
    )

    planner.plan(
        planner.environment.start,
        (9.0, 5.0, 0.0),
        max_expansions=2_000,
    )
    rebuilt_path, rebuilt_directions, rebuilt_steers = planner.reconstruct(first_terminal)

    np.testing.assert_allclose(rebuilt_path, first_path)
    np.testing.assert_array_equal(rebuilt_directions, first_directions)
    np.testing.assert_allclose(rebuilt_steers, first_steers)


def test_real_plan_nodes_remain_compact_and_have_direct_parents() -> None:
    planner = make_planner()
    path, directions, steers, terminal = planner.plan(
        planner.environment.start,
        planner.environment.goal,
        max_expansions=1_000,
    )

    chain = chain_from_terminal(terminal)
    assert chain
    assert chain[0].parent is None
    assert chain[0].primitive_length == 0.0
    assert all(node.parent is chain[index - 1] for index, node in enumerate(chain[1:], start=1))
    assert all(node.primitive_length == planner.primitive_length for node in chain[1:])
    assert all(not hasattr(node, "__dict__") for node in chain)
    assert all(not hasattr(node, "segment") for node in chain)
    assert len(path) == len(directions) == len(steers)
    np.testing.assert_allclose(path[0], planner.environment.start)
    np.testing.assert_allclose(path[-1], (terminal.x, terminal.y, terminal.yaw))


def test_pop_best_open_discards_a_stale_node_by_identity() -> None:
    planner = make_planner()
    stale = make_node(5.01, 10.0)
    current = make_node(5.10, 9.0)
    key = planner._state_key(stale)
    assert key == planner._state_key(current)

    queue = [
        (-100.0, 0.0, -stale.cost, 0, key, stale),
        (100.0, 0.0, -current.cost, 1, key, current),
    ]
    heapq.heapify(queue)

    result = planner._pop_best_open(queue, closed=set(), nodes={key: current})

    assert result is not None
    result_key, result_node = result
    assert result_key == key
    assert result_node is current


def test_live_search_selects_minimum_heuristic_across_open_and_closed_states() -> None:
    planner = make_planner(goal=(8.0, 5.0, 0.0))
    open_node = make_node(5.0, 1.0)
    closed_node = make_node(8.0, 2.0)
    open_key = planner._state_key(open_node)
    closed_key = planner._state_key(closed_node)
    published: list[tuple[planner_module.SearchSnapshot, planner_module.SearchSnapshot]] = []

    def capture(
        _expansion_count: int,
        best_total: planner_module.SearchSnapshot,
        best_heuristic: planner_module.SearchSnapshot,
        _states: tuple[planner_module.SearchNodeState, ...],
    ) -> None:
        published.append((best_total, best_heuristic))

    planner._publish_search_state(
        closed={closed_key},
        nodes={open_key: open_node, closed_key: closed_node},
        progress_callback=capture,
    )

    assert len(published) == 1
    best_total, best_heuristic = published[0]
    assert best_total.node is open_node
    assert best_heuristic.node is closed_node


def test_live_dijkstra_cost_to_goal_view_uses_the_cached_grid() -> None:
    planner = make_planner(heuristic_mode="dijkstra")
    planner.goal = planner.environment.goal
    planner.heuristic(*planner.environment.start)
    assert planner._dijkstra_cost_to_goal is not None
    node = Node(*planner.environment.start, 0.0, None, 1, 2, 0.0)
    snapshot = planner_module.SearchSnapshot(
        node=node,
        path=np.asarray([planner.environment.start]),
        heuristic=planner.heuristic(*planner.environment.start),
        total_estimate=0.0,
    )
    state = planner_module.SearchNodeState(
        node=node,
        heuristic=snapshot.heuristic,
        total_estimate=0.0,
        closed=False,
    )

    plot = demo_module.LiveSearchPlot(planner, planner.environment)
    try:
        plot.update(7, snapshot, snapshot, (state,))
        view = plot.dijkstra_cost_to_goal

        np.testing.assert_allclose(
            view.heatmap.get_array().filled(np.nan),
            planner._dijkstra_cost_to_goal,
            equal_nan=True,
        )
        assert tuple(view.heatmap.get_extent()) == (
            -0.5 * planner.xy_resolution,
            (planner._dijkstra_cost_to_goal.shape[1] - 0.5) * planner.xy_resolution,
            -0.5 * planner.xy_resolution,
            (planner._dijkstra_cost_to_goal.shape[0] - 0.5) * planner.xy_resolution,
        )
        assert len(plot.fig.axes) == 6
        assert "expansion 7" in view.ax.get_title()
    finally:
        demo_module.plt.close(plot.fig)


def test_dijkstra_grid_blocks_obstacle_boundaries() -> None:
    obstacle = Obstacle(7.0, 9.0, 0.0, 10.0)
    planner = make_planner(heuristic_mode="dijkstra", obstacles=(obstacle,))

    costs = planner._build_2d_dijkstra_to_goal_region()
    resolution = planner.xy_resolution

    assert math.isinf(costs[round(0.0 / resolution), round(8.0 / resolution)])
    assert math.isinf(costs[round(5.0 / resolution), round(7.0 / resolution)])
    assert math.isinf(costs[round(10.0 / resolution), round(8.0 / resolution)])


def test_live_plot_omits_dijkstra_view_for_other_heuristics() -> None:
    planner = make_planner(heuristic_mode="distance")

    plot = demo_module.LiveSearchPlot(planner, planner.environment)
    try:
        assert plot.dijkstra_cost_to_goal is None
        assert len(plot.fig.axes) == 4
    finally:
        demo_module.plt.close(plot.fig)


def test_control_aware_state_key_distinguishes_incoming_controls() -> None:
    planner = make_planner(state_key_mode="pose_control")
    left = make_node(5.0, 0.0, direction=1, steer_index=0)
    right = make_node(5.0, 0.0, direction=1, steer_index=len(planner.steers) - 1)
    reverse = make_node(5.0, 0.0, direction=-1, steer_index=0)
    root = make_node(5.0, 0.0)

    assert planner._state_key(left) != planner._state_key(right)
    assert planner._state_key(left) != planner._state_key(reverse)
    assert planner._state_key(root, initial=True) != planner._state_key(root)

    legacy = make_planner(state_key_mode="pose")
    assert legacy._state_key(left) == legacy._state_key(right)
    assert legacy._state_key(left) == legacy._state_key(reverse)


def test_gear_change_penalty_uses_parent_presence_and_selected_edge_length() -> None:
    planner = make_planner(gear_change_penalty=7.0, use_two_queues=True)
    root = make_node(5.0, 0.0, parent=None, direction=1)
    child = make_node(
        5.5,
        planner.primitive_length,
        parent=root,
        direction=1,
        primitive_length=planner.primitive_length,
    )

    root_reverse_cost = planner._successor_cost(
        root,
        -1,
        root.steer_index,
        planner.coarse_primitive_length,
    )
    child_reverse_cost = planner._successor_cost(
        child,
        -1,
        child.steer_index,
        planner.coarse_primitive_length,
    )

    expected_root = planner.coarse_primitive_length * planner.reverse_multiplier
    expected_child = (
        child.cost
        + planner.coarse_primitive_length * planner.reverse_multiplier
        + planner.gear_change_penalty
    )
    assert root_reverse_cost == pytest.approx(expected_root)
    assert child_reverse_cost == pytest.approx(expected_child)


def test_steering_change_is_an_event_cost_for_fine_and_coarse_edges() -> None:
    planner = make_planner(steering_change_penalty=2.5, use_two_queues=True)
    parent = make_node(6.0, 0.0, steer_index=2)
    target_steer_index = len(planner.steers) - 1

    fine_cost = planner._successor_cost(
        parent,
        1,
        target_steer_index,
        planner.primitive_length,
    )
    coarse_cost = planner._successor_cost(
        parent,
        1,
        target_steer_index,
        planner.coarse_primitive_length,
    )

    event_cost = planner.primitive_length * planner.steering_change_penalty
    assert fine_cost == pytest.approx(planner.primitive_length + event_cost)
    assert coarse_cost == pytest.approx(planner.coarse_primitive_length + event_cost)


def test_sample_distances_always_contains_exact_endpoint() -> None:
    assert sample_distances(0.5, 0.2) == pytest.approx([0.2, 0.4, 0.5])
    assert sample_distances(0.5, 0.1)[-1] == pytest.approx(0.5)
    assert sample_distances(0.05, 0.1) == pytest.approx([0.05])


def test_straight_arc_wraps_input_yaw() -> None:
    pose = arc_pose(1.0, 2.0, 4.0 * math.pi, 0.5, 0.0, 2.6)

    assert pose == pytest.approx((1.5, 2.0, 0.0))


def test_cyclic_yaw_key_matches_across_angle_seam() -> None:
    planner = make_planner()
    seam_approach = math.pi - planner.yaw_resolution * 0.1

    assert planner.key(5.0, 5.0, seam_approach) == planner.key(5.0, 5.0, -math.pi)
    assert planner.key(5.0, 5.0, -math.pi)[2] == 0


def test_heuristic_modes_match_their_documented_definitions() -> None:
    goal = (10.0, 5.0, math.pi / 2.0)
    state = (5.0, 5.0, 0.0)

    distance_planner = make_planner(goal, heuristic_mode="distance")
    assert distance_planner.heuristic(*state) == pytest.approx(5.0)

    default_planner = make_planner(goal, heuristic_mode="default")
    assert default_planner.heuristic(*state) == pytest.approx(
        5.0 + 0.5 * default_planner.vehicle.wheelbase * math.pi / 2.0
    )

    defaultw1_planner = make_planner(goal, heuristic_mode="defaultw1")
    assert defaultw1_planner.heuristic(*state) == pytest.approx(
        5.0 + defaultw1_planner.vehicle.wheelbase * math.pi / 2.0
    )

    tolerance_planner = make_planner(goal, heuristic_mode="tolerance")
    radius = tolerance_planner.vehicle.wheelbase / math.tan(tolerance_planner.vehicle.max_steer)
    expected = max(
        5.0 - tolerance_planner.position_tolerance,
        radius * (math.pi / 2.0 - tolerance_planner.yaw_tolerance),
    )
    assert tolerance_planner.heuristic(*state) == pytest.approx(expected)
    assert tolerance_planner.heuristic(*goal) == pytest.approx(0.0)


def test_two_queue_configuration_defaults_and_validation() -> None:
    planner = make_planner(use_two_queues=True)

    assert planner.coarse_primitive_length == pytest.approx(4 * planner.primitive_length)
    assert planner.coarse_heuristic_weight == planner.heuristic_weight
    assert planner.coarse_steer_indices == (0, len(planner.steers) // 2, len(planner.steers) - 1)
    assert planner.steer_indices == tuple(range(len(planner.steers)))

    with pytest.raises(ValueError, match="coarse_primitive_mult"):
        make_planner(use_two_queues=True, coarse_primitive_mult=0)
    with pytest.raises(ValueError, match="coarse_primitive_mult"):
        make_planner(use_two_queues=True, coarse_primitive_mult=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="queue_beta"):
        make_planner(use_two_queues=True, queue_beta=0.99)
    with pytest.raises(ValueError, match="coarse_heuristic_weight"):
        make_planner(use_two_queues=True, coarse_heuristic_weight=-0.1)
    with pytest.raises(ValueError, match="origin_priority_factor"):
        make_planner(use_two_queues=True, origin_priority_factor=0.99)


def test_coarse_heuristic_weight_only_changes_coarse_queue_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine_weight = 1.0
    coarse_weight = 3.0
    heuristic = 2.0
    planner = make_planner(
        goal=(25.0, 15.0, 0.0),
        heuristic_weight=fine_weight,
        coarse_heuristic_weight=coarse_weight,
        use_two_queues=True,
        origin_priority_factor=1.0,
    )
    monkeypatch.setattr(planner, "heuristic", lambda *_: heuristic)
    captured: list[planner_module.OpenEntry] = []
    original_heappush = heapq.heappush

    def recording_heappush(heap: list[planner_module.OpenEntry], item: planner_module.OpenEntry) -> None:
        captured.append(item)
        original_heappush(heap, item)

    monkeypatch.setattr(two_queue_module.heapq, "heappush", recording_heappush)

    with pytest.raises(RuntimeError, match="No path found"):
        planner.plan(
            planner.environment.start,
            planner.environment.goal,
            max_expansions=1,
            max_consecutive_coarse_expansions=0,
        )

    for fine_entry, coarse_entry in grouped_open_pushes(captured):
        cost = fine_entry[0] - fine_weight * heuristic
        assert coarse_entry[0] == pytest.approx(cost + coarse_weight * heuristic)


def test_two_queue_plan_can_reach_goal_with_one_coarse_edge() -> None:
    goal = (7.0, 5.0, 0.0)
    planner = make_planner(
        goal,
        use_two_queues=True,
        coarse_primitive_mult=4,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    path, directions, steers, terminal = planner.plan(
        planner.environment.start,
        goal,
        max_expansions=20,
    )

    chain = chain_from_terminal(terminal)
    assert planner.coarse_expansion_count >= 2
    assert planner.fine_expansion_count == 0
    assert [node.primitive_length for node in chain] == [0.0, planner.coarse_primitive_length]
    np.testing.assert_allclose(path[-1], goal, atol=1e-12, rtol=0.0)
    assert np.all(directions == 1)
    assert np.allclose(steers, 0.0)


def test_no_path_errors_report_queue_breakdown_without_base_queue_state() -> None:
    base = make_planner(goal=(25.0, 15.0, 0.0), use_two_queues=False)
    with pytest.raises(
        RuntimeError,
        match=r"\(1 fine, 0 coarse; two_queue=False\)",
    ):
        base.plan(base.environment.start, base.environment.goal, max_expansions=1)

    derived = make_planner(goal=(25.0, 15.0, 0.0), use_two_queues=True)
    with pytest.raises(
        RuntimeError,
        match=r"\([01] fine, [01] coarse; two_queue=True\)",
    ):
        derived.plan(
            derived.environment.start,
            derived.environment.goal,
            max_expansions=1,
        )


def test_base_plan_uses_only_standard_primitives_and_has_no_queue_specific_state() -> None:
    goal = (7.0, 5.0, 0.0)
    planner = make_planner(
        goal,
        use_two_queues=False,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    _, _, _, terminal = planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
    )

    chain = chain_from_terminal(terminal)
    assert all(node.primitive_length == planner.primitive_length for node in chain[1:])
    assert not any(name.startswith("fine_") for name in planner.__dict__)
    assert not any(name.startswith("coarse_") for name in planner.__dict__)
    assert "use_two_queues" not in planner.__dict__
    assert "last_expansion_queue" not in planner.__dict__


def test_separate_closed_sets_allow_same_start_state_to_receive_both_action_sets() -> None:
    planner = make_planner(
        goal=(25.0, 15.0, 0.0),
        use_two_queues=True,
        coarse_primitive_mult=4,
    )

    with pytest.raises(RuntimeError, match="No path found"):
        planner.plan(
            planner.environment.start,
            planner.environment.goal,
            max_expansions=2,
            max_consecutive_coarse_expansions=1,
        )

    assert planner.coarse_expansion_count == 1
    assert planner.fine_expansion_count == 1
    assert planner.unique_expanded_state_count == 1


def test_zero_coarse_streak_limit_forces_fine_when_fine_state_exists() -> None:
    planner = make_planner(
        goal=(25.0, 15.0, 0.0),
        use_two_queues=True,
        coarse_primitive_mult=4,
    )

    with pytest.raises(RuntimeError, match="No path found"):
        planner.plan(
            planner.environment.start,
            planner.environment.goal,
            max_expansions=1,
            max_consecutive_coarse_expansions=0,
        )

    assert planner.fine_expansion_count == 1
    assert planner.coarse_expansion_count == 0
    assert planner.last_expansion_queue == "fine"


@pytest.mark.parametrize(
    ("max_coarse_streak", "expected_relation"),
    [
        (0, "fine_generated"),
        (10, "coarse_generated"),
    ],
)
def test_origin_priority_factor_is_a_one_sided_coarse_queue_bias(
    monkeypatch: pytest.MonkeyPatch,
    max_coarse_streak: int,
    expected_relation: str,
) -> None:
    factor = 3.0
    planner = make_planner(
        goal=(25.0, 15.0, 0.0),
        use_two_queues=True,
        coarse_primitive_mult=4,
        origin_priority_factor=factor,
    )
    captured: list[planner_module.OpenEntry] = []
    original_heappush = heapq.heappush

    def recording_heappush(heap: list[planner_module.OpenEntry], item: planner_module.OpenEntry) -> None:
        captured.append(item)
        original_heappush(heap, item)

    monkeypatch.setattr(two_queue_module.heapq, "heappush", recording_heappush)

    with pytest.raises(RuntimeError, match="No path found"):
        planner.plan(
            planner.environment.start,
            planner.environment.goal,
            max_expansions=1,
            max_consecutive_coarse_expansions=max_coarse_streak,
        )

    pairs = list(grouped_open_pushes(captured))
    assert pairs
    for fine_entry, coarse_entry in pairs:
        fine_priority = fine_entry[0]
        coarse_priority = coarse_entry[0]
        if expected_relation == "fine_generated":
            assert coarse_priority == pytest.approx(fine_priority * factor)
        else:
            assert coarse_priority == pytest.approx(fine_priority)


def test_cost_only_plan_api_has_no_terminal_selection_parameter() -> None:
    signature = inspect.signature(HybridAStar.plan)
    two_queue_signature = inspect.signature(TwoQueueHybridAStar.plan)

    assert "terminal_selection" not in signature.parameters
    assert "enable_admissible_bound" not in signature.parameters
    assert "enable_admissible_bound" not in two_queue_signature.parameters
    assert "max_consecutive_coarse_expansions" not in signature.parameters
    assert not hasattr(planner_module, "TerminalSelection")
    assert not hasattr(HybridAStar, "_terminal_score")


def test_exact_path_length_sums_stored_primitive_lengths() -> None:
    planner = make_planner(use_two_queues=True, coarse_primitive_mult=3)
    root = make_node(6.0, 0.0, y=7.0, yaw=0.2)
    first = make_node(
        6.5,
        planner.primitive_length,
        parent=root,
        primitive_length=planner.primitive_length,
    )
    terminal = make_node(
        8.0,
        first.cost + planner.coarse_primitive_length,
        parent=first,
        primitive_length=planner.coarse_primitive_length,
    )

    assert planner.exact_path_length(terminal) == pytest.approx(
        planner.primitive_length + planner.coarse_primitive_length
    )


def test_post_goal_budget_is_checked_at_next_loop_top() -> None:
    goal = (5.5, 5.0, 0.0)
    planner = make_planner(
        goal,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
        post_goal_expansions=1,
    )

    assert planner.expansion_count == 3


def test_zero_post_goal_budget_does_not_expand_accepted_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal = (5.5, 5.0, 0.0)
    planner = make_planner(
        goal,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )
    goal_key = planner.key(*goal)
    original_check_primitive = planner.check_primitive

    def reject_goal_expansion(
        node: Node,
        direction: int,
        steer_index: int,
        collision_distances: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if planner.key(node.x, node.y, node.yaw) == goal_key:
            pytest.fail("an accepted cost terminal must not be expanded")
        return original_check_primitive(node, direction, steer_index, collision_distances)

    monkeypatch.setattr(planner, "check_primitive", reject_goal_expansion)

    planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
        post_goal_expansions=0,
    )

    assert planner.expansion_count == 2


def test_two_queue_search_uses_post_goal_budget() -> None:
    goal = (7.0, 5.0, 0.0)
    planner = make_planner(
        goal,
        use_two_queues=True,
        coarse_primitive_mult=4,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    planner.plan(
        planner.environment.start,
        goal,
        max_expansions=20,
        post_goal_expansions=1,
    )

    assert planner.expansion_count == 3


def test_open_exhaustion_does_not_claim_max_expansions_was_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    goal = (5.0, 5.0, 0.0)
    planner = make_planner(
        goal,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
        post_goal_expansions=10,
    )
    output = capsys.readouterr().out

    assert planner.expansion_count == 1
    assert "Warning: the search limit was reached" not in output


def test_post_goal_budget_completion_uses_expansion_count() -> None:
    planner = make_planner()
    planner.expansion_count = 12

    assert planner._post_goal_budget_complete(10, 2)
    assert not planner._post_goal_budget_complete(10, 3)
    assert not planner._post_goal_budget_complete(None, 0)


def test_finish_search_runs_callbacks_then_reports_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = make_planner()
    terminal = make_node(8.0, 3.0)
    key = planner._state_key(terminal)
    events: list[object] = []
    expected_path = np.asarray([[5.0, 5.0, 0.0], [8.0, 5.0, 0.0]])
    expected_directions = np.asarray([1, 1])
    expected_steers = np.asarray([0.0, 0.0])

    def fake_publish(*_args: object) -> None:
        events.append("publish")

    def fake_report(*_args: object) -> None:
        events.append("report")

    def fake_reconstruct(_terminal: Node) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        events.append("reconstruct")
        return expected_path, expected_directions, expected_steers

    monkeypatch.setattr(planner, "_publish_search_state", fake_publish)
    monkeypatch.setattr(planner, "report_goal", fake_report)
    monkeypatch.setattr(planner, "reconstruct", fake_reconstruct)
    planner.expansion_count = 17

    result = planner._finish_search(
        terminal,
        11,
        planner.environment.goal,
        set(),
        {key: terminal},
        1,
        lambda *_: events.append("progress_callback"),
        lambda count: events.append(("expansion_callback", count)),
    )

    assert events == [
        ("expansion_callback", 17),
        "publish",
        "report",
        "reconstruct",
    ]
    np.testing.assert_array_equal(result[0], expected_path)
    np.testing.assert_array_equal(result[1], expected_directions)
    np.testing.assert_array_equal(result[2], expected_steers)
    assert result[3] is terminal


def test_production_function_docstrings_use_structured_sections() -> None:
    source_paths = (
        Path(planner_module.__file__),
        Path(two_queue_module.__file__),
        Path(demo_module.__file__),
    )

    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            docstring = ast.get_docstring(function)
            assert docstring is not None, (source_path.name, function.name)

            parameters = [
                argument.arg
                for argument in (function.args.posonlyargs + function.args.args + function.args.kwonlyargs)
                if argument.arg not in {"self", "cls"}
            ]
            if parameters:
                assert "Args:" in docstring, (source_path.name, function.name)
            if function.returns is not None:
                assert "Returns:" in docstring, (source_path.name, function.name)

            stack = list(function.body)
            has_explicit_raise = False
            while stack:
                node = stack.pop()
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                ):
                    continue
                if isinstance(node, ast.Raise):
                    has_explicit_raise = True
                stack.extend(ast.iter_child_nodes(node))
            if has_explicit_raise:
                assert "Raises:" in docstring, (source_path.name, function.name)


def test_two_queue_instance_adds_only_queue_specific_state() -> None:
    base = make_planner(use_two_queues=False)
    derived = make_planner(use_two_queues=True)

    obsolete_names = {
        "use_two_queues",
        "fine_primitive_length",
        "fine_steer_indices",
        "_fine_collision_distances",
    }
    assert obsolete_names.isdisjoint(base.__dict__)
    assert obsolete_names.isdisjoint(derived.__dict__)

    derived_only = set(derived.__dict__) - set(base.__dict__)
    assert derived_only == {
        "coarse_heuristic_weight",
        "queue_beta",
        "origin_priority_factor",
        "coarse_primitive_length",
        "coarse_steer_indices",
        "_coarse_collision_distances",
        "fine_expansion_count",
        "coarse_expansion_count",
        "last_expansion_queue",
    }

    # OPEN queues and CLOSED sets are per-search locals rather than persistent
    # planner state. Only the reporting counters survive after a search.
    for transient_name in (
        "fine_queue",
        "coarse_queue",
        "fine_closed",
        "coarse_closed",
    ):
        assert transient_name not in derived.__dict__

    assert derived.primitive_length == base.primitive_length
    assert derived.steer_indices == base.steer_indices
    np.testing.assert_array_equal(
        derived._collision_distances,
        base._collision_distances,
    )


@pytest.mark.parametrize(
    ("two_queue", "expected_name"),
    [
        (False, "fine"),
        (True, "two"),
    ],
)
def test_main_selects_planner_class_from_two_queue_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    two_queue: bool,
    expected_name: str,
) -> None:
    import sys
    import hybrid_astar_main

    selected: list[str] = []

    class FakeProgress:
        n = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, amount: int) -> None:
            self.n += amount

        def close(self) -> None:
            pass

    class FakePlanner:
        planner_name = "fine"

        def __init__(self, **kwargs) -> None:
            selected.append(self.planner_name)
            self.environment = kwargs["environment"]
            self.vehicle = kwargs["vehicle"]
            self.safety_margin = kwargs["safety_margin"]
            self.integration_step = kwargs["integration_step"]
            self.collision_check_step = kwargs["collision_check_step"]
            self.heuristic_mode = kwargs["heuristic_mode"]
            self.state_key_mode = kwargs["state_key_mode"]
            self.heuristic_weight = kwargs["heuristic_weight"]
            self.primitive_length = self.environment.planner["primitive_length"]
            self.expansion_count = 1
            self.unique_expanded_state_count = 1

        def plan(self, start, goal, *_args, **_kwargs):
            root = Node(*start, 0.0, None, 1, 2, 0.0)
            terminal = Node(*goal, 1.0, root, 1, 2, 1.0)
            path = np.asarray([start, goal], dtype=float)
            return path, np.asarray([1, 1]), np.asarray([0.0, 0.0]), terminal

        @staticmethod
        def exact_path_length(_terminal: Node) -> float:
            return 1.0

    class FakeTwoQueuePlanner(FakePlanner):
        planner_name = "two"

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.coarse_heuristic_weight = kwargs.get("coarse_heuristic_weight", self.heuristic_weight)
            if self.coarse_heuristic_weight is None:
                self.coarse_heuristic_weight = self.heuristic_weight
            self.coarse_primitive_length = kwargs.get("coarse_primitive_mult", 4) * self.primitive_length
            self.queue_beta = kwargs.get("queue_beta", 1.5)
            self.origin_priority_factor = kwargs.get("origin_priority_factor", 2.0)
            self.fine_expansion_count = 1
            self.coarse_expansion_count = 0

    monkeypatch.setattr(hybrid_astar_main, "HybridAStar", FakePlanner)
    monkeypatch.setattr(hybrid_astar_main, "TwoQueueHybridAStar", FakeTwoQueuePlanner)
    monkeypatch.setattr(hybrid_astar_main, "tqdm", FakeProgress)
    monkeypatch.setattr(hybrid_astar_main, "save_plot", lambda *_args: None)
    argv = [
        "hybrid_astar_main.py",
        "--no_animation_plot",
        "--live_plot_every",
        "0",
        "--output_dir",
        str(tmp_path),
    ]
    if two_queue:
        argv.append("--two_queues")
    monkeypatch.setattr(sys, "argv", argv)

    result = hybrid_astar_main.main(hybrid_astar_main.parse_args())

    assert selected == [expected_name]
    assert result["coarse_expansions"] == 0


def test_parse_args_treats_primitive_length_as_a_single_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import hybrid_astar_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_astar_main.py",
            "--primitive_length",
            "0.8",
        ],
    )

    args = hybrid_astar_main.parse_args()

    assert args.primitive_length == 0.8
    assert not hasattr(args, "enable_admissible_bound")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], "mp4"),
        (["--animation_format", "gif"], "gif"),
    ],
)
def test_parse_args_selects_animation_format(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: str,
) -> None:
    import sys
    import hybrid_astar_main

    monkeypatch.setattr(sys, "argv", ["hybrid_astar_main.py", *arguments])

    assert hybrid_astar_main.parse_args().animation_format == expected


def test_parse_args_shows_final_animation_window_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import hybrid_astar_main

    monkeypatch.setattr(sys, "argv", ["hybrid_astar_main.py"])

    args = hybrid_astar_main.parse_args()

    assert not args.no_animation_plot
    assert not args.save_video


def test_parse_args_disables_final_animation_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import hybrid_astar_main

    monkeypatch.setattr(
        sys,
        "argv",
        ["hybrid_astar_main.py", "--no_animation_plot"],
    )

    args = hybrid_astar_main.parse_args()

    assert args.no_animation_plot
    assert not args.save_video


def test_parse_args_controls_video_saving_and_playback_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import hybrid_astar_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_astar_main.py",
            "--save_video",
            "--no_animation_plot",
        ],
    )

    args = hybrid_astar_main.parse_args()

    assert args.no_animation_plot
    assert args.save_video


@pytest.mark.parametrize(
    ("suffix", "use_nvenc", "expected_encoder", "expected_codec"),
    [
        (".mp4", True, "h264_nvenc", "h264_nvenc"),
        (".mp4", False, "libx264", "libx264"),
        (".gif", None, "pillow", None),
    ],
)
def test_animation_writer_selects_requested_encoder(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    use_nvenc: bool | None,
    expected_encoder: str,
    expected_codec: str | None,
) -> None:
    import hybrid_astar_main

    monkeypatch.setattr(
        hybrid_astar_main.FFMpegWriter,
        "isAvailable",
        classmethod(lambda _cls: True),
    )

    writer, encoder = hybrid_astar_main.animation_writer(suffix, use_nvenc=use_nvenc)

    assert encoder == expected_encoder
    if expected_codec is None:
        assert isinstance(writer, hybrid_astar_main.PillowWriter)
    else:
        assert writer.codec == expected_codec


@pytest.mark.parametrize("show", [True, False])
def test_render_animation_only_shows_failed_save_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    show: bool,
) -> None:
    import hybrid_astar_main

    class FailingAnimation:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def save(*_args, **_kwargs) -> None:
            raise RuntimeError("encoder failed")

    planner = make_planner()
    environment = planner.environment
    path = np.asarray([environment.start, environment.goal], dtype=float)
    show_calls = []
    monkeypatch.setattr(hybrid_astar_main, "FuncAnimation", FailingAnimation)
    monkeypatch.setattr(
        hybrid_astar_main,
        "animation_writer",
        lambda _suffix: (object(), "libx264"),
    )
    monkeypatch.setattr(
        hybrid_astar_main.plt,
        "show",
        lambda **kwargs: show_calls.append(kwargs),
    )

    with pytest.warns(RuntimeWarning, match="Saving the animation failed"):
        hybrid_astar_main.render_animation(
            tmp_path / "animation.mp4",
            planner,
            environment,
            path,
            np.asarray([1, 1]),
            np.asarray([0.0, 0.0]),
            save_video=True,
            show=show,
        )

    assert show_calls == ([{"block": True}] if show else [])


def test_render_animation_does_not_save_video_unless_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hybrid_astar_main

    class PlaybackOnlyAnimation:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def save(*_args, **_kwargs) -> None:
            pytest.fail("animation.save should not be called")

    planner = make_planner()
    environment = planner.environment
    path = np.asarray([environment.start, environment.goal], dtype=float)
    show_calls = []
    monkeypatch.setattr(hybrid_astar_main, "FuncAnimation", PlaybackOnlyAnimation)
    monkeypatch.setattr(
        hybrid_astar_main.plt,
        "show",
        lambda **kwargs: show_calls.append(kwargs),
    )

    hybrid_astar_main.render_animation(
        tmp_path / "animation.mp4",
        planner,
        environment,
        path,
        np.asarray([1, 1]),
        np.asarray([0.0, 0.0]),
        save_video=False,
        show=True,
    )

    assert show_calls == [{"block": True}]


def test_nvenc_available_requires_successful_encoder_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hybrid_astar_main

    class ProbeResult:
        returncode = 0

    monkeypatch.setattr(
        hybrid_astar_main.FFMpegWriter,
        "isAvailable",
        classmethod(lambda _cls: True),
    )
    monkeypatch.setattr(
        hybrid_astar_main.FFMpegWriter,
        "bin_path",
        classmethod(lambda _cls: "/test/ffmpeg"),
    )
    monkeypatch.setattr(
        hybrid_astar_main.subprocess,
        "run",
        lambda *_args, **_kwargs: ProbeResult(),
    )
    hybrid_astar_main.nvenc_available.cache_clear()

    try:
        assert hybrid_astar_main.nvenc_available()
    finally:
        hybrid_astar_main.nvenc_available.cache_clear()
