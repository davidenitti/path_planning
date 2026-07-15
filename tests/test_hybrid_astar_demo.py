import heapq
import inspect
import math
from collections import defaultdict
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import hybrid_astar_demo as planner_module
from hybrid_astar_demo import (
    Environment,
    HybridAStar,
    Node,
    Obstacle,
    Vehicle,
    _sample_collision_free_primitive,
    arc_pose,
    sample_distances,
)


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
    planner = HybridAStar(
        environment=environment,
        vehicle=Vehicle(),
        safety_margin=0.0,
        integration_step=integration_step,
        collision_check_step=collision_check_step,
        heuristic_mode=heuristic_mode,
        state_key_mode=state_key_mode,
        heuristic_weight=heuristic_weight,
        use_two_queues=use_two_queues,
        coarse_heuristic_weight=coarse_heuristic_weight,
        coarse_primitive_mult=coarse_primitive_mult,
        queue_beta=queue_beta,
        origin_priority_factor=origin_priority_factor,
    )
    planner.goal = goal
    return planner


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
        length = planner.fine_primitive_length
        collision_distances = planner._fine_collision_distances
        steer_indices = planner.fine_steer_indices
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
        planner.fine_primitive_length,
        float(planner.steers[first_steer_index]),
        planner.vehicle.wheelbase,
    )
    first = Node(
        *first_pose,
        planner.fine_primitive_length,
        root,
        1,
        first_steer_index,
        planner.fine_primitive_length,
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
    assert all(node.primitive_length == planner.fine_primitive_length for node in chain[1:])
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
        fine_closed={closed_key},
        coarse_closed=set(),
        nodes={open_key: open_node, closed_key: closed_node},
        progress_callback=capture,
    )

    assert len(published) == 1
    best_total, best_heuristic = published[0]
    assert best_total.node is open_node
    assert best_heuristic.node is closed_node


def test_admissible_a_star_bound_ends_search_without_post_goal_expansions() -> None:
    goal = (5.5, 5.0, 0.0)
    planner = make_planner(
        goal,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )
    _, _, _, terminal = planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
        post_goal_expansions=50,
        enable_admissible_bound=True,
    )

    assert terminal.cost == pytest.approx(planner.fine_primitive_length)
    assert planner.expansion_count == 2


def test_weighted_a_star_uses_the_unweighted_f_queue_for_its_goal_bound() -> None:
    goal = (5.5, 5.0, 0.0)
    planner = make_planner(
        goal,
        heuristic_weight=1.5,
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )
    planner.plan(
        planner.environment.start,
        goal,
        max_expansions=100,
        post_goal_expansions=50,
        enable_admissible_bound=True,
    )

    assert planner.expansion_count == 2


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
        planner.fine_primitive_length,
        parent=root,
        direction=1,
        primitive_length=planner.fine_primitive_length,
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
        planner.fine_primitive_length,
    )
    coarse_cost = planner._successor_cost(
        parent,
        1,
        target_steer_index,
        planner.coarse_primitive_length,
    )

    event_cost = planner.fine_primitive_length * planner.steering_change_penalty
    assert fine_cost == pytest.approx(planner.fine_primitive_length + event_cost)
    assert coarse_cost == pytest.approx(planner.coarse_primitive_length + event_cost)


def test_sample_distances_always_contains_exact_endpoint() -> None:
    assert sample_distances(0.5, 0.2) == pytest.approx([0.2, 0.4, 0.5])
    assert sample_distances(0.5, 0.1)[-1] == pytest.approx(0.5)
    assert sample_distances(0.05, 0.1) == pytest.approx([0.05])


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

    assert planner.coarse_primitive_length == pytest.approx(4 * planner.fine_primitive_length)
    assert planner.coarse_heuristic_weight == planner.heuristic_weight
    assert planner.coarse_steer_indices == (0, len(planner.steers) // 2, len(planner.steers) - 1)
    assert planner.fine_steer_indices == tuple(range(len(planner.steers)))

    with pytest.raises(ValueError, match="coarse_primitive_mult"):
        make_planner(use_two_queues=True, coarse_primitive_mult=0)
    with pytest.raises(AssertionError, match="coarse_primitive_mult"):
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

    monkeypatch.setattr(planner_module.heapq, "heappush", recording_heappush)

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


def test_fine_only_plan_never_records_coarse_expansions_or_edges() -> None:
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
    assert planner.coarse_expansion_count == 0
    assert planner.fine_expansion_count == planner.expansion_count
    assert all(node.primitive_length == planner.fine_primitive_length for node in chain[1:])


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

    monkeypatch.setattr(planner_module.heapq, "heappush", recording_heappush)

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

    assert "terminal_selection" not in signature.parameters
    assert not hasattr(planner_module, "TerminalSelection")
    assert not hasattr(HybridAStar, "_terminal_score")


def test_exact_path_length_sums_stored_primitive_lengths() -> None:
    planner = make_planner(use_two_queues=True, coarse_primitive_mult=3)
    root = make_node(6.0, 0.0, y=7.0, yaw=0.2)
    first = make_node(
        6.5,
        planner.fine_primitive_length,
        parent=root,
        primitive_length=planner.fine_primitive_length,
    )
    terminal = make_node(
        8.0,
        first.cost + planner.coarse_primitive_length,
        parent=first,
        primitive_length=planner.coarse_primitive_length,
    )

    assert planner.exact_path_length(terminal) == pytest.approx(
        planner.fine_primitive_length + planner.coarse_primitive_length
    )


def test_enabled_bound_can_be_satisfied_on_final_allowed_expansion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    goal = (5.5, 5.0, 0.0)
    planner = make_planner(
        goal,
        heuristic_mode="tolerance",
        position_tolerance=0.01,
        yaw_tolerance=0.01,
    )

    _, _, _, terminal = planner.plan(
        planner.environment.start,
        goal,
        max_expansions=2,
        post_goal_expansions=100,
        enable_admissible_bound=True,
    )
    output = capsys.readouterr().out

    assert terminal.cost == pytest.approx(planner.fine_primitive_length)
    assert planner.expansion_count == 2
    assert "Bound condition satisfied at expansion 2" in output
    assert "Warning:" not in output


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
        enable_admissible_bound=False,
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


def test_two_queue_search_uses_post_goal_budget_even_if_bound_requested() -> None:
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
        enable_admissible_bound=True,
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


def test_inadmissible_order_uses_separate_tolerance_lower_bound_heap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = make_planner(
        goal=(25.0, 15.0, 0.0),
        heuristic_mode="default",
        heuristic_weight=2.0,
    )
    configured_h = 7.0
    admissible_h = 3.0
    monkeypatch.setattr(planner, "heuristic", lambda *_: configured_h)
    monkeypatch.setattr(planner, "_tolerance_aware_lower_bound", lambda *_: admissible_h)
    captured: list[planner_module.OpenEntry] = []
    original_heappush = heapq.heappush

    def recording_heappush(
        heap: list[planner_module.OpenEntry],
        item: planner_module.OpenEntry,
    ) -> None:
        captured.append(item)
        original_heappush(heap, item)

    monkeypatch.setattr(planner_module.heapq, "heappush", recording_heappush)

    with pytest.raises(RuntimeError, match="No path found"):
        planner.plan(
            planner.environment.start,
            planner.environment.goal,
            max_expansions=1,
            enable_admissible_bound=True,
        )

    by_serial: dict[int, list[planner_module.OpenEntry]] = defaultdict(list)
    for entry in captured:
        by_serial[entry[3]].append(entry)

    successor_groups = [entries for serial, entries in by_serial.items() if serial != 0]
    assert successor_groups
    for entries in successor_groups:
        assert len(entries) == 2
        priorities = sorted(entry[0] for entry in entries)
        node_cost = entries[0][5].cost
        assert priorities == pytest.approx(
            sorted(
                [
                    node_cost + admissible_h,
                    node_cost + planner.heuristic_weight * configured_h,
                ]
            )
        )


def test_stop_condition_uses_separate_admissible_lower_bound_heap() -> None:
    planner = make_planner()
    best_terminal = make_node(8.0, 10.0)
    frontier = make_node(7.0, 4.0)
    frontier_key = planner._state_key(frontier)
    nodes = {frontier_key: frontier}

    lower_bound_queue: list[planner_module.OpenEntry] = [
        (9.0, 5.0, -frontier.cost, 1, frontier_key, frontier)
    ]
    heapq.heapify(lower_bound_queue)

    assert not planner.stop_condition(
        best_terminal,
        1,
        True,
        False,
        [],
        lower_bound_queue,
        set(),
        nodes,
        0,
    )

    lower_bound_queue[0] = (10.0, 6.0, -frontier.cost, 1, frontier_key, frontier)
    heapq.heapify(lower_bound_queue)
    assert planner.stop_condition(
        best_terminal,
        1,
        True,
        False,
        [],
        lower_bound_queue,
        set(),
        nodes,
        0,
    )


def test_stop_condition_uses_post_goal_expansion_budget() -> None:
    planner = make_planner()
    best_terminal = make_node(8.0, 10.0)
    planner.expansion_count = 12

    assert planner.stop_condition(
        best_terminal,
        10,
        False,
        False,
        [],
        [],
        set(),
        {},
        2,
    )
    assert not planner.stop_condition(
        best_terminal,
        10,
        False,
        False,
        [],
        [],
        set(),
        {},
        3,
    )


def test_finish_search_runs_callbacks_then_reports_and_reconstructs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        set(),
        {key: terminal},
        1,
        lambda *_: events.append("progress_callback"),
        lambda count: events.append(("expansion_callback", count)),
        announce_bound=True,
    )
    output = capsys.readouterr().out

    assert events == [
        ("expansion_callback", 17),
        "publish",
        "report",
        "reconstruct",
    ]
    assert "Bound condition satisfied at expansion 17" in output
    np.testing.assert_array_equal(result[0], expected_path)
    np.testing.assert_array_equal(result[1], expected_directions)
    np.testing.assert_array_equal(result[2], expected_steers)
    assert result[3] is terminal
