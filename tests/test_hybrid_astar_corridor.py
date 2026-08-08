import math

import numpy as np
import pytest

from hybrid_astar_corridor import CoarsePathCorridor, _segment_avoids_boxes
from hybrid_astar_planner import (
    Environment,
    HybridAStar,
    Node,
    SearchNodeState,
    SearchSnapshot,
    Vehicle,
)


def make_environment(
    *,
    start: tuple[float, float, float] = (5.0, 5.0, 0.0),
    goal: tuple[float, float, float] = (7.0, 5.0, 0.0),
) -> Environment:
    return Environment(
        name="corridor-test",
        title="Corridor test",
        width=12.0,
        height=10.0,
        obstacles=(),
        start=start,
        goal=goal,
        planner={
            "xy_resolution": 0.5,
            "yaw_resolution": math.radians(10.0),
            "primitive_length": 0.5,
            "position_tolerance": 0.05,
            "yaw_tolerance": math.radians(2.0),
            "reverse_multiplier": 1.0,
            "gear_change_penalty": 0.0,
            "steering_change_penalty": 0.0,
        },
    )


def test_coarse_astar_routes_around_point_occupancy() -> None:
    corridor = CoarsePathCorridor(
        6.0,
        6.0,
        ((2.0, 4.0, 0.0, 4.0),),
        coarse_resolution=1.0,
        corridor_width=1.0,
    )

    path = corridor.build((1.0, 1.0), (5.0, 1.0))

    assert np.max(path[:, 1]) > 4.0
    for x, y in path[1:-1]:
        assert not (2.0 <= x <= 4.0 and 0.0 <= y <= 4.0)


def test_segment_box_kernel_is_numba_jitted_and_treats_touching_as_collision() -> None:
    boxes = np.asarray(((2.0, 3.0, 2.0, 3.0),), dtype=float)

    assert hasattr(_segment_avoids_boxes, "py_func")
    assert _segment_avoids_boxes._cache is not None
    assert _segment_avoids_boxes(0.0, 2.0, 2.0, 2.0, boxes) is False
    assert _segment_avoids_boxes(0.0, 1.999, 4.0, 1.999, boxes) is True


def test_coarse_occupancy_inflates_obstacles_by_clearance() -> None:
    corridor = CoarsePathCorridor(
        5.0,
        5.0,
        ((2.0, 3.0, 2.0, 3.0),),
        coarse_resolution=0.5,
        corridor_width=1.0,
        obstacle_clearance=0.5,
    )

    assert corridor.occupancy[round(2.0 / 0.5), round(1.5 / 0.5)]
    assert not corridor.occupancy[round(2.0 / 0.5), round(1.0 / 0.5)]


def test_coarse_astar_does_not_cut_diagonally_between_blocked_cells() -> None:
    corridor = CoarsePathCorridor(
        3.0,
        3.0,
        ((2.0, 2.1, 0.9, 1.1), (0.9, 1.1, 2.0, 2.1)),
        coarse_resolution=1.0,
        corridor_width=0.5,
    )

    with pytest.raises(RuntimeError, match=r"Coarse 2D A\* found no path"):
        corridor.build((1.0, 1.0), (2.0, 2.0))


def test_coarse_astar_does_not_cross_obstacle_between_free_vertices() -> None:
    corridor = CoarsePathCorridor(
        4.0,
        4.0,
        ((1.4, 1.6, 0.0, 4.0),),
        coarse_resolution=1.0,
        corridor_width=0.5,
    )

    with pytest.raises(RuntimeError, match=r"Coarse 2D A\* found no path"):
        corridor.build((1.0, 2.0), (2.0, 2.0))


def test_corridor_uses_radius_distance_from_polyline() -> None:
    corridor = CoarsePathCorridor(6.0, 5.0, (), 1.0, 0.5)
    corridor.build((1.0, 2.0), (5.0, 2.0))

    assert corridor.contains(2.0, 2.5)
    assert not corridor.contains(2.0, 2.5001)
    assert corridor.path_length == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("start", "goal", "endpoint_name"),
    [
        ((3.5, 5.0), (8.0, 5.0), "start"),
        ((2.0, 5.0), (6.5, 5.0), "goal"),
    ],
)
def test_exact_endpoints_must_avoid_inflated_obstacles(
    start: tuple[float, float],
    goal: tuple[float, float],
    endpoint_name: str,
) -> None:
    corridor = CoarsePathCorridor(
        10.0,
        10.0,
        ((4.0, 6.0, 4.0, 6.0),),
        coarse_resolution=1.0,
        corridor_width=1.0,
        obstacle_clearance=1.0,
    )

    with pytest.raises(ValueError, match=rf"{endpoint_name} violates.*clearance"):
        corridor.build(start, goal)


def test_exact_endpoints_must_avoid_world_boundary_clearance() -> None:
    corridor = CoarsePathCorridor(10.0, 10.0, (), 1.0, 1.0, obstacle_clearance=0.9)

    with pytest.raises(ValueError, match=r"start violates.*world-boundary clearance"):
        corridor.build((0.5, 5.0), (8.0, 5.0))


def test_coarse_occupancy_blocks_world_boundary_clearance() -> None:
    corridor = CoarsePathCorridor(5.0, 5.0, (), 1.0, 1.0, obstacle_clearance=0.9)

    assert np.all(corridor.occupancy[0, :])
    assert np.all(corridor.occupancy[-1, :])
    assert np.all(corridor.occupancy[:, 0])
    assert np.all(corridor.occupancy[:, -1])
    assert not corridor.occupancy[2, 2]


def test_connector_candidates_reject_segments_crossing_inflated_obstacle() -> None:
    corridor = CoarsePathCorridor(
        10.0,
        10.0,
        ((1.4, 1.6, 2.4, 3.1),),
        coarse_resolution=2.0,
        corridor_width=1.0,
    )

    candidates = corridor._connector_candidates("start", (1.0, 3.0))
    candidate_indices = {(ix, iy) for _, ix, iy in candidates}

    assert (1, 1) not in candidate_indices
    assert (1, 2) in candidate_indices


def test_missing_local_connector_has_descriptive_error() -> None:
    corridor = CoarsePathCorridor(
        10.0,
        10.0,
        ((1.9, 2.1, 1.9, 2.1),),
        coarse_resolution=2.0,
        corridor_width=1.0,
        obstacle_clearance=0.5,
    )

    with pytest.raises(RuntimeError, match=r"start has no nearby free grid vertex"):
        corridor.build((1.0, 1.0), (8.0, 8.0))


def test_planner_rejects_arc_that_leaves_corridor() -> None:
    environment = make_environment()
    planner = HybridAStar(
        environment,
        Vehicle(),
        safety_margin=0.0,
        integration_step=0.1,
        collision_check_step=0.05,
        corridor_width=0.01,
        coarse_resolution=1.0,
    )
    planner._prepare_corridor(environment.start, environment.goal)
    start = Node(*environment.start, 0.0, None, 1, 2, 0.0)

    straight = planner.check_primitive(start, 1, 2, planner._collision_distances)
    turning = planner.check_primitive(start, 1, 4, planner._collision_distances)

    assert straight is not None
    assert turning is None


def test_planner_builds_and_uses_corridor_during_plan() -> None:
    environment = make_environment()
    planner = HybridAStar(
        environment,
        Vehicle(),
        safety_margin=0.0,
        integration_step=0.1,
        collision_check_step=0.05,
        corridor_width=0.5,
        coarse_resolution=1.0,
    )

    path, _, _, _ = planner.plan(environment.start, environment.goal, max_expansions=100)

    assert planner.corridor is not None
    assert planner.corridor.obstacle_clearance == pytest.approx(planner.vehicle.width / 2.0)
    assert all(planner.corridor.contains(x, y) for x, y in path[:, :2])


def test_corridor_centerline_clearance_includes_the_safety_margin() -> None:
    environment = make_environment()
    safety_margin = 0.25
    planner = HybridAStar(
        environment,
        Vehicle(),
        safety_margin=safety_margin,
        integration_step=0.1,
        collision_check_step=0.05,
        corridor_width=0.5,
        coarse_resolution=1.0,
    )

    planner._prepare_corridor(environment.start, environment.goal)

    assert planner.corridor is not None
    assert planner.corridor.obstacle_clearance == pytest.approx(
        planner.vehicle.width / 2.0 + safety_margin
    )


def test_dijkstra_search_is_restricted_to_built_corridor() -> None:
    environment = make_environment()
    planner = HybridAStar(
        environment,
        Vehicle(),
        safety_margin=0.0,
        integration_step=0.1,
        collision_check_step=0.05,
        heuristic_mode="dijkstra",
        corridor_width=0.5,
        coarse_resolution=1.0,
    )
    planner.plan(environment.start, environment.goal, max_expansions=100)

    costs = planner._dijkstra_cost_to_goal
    assert costs is not None
    resolution = planner.xy_resolution

    assert math.isfinite(costs[round(5.0 / resolution), round(6.0 / resolution)])
    assert math.isinf(costs[round(6.0 / resolution), round(6.0 / resolution)])
    finite_y, finite_x = np.nonzero(np.isfinite(costs))
    assert planner.corridor is not None
    assert all(
        planner.corridor.contains(ix * resolution, iy * resolution)
        for iy, ix in zip(finite_y, finite_x)
    )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_corridor_width_is_rejected(value: float) -> None:
    environment = make_environment()

    with pytest.raises(ValueError, match="corridor_width"):
        HybridAStar(environment, Vehicle(), 0.0, 0.1, 0.05, corridor_width=value)


def test_cli_corridor_width_enables_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from hybrid_astar_main import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hybrid_astar_main.py",
            "--corridor_width",
            "2.5",
            "--coarse_resolution",
            "0.75",
        ],
    )

    args = parse_args()

    assert args.corridor_width == pytest.approx(2.5)
    assert args.coarse_resolution == pytest.approx(0.75)


def test_live_plot_shades_only_outside_corridor() -> None:
    import hybrid_astar_main as demo_module

    environment = make_environment()
    planner = HybridAStar(
        environment,
        Vehicle(),
        safety_margin=0.0,
        integration_step=0.1,
        collision_check_step=0.05,
        corridor_width=0.5,
        coarse_resolution=1.0,
    )
    path, _, _, terminal = planner.plan(environment.start, environment.goal, max_expansions=100)
    heuristic = planner.heuristic(terminal.x, terminal.y, terminal.yaw)
    snapshot = SearchSnapshot(terminal, path, heuristic, terminal.cost + heuristic)
    state = SearchNodeState(terminal, heuristic, snapshot.total_estimate, closed=False)

    plot = demo_module.LiveSearchPlot(planner, environment)
    try:
        plot.update(planner.expansion_count, snapshot, snapshot, (state,))

        assert len(plot.corridor_overlays) == 2
        alpha = np.asarray(plot.corridor_overlays[0].get_array())[:, :, 3]
        assert alpha.min() == pytest.approx(0.0)
        assert alpha.max() == pytest.approx(0.42)
    finally:
        demo_module.plt.close(plot.fig)
