#!/usr/bin/env python3
"""Hybrid A* for a car-like robot without analytic Reeds-Shepp expansion.

Examples:
    python hybrid_astar_demo.py
    python hybrid_astar_demo.py --env walls
    python hybrid_astar_demo.py --env parking
    python hybrid_astar_demo.py --env parking --live_plot_every 250
    python hybrid_astar_demo.py --env parking --heuristic dijkstra
    python hybrid_astar_demo.py --env parking --post_goal_expansions 10_000
    python hybrid_astar_demo.py --env parking --safety_margin 0.50 \
        --integration_step 0.08 --collision_check_step 0.05

The sampling/safety arguments have the same defaults for every environment:

    margin=0.20 m, integration=0.10 m, collision=0.05 m

``integration_step`` controls reconstructed path/animation sampling.
``collision_check_step`` independently controls swept-path collision sampling.
Motion primitives use exact constant-curvature bicycle arcs, so changing either
sampling step does not change a primitive's endpoint.

``heuristic`` selects the distance-only, legacy distance-and-heading,
tolerance-aware, or obstacle-aware estimate used for queue ordering and the
live-search comparison. Open states are expanded by the weighted total estimate
``g+weight*h``.

Enable ``--enable_admissible_bound`` to certify fine-only cost search with an
admissible lower-bound queue. Ordinary A* with the tolerance heuristic and
weight 1 reuses the main queue; weighted or otherwise greedily ordered
fine-only search keeps a separate ``g+tolerance_lower_bound`` queue over the
same live states. The best goal is returned when its cost is no greater than
the minimum remaining lower bound. If ``max_expansions`` is reached first, the
best incumbent is returned with a warning that optimality was not certified.

``post_goal_expansions`` is the fallback termination policy when the admissible
certificate is disabled, including two-queue search. It continues search for a
requested number of additional action-set expansions after the first goal while
retaining the lowest-cost terminal seen.

The optional live plot colors every open state by ``g+weight*h`` or ``h``. Closed
states use gray dots underneath the open-state dots. The left panel keeps a full
vehicle box on its minimum-score open state, while the right panel selects the
minimum-heuristic state across both open and closed states. Per-state heading arrows are
omitted because dense searches become unreadable.

This variant optionally uses two OPEN queues over one shared state graph:

* the fine queue expands short primitives with all five steering values;
* the coarse queue expands longer primitives with three steering values
  ``{-max, 0, +max}``.

Enable the second queue with ``--two_queues``. Set its edge length with
``--coarse_primitive_mult``; it defaults to four times the fine
``--primitive_length``. ``--coarse_heuristic_weight`` can give the coarse
queue a different heuristic multiplier; when omitted, it uses
``--heuristic_weight``. At each iteration the coarse queue is eligible when
``min_coarse_priority <= queue_beta * min_fine_priority``. A multiplicative
origin-priority factor penalizes fine-generated nodes in the coarse queue; it
does not affect coarse-generated nodes. To prevent the fine queue from being
starved, at most X consecutive coarse expansions are allowed while a fine state
remains available; the next expansion is then forced from the fine queue.
Fine actions therefore remain available everywhere, but are not
generated on every coarse expansion. Both primitive sets use the same swept
collision-check spacing. Without ``--two_queues`` the planner runs the original
fine-only search and does not apply the origin-priority factor.
"""

import argparse
import json
import heapq
import math
from datetime import datetime
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

plt.rcParams.update({"font.size": 8})  # , "figure.dpi": 100})


@njit(cache=True)
def _sample_collision_free_primitive(
    x0: float,
    y0: float,
    yaw0: float,
    direction: int,
    steer: float,
    wheelbase: float,
    collision_distances: np.ndarray,
    half_length: float,
    half_width: float,
    center_offset: float,
    world_width: float,
    world_height: float,
    obstacle_boxes: np.ndarray,
) -> tuple[bool, float, float, float]:
    """Integrate and collision-check one constant-steering motion primitive.

    The numeric loop is self-contained so Numba can compile arc integration and separating-
    axis checks together.

    Args:
        x0: Rear-axle x-coordinate at the start of the primitive, in metres.
        y0: Rear-axle y-coordinate at the start of the primitive, in metres.
        yaw0: Vehicle heading at the start of the primitive, in radians.
        direction: Travel direction: ``1`` for forward or ``-1`` for reverse.
        steer: Constant front-wheel steering angle, in radians.
        wheelbase: Distance between the front and rear axles, in metres.
        collision_distances: Arc-length samples used for swept collision checks.
        half_length: Half of the safety-inflated vehicle length, in metres.
        half_width: Half of the safety-inflated vehicle width, in metres.
        center_offset: Longitudinal offset from the rear axle to the box centre.
        world_width: Width of the rectangular planning world, in metres.
        world_height: Height of the rectangular planning world, in metres.
        obstacle_boxes: Obstacle rows formatted as ``[xmin, xmax, ymin, ymax]``.

    Returns:
        A ``(collision_free, x, y, yaw)`` tuple containing the primitive endpoint.
    """
    curvature = math.tan(steer) / wheelbase
    straight = abs(curvature) < 1e-12

    # Initialize the endpoint so an empty sampling array safely returns the
    # unchanged input pose. Normal planner construction always supplies at
    # least the exact primitive endpoint.
    x = x0
    y = y0
    yaw = yaw0

    # Collision samples normally dominate planning time.  Keeping SAT and arc
    # integration together lets Numba compile the whole loop without Python calls.
    for distance in collision_distances:
        signed_distance = direction * distance
        if straight:
            yaw = yaw0
            x = x0 + signed_distance * math.cos(yaw0)
            y = y0 + signed_distance * math.sin(yaw0)
        else:
            unwrapped_yaw = yaw0 + signed_distance * curvature
            x = x0 + (math.sin(unwrapped_yaw) - math.sin(yaw0)) / curvature
            y = y0 - (math.cos(unwrapped_yaw) - math.cos(yaw0)) / curvature
            yaw = (unwrapped_yaw + math.pi) % (2.0 * math.pi) - math.pi

        c, s = math.cos(yaw), math.sin(yaw)
        cx = x + center_offset * c
        cy = y + center_offset * s
        ac, ass = abs(c), abs(s)
        radius_x = half_length * ac + half_width * ass
        radius_y = half_length * ass + half_width * ac
        if (
            cx - radius_x <= 0.0
            or cx + radius_x >= world_width
            or cy - radius_y <= 0.0
            or cy + radius_y >= world_height
        ):
            return False, x0, y0, yaw0

        for obstacle in obstacle_boxes:
            ox = (obstacle[0] + obstacle[1]) * 0.5
            oy = (obstacle[2] + obstacle[3]) * 0.5
            ex = (obstacle[1] - obstacle[0]) * 0.5
            ey = (obstacle[3] - obstacle[2]) * 0.5
            dx, dy = ox - cx, oy - cy
            if abs(dx) > ex + radius_x or abs(dy) > ey + radius_y:
                continue
            if abs(dx * c + dy * s) > half_length + ex * ac + ey * ass:
                continue
            if abs(-dx * s + dy * c) > half_width + ex * ass + ey * ac:
                continue
            return False, x0, y0, yaw0

    # ``collision_distances`` always includes the exact primitive endpoint,
    # so the last loop iteration already calculated the terminal pose.
    return True, x, y, yaw


def wrap(angle: float) -> float:
    """Normalize an angle to the half-open interval ``[-pi, pi)``.

    Args:
        angle: Angle in radians; values outside one revolution are allowed.

    Returns:
        The equivalent wrapped angle in radians.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Vehicle:
    """Store geometry and steering limits for the bicycle model."""

    wheelbase: float = 2.6  # Distance between rear and front axles, in metres.
    length: float = 4.4  # Overall body length, in metres.
    width: float = 1.8  # Overall body width, in metres.
    rear_overhang: float = 1.0  # Distance from rear bumper to rear axle, in metres.
    max_steer: float = math.radians(40.0)  # Maximum absolute steering angle, in radians.


@dataclass(frozen=True)
class Obstacle:
    """Describe one axis-aligned rectangular obstacle."""

    xmin: float  # Minimum obstacle x-coordinate.
    xmax: float  # Maximum obstacle x-coordinate.
    ymin: float  # Minimum obstacle y-coordinate.
    ymax: float  # Maximum obstacle y-coordinate.
    kind: str = "wall"  # Display category for the obstacle.


@dataclass(frozen=True)
class Environment:
    """Describe a complete planning scene and planner overrides."""

    name: str  # Stable scene identifier.
    title: str  # Human-readable figure title.
    width: float  # World width, in metres.
    height: float  # World height in metres.
    obstacles: tuple[Obstacle, ...]  # Axis-aligned obstacles in the scene.
    start: tuple[float, float, float]  # Initial rear-axle pose.
    goal: tuple[float, float, float]  # Nominal rear-axle goal pose.
    planner: dict[str, float] = field(default_factory=dict)  # Planner overrides.


def _validated_pose(
    name: str,
    pose: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate and normalize one public pose argument.

    Args:
        name: Human-readable argument name used in error messages.
        pose: Candidate ``(x, y, yaw)`` sequence.

    Returns:
        A three-float pose tuple with yaw wrapped into ``[-pi, pi)``.

    Raises:
        ValueError: If the pose does not contain exactly three finite values.
    """
    try:
        values = tuple(float(value) for value in pose)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three numeric values") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain exactly three finite values")
    return values[0], values[1], wrap(values[2])


def validate_planner_inputs(
    environment: Environment,
    vehicle: Vehicle,
    safety_margin: float,
    integration_step: float,
    collision_check_step: float,
) -> None:
    """Validate the scene, vehicle geometry, and sampling parameters.

    Args:
        environment: Planning bounds, obstacles, and start/goal poses.
        vehicle: Vehicle dimensions and steering limits.
        safety_margin: Clearance added to every side during collision checking.
        integration_step: Spacing of reconstructed path samples.
        collision_check_step: Spacing of swept-path collision samples.

    Raises:
        ValueError: If any input is invalid.
    """
    _validated_pose("environment.start", environment.start)
    _validated_pose("environment.goal", environment.goal)
    if not math.isfinite(environment.width) or environment.width <= 0.0:
        raise ValueError("environment.width must be a finite positive number")
    if not math.isfinite(environment.height) or environment.height <= 0.0:
        raise ValueError("environment.height must be a finite positive number")
    for obstacle in environment.obstacles:
        coordinates = (obstacle.xmin, obstacle.xmax, obstacle.ymin, obstacle.ymax)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("obstacle coordinates must be finite")
        if obstacle.xmin >= obstacle.xmax or obstacle.ymin >= obstacle.ymax:
            raise ValueError("obstacle minimum bounds must be below maximum bounds")

    for name, value in (
        ("vehicle.wheelbase", vehicle.wheelbase),
        ("vehicle.length", vehicle.length),
        ("vehicle.width", vehicle.width),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if (
        not math.isfinite(vehicle.rear_overhang)
        or vehicle.rear_overhang < 0.0
        or vehicle.rear_overhang > vehicle.length
    ):
        raise ValueError("vehicle.rear_overhang must be finite and within [0, length]")
    if not math.isfinite(vehicle.max_steer) or vehicle.max_steer <= 0.0 or vehicle.max_steer >= math.pi / 2.0:
        raise ValueError("vehicle.max_steer must be finite and in (0, pi/2)")
    if not math.isfinite(safety_margin) or safety_margin < 0.0:
        raise ValueError("safety_margin must be a finite non-negative number")
    if not math.isfinite(integration_step) or integration_step <= 0.0:
        raise ValueError("integration_step must be a finite positive number")
    if not math.isfinite(collision_check_step) or collision_check_step <= 0.0:
        raise ValueError("collision_check_step must be a finite positive number")


def validate_planner_options(options: dict[str, float]) -> None:
    """Validate the required environment-specific planner options.

    Args:
        options: Planner configuration values supplied by an environment.

    Raises:
        ValueError: If a required option is missing or has an invalid value.
    """
    required_options = {
        "xy_resolution",
        "yaw_resolution",
        "primitive_length",
        "position_tolerance",
        "yaw_tolerance",
        "reverse_multiplier",
        "gear_change_penalty",
        "steering_change_penalty",
    }
    missing_options = required_options.difference(options)
    if missing_options:
        missing = ", ".join(sorted(missing_options))
        raise ValueError(f"environment.planner is missing required options: {missing}")

    for name in (
        "xy_resolution",
        "yaw_resolution",
        "primitive_length",
        "reverse_multiplier",
    ):
        value = options[name]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if options["yaw_resolution"] > math.tau:
        raise ValueError("yaw_resolution must not exceed one full revolution")
    for name in (
        "position_tolerance",
        "yaw_tolerance",
        "gear_change_penalty",
        "steering_change_penalty",
    ):
        value = options[name]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")


def validate_search_inputs(
    max_expansions: int,
    live_plot_every: int,
    post_goal_expansions: int,
    max_consecutive_coarse_expansions: int,
) -> None:
    """Validate public search-control arguments.

    Args:
        max_expansions: Maximum number of fine-plus-coarse action-set expansions.
        live_plot_every: Live-search update interval; zero disables updates.
        post_goal_expansions: Additional expansions after the first accepted goal.
        max_consecutive_coarse_expansions: Maximum consecutive coarse expansions
            while a fine state remains available.

    Raises:
        ValueError: If any search-control argument is invalid.
    """
    if not isinstance(max_expansions, Integral) or isinstance(max_expansions, bool):
        raise ValueError("max_expansions must be an integer")
    if max_expansions <= 0:
        raise ValueError("max_expansions must be greater than zero")
    if not isinstance(live_plot_every, Integral) or isinstance(live_plot_every, bool):
        raise ValueError("live_plot_every must be an integer")
    if live_plot_every < 0:
        raise ValueError("live_plot_every must be non-negative")
    if not isinstance(post_goal_expansions, Integral) or isinstance(post_goal_expansions, bool):
        raise ValueError("post_goal_expansions must be an integer")
    if post_goal_expansions < 0:
        raise ValueError("post_goal_expansions must be non-negative")
    if not isinstance(max_consecutive_coarse_expansions, Integral) or isinstance(
        max_consecutive_coarse_expansions, bool
    ):
        raise ValueError("max_consecutive_coarse_expansions must be an integer")
    if max_consecutive_coarse_expansions < 0:
        raise ValueError("max_consecutive_coarse_expansions must be non-negative")


# Search bookkeeping identifies a state by discretized rear-axle pose, optionally
# augmented with the incoming direction and steering index. Direction and previous
# steering affect gear-change and steering-change costs, so ``pose_control`` is the
# Markov representation for this objective. ``pose`` is retained only as the legacy
# compact mode; it can merge arrivals with different continuation costs.
PoseKey = tuple[int, int, int]
StateKey = tuple[int, ...]
StateKeyMode = Literal["pose", "pose_control"]
HeuristicMode = Literal["distance", "default", "defaultw1", "tolerance", "dijkstra"]


@dataclass(slots=True, eq=False)
class Node:
    """Store one compact generated node and its incoming control."""

    x: float  # Rear-axle x-coordinate.
    y: float  # Rear-axle y-coordinate.
    yaw: float  # Vehicle heading in radians.
    cost: float  # Accumulated path cost from the start.
    parent: Optional["Node"]  # Exact predecessor, or ``None`` at the start.
    direction: int  # Incoming travel direction.
    steer_index: int  # Index of the incoming steering angle in ``HybridAStar.steers``.
    primitive_length: float  # Incoming primitive length; zero only for the start [m].


@dataclass(frozen=True)
class SearchSnapshot:
    """Store immutable path and score data for one selected node."""

    node: Node  # Selected/current search node.
    path: np.ndarray  # Reconstructed path to the node.
    heuristic: float  # Heuristic score h.
    total_estimate: float  # Combined score g+weight*h.


@dataclass(frozen=True)
class SearchNodeState:
    """Store the best node and display scores for one state key."""

    node: Node  # Selected/current search node.
    heuristic: float  # Heuristic score h.
    total_estimate: float  # Combined score g+weight*h.
    closed: bool  # Whether all enabled action sets expanded the current state.


# Heap entries store the total priority, heuristic, negative path cost, serial,
# state key, and exact Node object. Object identity rejects stale heap entries.
OpenEntry = tuple[float, float, float, int, StateKey, Node]
ProgressCallback = Callable[
    [int, SearchSnapshot, SearchSnapshot, tuple[SearchNodeState, ...]],
    None,
]
ExpansionCallback = Callable[[int], None]


def vehicle_polygon(
    x: float,
    y: float,
    yaw: float,
    vehicle: Vehicle,
    margin: float = 0.0,
) -> np.ndarray:
    """Compute the vehicle body's four world-space rectangle corners.

    Args:
        x: Rear-axle or grid-point x-coordinate in world metres.
        y: Rear-axle or grid-point y-coordinate in world metres.
        yaw: Vehicle heading in radians.
        vehicle: Vehicle dimensions, wheelbase, and steering limits.
        margin: Optional clearance added to every side of the footprint.

    Returns:
        A ``4 x 2`` array containing world-space body corners.
    """
    rear = -vehicle.rear_overhang - margin
    front = vehicle.length - vehicle.rear_overhang + margin
    half_width = vehicle.width / 2.0 + margin
    local = np.array(
        [[front, half_width], [front, -half_width], [rear, -half_width], [rear, half_width]],
        dtype=float,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def vehicle_heading_arrow(
    x: float,
    y: float,
    yaw: float,
    vehicle: Vehicle,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute endpoints for an arrow indicating the vehicle's forward direction.

    Args:
        x: Rear-axle or grid-point x-coordinate in world metres.
        y: Rear-axle or grid-point y-coordinate in world metres.
        yaw: Vehicle heading in radians.
        vehicle: Vehicle dimensions, wheelbase, and steering limits.

    Returns:
        The arrow start point and tip point.
    """
    front = vehicle.length - vehicle.rear_overhang
    tip_distance = front - 0.25
    c, s = math.cos(yaw), math.sin(yaw)
    return ((x, y), (x + tip_distance * c, y + tip_distance * s))


def vehicle_tire_polygons(
    x: float,
    y: float,
    yaw: float,
    steer: float,
    vehicle: Vehicle,
) -> list[np.ndarray]:
    """Compute world-space polygons for the rear and steered front tires.

    Args:
        x: Rear-axle or grid-point x-coordinate in world metres.
        y: Rear-axle or grid-point y-coordinate in world metres.
        yaw: Vehicle heading in radians.
        steer: Constant front-wheel steering angle, in radians.
        vehicle: Vehicle dimensions, wheelbase, and steering limits.

    Returns:
        Four ``4 x 2`` tire-footprint arrays.
    """
    tire_length = 0.68
    tire_width = 0.24
    half_track = vehicle.width / 2.0 - tire_width / 2.0
    axle_positions = ((0.0, yaw), (vehicle.wheelbase, yaw + steer))
    tires: list[np.ndarray] = []
    for axle_x, tire_yaw in axle_positions:
        for side in (-half_track, half_track):
            # Position each tire center in the vehicle frame, then rotate it into
            # the world frame using the body yaw.
            center = np.array([axle_x, side])
            c, s = math.cos(yaw), math.sin(yaw)
            center = center @ np.array([[c, -s], [s, c]]).T + np.array([x, y])
            local = np.array(
                [
                    [tire_length / 2.0, tire_width / 2.0],
                    [tire_length / 2.0, -tire_width / 2.0],
                    [-tire_length / 2.0, -tire_width / 2.0],
                    [-tire_length / 2.0, tire_width / 2.0],
                ]
            )
            tc, ts = math.cos(tire_yaw), math.sin(tire_yaw)
            tires.append(local @ np.array([[tc, -ts], [ts, tc]]).T + center)
    return tires


def sample_distances(length: float, step: float) -> list[float]:
    """Create positive sample distances that always include the endpoint.

    Args:
        length: Total segment length in metres.
        step: Maximum nominal spacing between samples, in metres.

    Returns:
        Distances in ``(0, length]`` ending at the exact endpoint.

    Raises:
        ValueError: If ``length`` or ``step`` is non-finite or not positive.
    """
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a finite positive number")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number")
    count = int(math.floor(length / step))
    values = [index * step for index in range(1, count + 1)]
    if not values or length - values[-1] > 1e-10:
        values.append(length)
    else:
        # Replace a numerically close final multiple with the exact endpoint.
        values[-1] = length
    return values


def arc_pose(
    x0: float,
    y0: float,
    yaw0: float,
    signed_distance: float,
    steer: float,
    wheelbase: float,
) -> tuple[float, float, float]:
    """Integrate an exact constant-steering kinematic-bicycle arc.

    Args:
        x0: Rear-axle x-coordinate at the start of the primitive, in metres.
        y0: Rear-axle y-coordinate at the start of the primitive, in metres.
        yaw0: Vehicle heading at the start of the primitive, in radians.
        signed_distance: Travel distance in metres; negative values mean reverse.
        steer: Constant front-wheel steering angle, in radians.
        wheelbase: Distance between the front and rear axles, in metres.

    Returns:
        The terminal ``(x, y, yaw)`` pose with wrapped yaw.
    """
    curvature = math.tan(steer) / wheelbase
    if abs(curvature) < 1e-12:
        # Zero steering is the limiting straight-line case.
        return (
            x0 + signed_distance * math.cos(yaw0),
            y0 + signed_distance * math.sin(yaw0),
            yaw0,
        )
    yaw = yaw0 + signed_distance * curvature
    x = x0 + (math.sin(yaw) - math.sin(yaw0)) / curvature
    y = y0 - (math.cos(yaw) - math.cos(yaw0)) / curvature
    return x, y, wrap(yaw)


class HybridAStar:
    """Plan car-like paths with continuous bicycle arcs and discretized pose keys.

    Continuous endpoints are merged into discretized x, y, and yaw bins.
    """

    def __init__(
        self,
        environment: Environment,
        vehicle: Vehicle,
        safety_margin: float,
        integration_step: float,
        collision_check_step: float,
        heuristic_mode: HeuristicMode = "default",
        state_key_mode: StateKeyMode = "pose_control",
        heuristic_weight: float = 1.0,
        use_two_queues: bool = False,
        coarse_heuristic_weight: Optional[float] = None,
        coarse_primitive_mult: int = 4,
        queue_beta: float = 1.5,
        origin_priority_factor: float = 2.0,
    ) -> None:
        """Configure planner geometry, discretization, costs, and search ordering.

        Args:
            environment: Planning bounds, obstacles, poses, and optional overrides.
            vehicle: Vehicle dimensions, wheelbase, and steering limits.
            safety_margin: Clearance added to every side during collision checking.
            integration_step: Spacing of samples retained for path output and animation.
            collision_check_step: Independent spacing of swept-path collision samples.
            heuristic_mode: Heuristic name: ``distance``, ``default``,
                ``defaultw1``, ``tolerance``, or ``dijkstra``.
            state_key_mode: ``pose_control`` for a Markov control-aware state, or
                ``pose`` for the legacy pose-only state.
            heuristic_weight: Non-negative multiplier applied to ``h`` for total priority.
            use_two_queues: Whether to enable the coarse acceleration queue.
            coarse_heuristic_weight: Optional non-negative multiplier applied to
                ``h`` in the coarse queue. ``None`` uses ``heuristic_weight``.
            coarse_primitive_mult: Integer multiplier used to derive the coarse
                primitive length from the fine primitive length.
            queue_beta: Select the coarse queue when its minimum priority is at most
                this factor times the fine queue's minimum priority.
            origin_priority_factor: Multiplier applied to a fine-generated
                node's priority in the coarse queue when two queues are
                enabled. Coarse-generated nodes use the base priority in both
                queues. ``1.0`` disables this bias.

        Returns:
            None.

        Raises:
            ValueError: If a mode, weight, queue parameter, sampling step, or
                primitive length is invalid.
        """
        self.environment = environment
        self.vehicle = vehicle

        validate_planner_inputs(
            environment,
            vehicle,
            safety_margin,
            integration_step,
            collision_check_step,
        )
        self.safety_margin = safety_margin
        self.integration_step = integration_step
        self.collision_check_step = collision_check_step
        if heuristic_mode not in {"distance", "default", "defaultw1", "tolerance", "dijkstra"}:
            raise ValueError(
                "heuristic_mode must be 'distance', 'default', 'defaultw1', " "'tolerance', or 'dijkstra'"
            )
        if state_key_mode not in {"pose", "pose_control"}:
            raise ValueError("state_key_mode must be 'pose' or 'pose_control'")
        if not math.isfinite(heuristic_weight) or heuristic_weight < 0.0:
            raise ValueError("heuristic_weight must be a finite non-negative number")
        if coarse_heuristic_weight is not None and (
            not math.isfinite(coarse_heuristic_weight) or coarse_heuristic_weight < 0.0
        ):
            raise ValueError("coarse_heuristic_weight must be a finite non-negative number or None")
        assert isinstance(coarse_primitive_mult, int), "coarse_primitive_mult must be an int"
        if not math.isfinite(queue_beta) or queue_beta < 1.0:
            raise ValueError("queue_beta must be finite and at least 1.0")
        if not math.isfinite(origin_priority_factor) or origin_priority_factor < 1.0:
            raise ValueError("origin_priority_factor must be finite and at least 1.0")
        self.heuristic_mode = heuristic_mode
        self.state_key_mode = state_key_mode
        self.heuristic_weight = heuristic_weight
        self.use_two_queues = bool(use_two_queues)
        self.coarse_heuristic_weight = (
            heuristic_weight if coarse_heuristic_weight is None else coarse_heuristic_weight
        )
        self.queue_beta = queue_beta
        self.origin_priority_factor = origin_priority_factor

        # Environment-specific values override the planner defaults.
        options = environment.planner
        validate_planner_options(options)

        self.xy_resolution = options["xy_resolution"]
        requested_yaw_resolution = options["yaw_resolution"]

        self.yaw_bin_count = max(1, int(round(math.tau / requested_yaw_resolution)))
        # Use an exact divisor of one revolution so every cyclic yaw bin has the
        # same width, including the bin crossing the -pi/+pi boundary.
        self.yaw_resolution = math.tau / self.yaw_bin_count
        self.fine_primitive_length = options["primitive_length"]
        self.coarse_primitive_length = coarse_primitive_mult * self.fine_primitive_length
        if self.use_two_queues and self.coarse_primitive_length < self.fine_primitive_length:
            raise ValueError("coarse_primitive_mult must be at least 1 when two queues are enabled")
        if self.fine_primitive_length < self.xy_resolution:
            raise ValueError(
                "primitive_length must be at least xy_resolution; shorter primitives can "
                "collapse into the same discretized position state"
            )
        self.position_tolerance = options["position_tolerance"]
        self.yaw_tolerance = options["yaw_tolerance"]
        self.reverse_multiplier = options["reverse_multiplier"]
        self.gear_change_penalty = options["gear_change_penalty"]
        self.steering_change_penalty = options["steering_change_penalty"]
        self.steers = np.linspace(-vehicle.max_steer, vehicle.max_steer, 5)
        self.fine_steer_indices = tuple(range(len(self.steers)))
        self.coarse_steer_indices = (0, len(self.steers) // 2, len(self.steers) - 1)

        # Fine and coarse edges have different graph lengths but use the same
        # sampling resolutions. Precomputing collision and reconstruction samples
        # avoids repeated allocation in both the search and path reconstruction.
        self._fine_collision_distances = np.asarray(
            sample_distances(self.fine_primitive_length, collision_check_step), dtype=float
        )
        self._coarse_collision_distances = np.asarray(
            sample_distances(self.coarse_primitive_length, collision_check_step), dtype=float
        )
        self._integration_distances_by_length = {
            length: np.asarray(sample_distances(length, integration_step), dtype=float)
            for length in {self.fine_primitive_length, self.coarse_primitive_length}
        }
        self._obstacle_boxes = np.asarray(
            [(o.xmin, o.xmax, o.ymin, o.ymax) for o in environment.obstacles],
            dtype=float,
        ).reshape((-1, 4))
        self._half_length = vehicle.length / 2.0 + safety_margin
        self._half_width = vehicle.width / 2.0 + safety_margin
        self._center_offset = vehicle.length / 2.0 - vehicle.rear_overhang

        # These members are reset for each call to ``plan``.
        self.goal: tuple[float, float, float] | None = None
        self.expanded: list[tuple[float, float]] = []
        self.expansion_count = 0
        self.fine_expansion_count = 0
        self.coarse_expansion_count = 0
        self.unique_expanded_state_count = 0
        self.last_expansion_queue = "fine"
        self._dijkstra_cost_to_goal: Optional[np.ndarray] = None

    def key(self, x: float, y: float, yaw: float) -> PoseKey:
        """Quantize a continuous pose into geometric lattice coordinates.

        The yaw index is cyclic, so equivalent orientations on opposite sides of
        the ``-pi``/``+pi`` boundary share one state key.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.
            yaw: Vehicle heading in radians.

        Returns:
            Integer ``(x_index, y_index, yaw_index)`` lattice coordinates.
        """
        return (
            int(round(x / self.xy_resolution)),
            int(round(y / self.xy_resolution)),
            int(round((wrap(yaw) + math.pi) / self.yaw_resolution)) % self.yaw_bin_count,
        )

    def _state_key(self, node: Node, *, initial: bool = False) -> StateKey:
        """Return the configured pose-only or control-aware search key.

        Args:
            node: Node whose discretized pose identifies the state; its incoming controls
                are included only in ``pose_control`` mode.
            initial: Whether ``node`` is the start node, which has no incoming direction.

        Returns:
            A pose-only key in legacy mode, or a key augmented with incoming direction
            and nearest discrete steering index in control-aware mode.
        """
        pose_key = self.key(node.x, node.y, node.yaw)
        if self.state_key_mode == "pose":
            return pose_key
        direction = 0 if initial else int(node.direction)
        return (*pose_key, direction, node.steer_index)

    def collides(self, x: float, y: float, yaw: float) -> bool:
        """Test the safety-inflated vehicle box against bounds and obstacles.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.
            yaw: Vehicle heading in radians.

        Returns:
            ``True`` when the inflated footprint intersects the world or an obstacle.
        """
        half_length = self.vehicle.length / 2.0 + self.safety_margin
        half_width = self.vehicle.width / 2.0 + self.safety_margin
        center_offset = self.vehicle.length / 2.0 - self.vehicle.rear_overhang
        c, s = math.cos(yaw), math.sin(yaw)
        cx = x + center_offset * c
        cy = y + center_offset * s
        ac, ass = abs(c), abs(s)
        radius_x = half_length * ac + half_width * ass
        radius_y = half_length * ass + half_width * ac

        # First reject poses whose projected footprint touches the world boundary.
        if (
            cx - radius_x <= 0.0
            or cx + radius_x >= self.environment.width
            or cy - radius_y <= 0.0
            or cy + radius_y >= self.environment.height
        ):
            return True

        for obstacle in self.environment.obstacles:
            ox = (obstacle.xmin + obstacle.xmax) / 2.0
            oy = (obstacle.ymin + obstacle.ymax) / 2.0
            ex = (obstacle.xmax - obstacle.xmin) / 2.0
            ey = (obstacle.ymax - obstacle.ymin) / 2.0
            dx, dy = ox - cx, oy - cy

            # Cheap world-axis projections reject most distant obstacles.
            if abs(dx) > ex + radius_x:
                continue
            if abs(dy) > ey + radius_y:
                continue

            # The vehicle's longitudinal and lateral axes complete the SAT test.
            if abs(dx * c + dy * s) > half_length + ex * ac + ey * ass:
                continue
            if abs(-dx * s + dy * c) > half_width + ex * ass + ey * ac:
                continue
            return True
        return False

    def _minimum_cost_per_metre(self) -> float:
        """Return the cheapest configured translational cost for one metre.

        Returns:
            The minimum forward/reverse per-metre multiplier.
        """
        return min(1.0, self.reverse_multiplier)

    def _tolerance_aware_lower_bound(self, x: float, y: float, yaw: float) -> float:
        """Lower-bound the cost needed to enter the accepted goal tolerances.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.
            yaw: Vehicle heading in radians.

        Returns:
            A non-negative lower bound in planner cost units.

        Raises:
            AssertionError: If no search goal has been assigned.
        """
        assert self.goal is not None
        gx, gy, gyaw = self.goal
        distance_error = math.hypot(gx - x, gy - y)
        heading_error = abs(wrap(gyaw - yaw))

        # Only distance outside the accepted terminal disk remains mandatory.
        distance_lb = max(0.0, distance_error - self.position_tolerance)

        # A curvature-limited vehicle needs at least R * delta_yaw path length to
        # remove heading error outside the accepted yaw interval.
        minimum_turning_radius = self.vehicle.wheelbase / math.tan(self.vehicle.max_steer)
        heading_lb = minimum_turning_radius * max(
            0.0,
            heading_error - self.yaw_tolerance,
        )
        return self._minimum_cost_per_metre() * max(distance_lb, heading_lb)

    def _point_is_inside_obstacle(self, x: float, y: float) -> bool:
        """Check point occupancy for the relaxed two-dimensional Dijkstra grid.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.

        Returns:
            ``True`` when the point lies strictly inside an obstacle.
        """
        # This is deliberately a point-robot relaxation: it ignores the ego footprint,
        # safety margin, heading, steering, and corner-cutting constraints.
        return any(
            obstacle.xmin < x < obstacle.xmax and obstacle.ymin < y < obstacle.ymax
            for obstacle in self.environment.obstacles
        )

    def _build_2d_dijkstra_to_goal_region(self) -> np.ndarray:
        """Build an eight-connected cost-to-go grid from the goal region.

        Returns:
            A ``(ny, nx)`` relaxed point-robot cost-to-go array.

        Raises:
            AssertionError: If no search goal has been assigned.
        """
        assert self.goal is not None
        resolution = self.xy_resolution
        nx = int(math.floor(self.environment.width / resolution + 1e-9)) + 1
        ny = int(math.floor(self.environment.height / resolution + 1e-9)) + 1
        costs = np.full((ny, nx), math.inf, dtype=float)
        blocked = np.zeros((ny, nx), dtype=bool)

        # Rasterize obstacle interiors onto the point-robot grid.
        for iy in range(ny):
            y = iy * resolution
            for ix in range(nx):
                x = ix * resolution
                blocked[iy, ix] = self._point_is_inside_obstacle(x, y)

        # Multi-source Dijkstra starts from every free cell inside the accepted
        # positional goal tolerance.
        gx, gy, _ = self.goal
        queue: list[tuple[float, int, int]] = []
        for iy in range(ny):
            y = iy * resolution
            for ix in range(nx):
                if blocked[iy, ix]:
                    continue
                x = ix * resolution
                if math.hypot(x - gx, y - gy) <= self.position_tolerance + 1e-12:
                    costs[iy, ix] = 0.0
                    heapq.heappush(queue, (0.0, ix, iy))

        if not queue:
            # A very small tolerance may contain no grid point. Seed the nearest free
            # point with its remaining straight-line distance to the tolerance region.
            nearest: Optional[tuple[float, int, int]] = None
            for iy in range(ny):
                y = iy * resolution
                for ix in range(nx):
                    if blocked[iy, ix]:
                        continue
                    x = ix * resolution
                    seed_cost = max(0.0, math.hypot(x - gx, y - gy) - self.position_tolerance)
                    candidate = (seed_cost, ix, iy)
                    if nearest is None or candidate < nearest:
                        nearest = candidate
            if nearest is not None:
                seed_cost, ix, iy = nearest
                costs[iy, ix] = seed_cost
                heapq.heappush(queue, nearest)

        diagonal = math.sqrt(2.0) * resolution
        moves = (
            (-1, 0, resolution),
            (1, 0, resolution),
            (0, -1, resolution),
            (0, 1, resolution),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )

        # Standard Dijkstra relaxation propagates cost outward from the goal.
        while queue:
            cost, ix, iy = heapq.heappop(queue)
            if cost > costs[iy, ix] + 1e-12:
                continue
            for dx, dy, step_cost in moves:
                nx_index, ny_index = ix + dx, iy + dy
                if not (0 <= nx_index < nx and 0 <= ny_index < ny):
                    continue
                if blocked[ny_index, nx_index]:
                    continue
                candidate = cost + step_cost
                if candidate + 1e-12 < costs[ny_index, nx_index]:
                    costs[ny_index, nx_index] = candidate
                    heapq.heappush(queue, (candidate, nx_index, ny_index))
        return costs

    def _dijkstra_distance_to_goal_region(self, x: float, y: float) -> float:
        """Read a continuous-pose estimate from the cached Dijkstra grid.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.

        Returns:
            A non-negative estimate, or infinity when unreachable.

        Raises:
            AssertionError: If no search goal has been assigned.
        """
        assert self.goal is not None
        gx, gy, _ = self.goal
        if math.hypot(x - gx, y - gy) <= self.position_tolerance:
            return 0.0
        if self._dijkstra_cost_to_goal is None:
            self._dijkstra_cost_to_goal = self._build_2d_dijkstra_to_goal_region()

        resolution = self.xy_resolution
        center_x = int(round(x / resolution))
        center_y = int(round(y / resolution))
        ny, nx = self._dijkstra_cost_to_goal.shape
        candidates: list[float] = []

        # Inspect nearby cells so a valid continuous pose is not made infinite by
        # rounding to one blocked grid point. Subtracting snap distance maps the
        # grid estimate back toward the continuous pose.
        for iy in range(center_y - 1, center_y + 2):
            for ix in range(center_x - 1, center_x + 2):
                if not (0 <= ix < nx and 0 <= iy < ny):
                    continue
                grid_cost = float(self._dijkstra_cost_to_goal[iy, ix])
                if not math.isfinite(grid_cost):
                    continue
                snap_distance = math.hypot(x - ix * resolution, y - iy * resolution)
                candidates.append(grid_cost - snap_distance)
        if not candidates:
            return math.inf
        return max(0.0, max(candidates))

    def heuristic(self, x: float, y: float, yaw: float) -> float:
        """Evaluate the configured heuristic for a continuous vehicle pose.

        Args:
            x: Rear-axle or grid-point x-coordinate in world metres.
            y: Rear-axle or grid-point y-coordinate in world metres.
            yaw: Vehicle heading in radians.

        Returns:
            The selected heuristic value in planner cost units.

        Raises:
            AssertionError: If no search goal has been assigned.
        """
        assert self.goal is not None
        gx, gy, gyaw = self.goal
        distance = math.hypot(gx - x, gy - y)
        heading = abs(wrap(gyaw - yaw))

        if self.heuristic_mode == "distance":
            return distance
        if self.heuristic_mode == "default":
            # A useful goal-directed score: the angular term is converted to
            # length units using wheelbase. It is intentionally not admissible.
            return distance + 0.5 * self.vehicle.wheelbase * heading
        if self.heuristic_mode == "defaultw1":
            # A useful goal-directed score: the angular term is converted to
            # length units using wheelbase. It is intentionally not admissible.
            return distance + self.vehicle.wheelbase * heading
        tolerance_lower_bound = self._tolerance_aware_lower_bound(x, y, yaw)
        if self.heuristic_mode == "tolerance":
            return tolerance_lower_bound

        if self.heuristic_mode == "dijkstra":
            # The obstacle-aware mode combines two independent estimates by taking
            # their maximum rather than adding potentially overlapping path length.
            obstacle_distance = self._dijkstra_distance_to_goal_region(x, y)
            if not math.isfinite(obstacle_distance):
                # The relaxed raster grid may report disconnection because of coarse
                # discretization even when the continuous search may still succeed.
                # Keep the state eligible by reverting to the finite kinematic estimate.
                return tolerance_lower_bound
            obstacle_lower_bound = self._minimum_cost_per_metre() * obstacle_distance
            return max(tolerance_lower_bound, obstacle_lower_bound)

        raise AssertionError(f"Unhandled validated heuristic mode: {self.heuristic_mode}")

    def check_primitive(
        self,
        node: Node,
        direction: int,
        steer_index: int,
        collision_distances: np.ndarray,
    ) -> Optional[tuple[float, float, float]]:
        """Return the endpoint of one collision-free fine or coarse primitive.

        Args:
            node: Search node supplying the primitive's initial pose.
            direction: Travel direction: ``1`` for forward or ``-1`` for reverse.
            steer_index: Index of the constant front-wheel steering angle.
            collision_distances: Positive arc-length samples ending at the exact
                fine or coarse primitive length.

        Returns:
            The terminal ``(x, y, yaw)`` pose, or ``None`` if any swept sample
            collides with an obstacle or the world boundary.
        """
        collision_free, x, y, yaw = _sample_collision_free_primitive(
            node.x,
            node.y,
            node.yaw,
            direction,
            float(self.steers[steer_index]),
            self.vehicle.wheelbase,
            collision_distances,
            self._half_length,
            self._half_width,
            self._center_offset,
            self.environment.width,
            self.environment.height,
            self._obstacle_boxes,
        )
        if not collision_free:
            return None
        return x, y, yaw

    def _successor_cost(
        self,
        node: Node,
        direction: int,
        steer_index: int,
        primitive_length: float,
    ) -> float:
        """Calculate a representation-consistent fine or coarse successor cost.

        Travel cost scales with edge length. Steering and gear changes are event
        costs, so one coarse edge and its equivalent sequence of fine edges do
        not receive different change penalties merely because of resolution.
        The fine-length factor preserves the original fine-only cost scale.

        Args:
            node: Parent node supplying accumulated cost and incoming control.
            direction: Successor travel direction: ``1`` forward or ``-1`` reverse.
            steer_index: Index of the successor steering angle.
            primitive_length: Length of the selected fine or coarse edge [m].

        Returns:
            Accumulated path cost at the successor.
        """
        steer = float(self.steers[steer_index])
        previous_steer = float(self.steers[node.steer_index])
        reverse = self.reverse_multiplier if direction < 0 else 1.0
        travel_cost = primitive_length * reverse
        steering_change_cost = (
            self.fine_primitive_length
            * self.steering_change_penalty
            * abs(steer - previous_steer)
            / self.vehicle.max_steer
        )
        gear_change_cost = (
            self.gear_change_penalty if node.parent is not None and direction != node.direction else 0.0
        )
        return node.cost + travel_cost + steering_change_cost + gear_change_cost

    def reconstruct(self, terminal: Node) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct sampled poses and controls from variable-length parent edges.

        Args:
            terminal: Terminal node whose parent chain defines the path.

        Returns:
            Matching arrays ``(path, directions, steers)``. ``path`` contains
            sampled ``(x, y, yaw)`` poses; the other arrays contain one control
            value per path sample.
        """
        chain: list[Node] = []
        node: Optional[Node] = terminal
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()

        path_parts = [np.asarray([[chain[0].x, chain[0].y, chain[0].yaw]], dtype=float)]
        direction_parts = [np.asarray([chain[0].direction], dtype=int)]
        steer_parts = [np.asarray([float(self.steers[chain[0].steer_index])], dtype=float)]

        for parent, child in zip(chain, chain[1:]):
            distances = self._integration_distances_by_length.get(child.primitive_length)
            assert distances is not None
            segment = np.empty((len(distances), 3), dtype=float)
            steer = float(self.steers[child.steer_index])
            for index, distance in enumerate(distances):
                segment[index] = arc_pose(
                    parent.x,
                    parent.y,
                    parent.yaw,
                    child.direction * float(distance),
                    steer,
                    self.vehicle.wheelbase,
                )
            # Preserve the exact endpoint stored by the search kernel.
            segment[-1] = (child.x, child.y, child.yaw)
            path_parts.append(segment)
            direction_parts.append(np.full(len(segment), child.direction, dtype=int))
            steer_parts.append(np.full(len(segment), steer, dtype=float))

        return (
            np.concatenate(path_parts),
            np.concatenate(direction_parts),
            np.concatenate(steer_parts),
        )

    @staticmethod
    def exact_path_length(terminal: Node) -> float:
        """Return the geometric path length from stored motion primitives.

        This sums the exact arc length of each primitive in the terminal's
        parent chain. Unlike the reconstructed polyline length, it is independent
        of ``integration_step`` and does not replace curved arcs with shorter
        straight chords.

        Args:
            terminal: Terminal node whose parent chain defines the path.

        Returns:
            Exact path length in metres.
        """
        primitive_lengths: list[float] = []
        node: Optional[Node] = terminal
        while node is not None and node.parent is not None:
            primitive_lengths.append(node.primitive_length)
            node = node.parent
        return math.fsum(primitive_lengths)

    @staticmethod
    def _entry_is_current(
        entry: OpenEntry,
        closed: set[StateKey],
        nodes: dict[StateKey, Node],
    ) -> bool:
        """Check whether a lazy heap entry still represents the current open node.

        Args:
            entry: Heap entry containing priority, tie-breaker, key, and node object.
            closed: CLOSED set for the queue whose entry is being checked.
            nodes: Current best-known node object for every generated state key.

        Returns:
            Whether the entry is current and still open.
        """
        _, _, _, _, key, queued_node = entry
        return key not in closed and nodes.get(key) is queued_node

    def _pop_best_open(
        self,
        queue: list[OpenEntry],
        closed: set[StateKey],
        nodes: dict[StateKey, Node],
    ) -> Optional[tuple[StateKey, Node]]:
        """Discard stale heap entries and return the best valid open node.

        Args:
            queue: Priority heap ordered by the configured search score.
            closed: CLOSED set associated with the selected OPEN heap.
            nodes: Current best-known node object for every generated state key.

        Returns:
            The best ``(state_key, node)`` pair, or ``None``.
        """
        while queue:
            entry = heapq.heappop(queue)
            if self._entry_is_current(entry, closed, nodes):
                _, _, _, _, key, node = entry
                return key, node
        return None

    def _peek_best_open_priority(
        self,
        queue: list[OpenEntry],
        closed: set[StateKey],
        nodes: dict[StateKey, Node],
    ) -> Optional[float]:
        """Return the minimum current priority without removing the entry.

        Stale lazy-heap entries are removed before reading the minimum.

        Args:
            queue: Fine or coarse OPEN heap.
            closed: CLOSED set associated with that heap's action set.
            nodes: Current best-known node for every generated state key.

        Returns:
            The minimum valid priority, including infinity when that is the
            stored score, or ``None`` when no current open entry remains.
        """
        while queue and not self._entry_is_current(queue[0], closed, nodes):
            heapq.heappop(queue)
        return queue[0][0] if queue else None

    def _snapshot(self, node: Node, heuristic: Optional[float] = None) -> SearchSnapshot:
        """Build path and score data for one generated search node.

        Args:
            node: Search node supplying the pose, accumulated cost, and incoming controls.
            heuristic: Optional precomputed heuristic value for the selected node.

        Returns:
            An immutable path and score snapshot.
        """
        if heuristic is None:
            heuristic = self.heuristic(node.x, node.y, node.yaw)
        path, _, _ = self.reconstruct(node)
        return SearchSnapshot(
            node=node,
            path=path,
            heuristic=heuristic,
            total_estimate=node.cost + self.heuristic_weight * heuristic,
        )

    def _publish_search_state(
        self,
        fine_closed: set[StateKey],
        coarse_closed: set[StateKey],
        nodes: dict[StateKey, Node],
        progress_callback: ProgressCallback,
    ) -> None:
        """Publish the union frontier from the fine and coarse searches.

        A state is displayed as closed only after the current best representative
        has been expanded by every enabled action set.

        Args:
            fine_closed: State keys expanded with the five-steer fine action set.
            coarse_closed: State keys expanded with the three-steer coarse action set.
            nodes: Current best-known node for every generated state key.
            progress_callback: Consumer of the two selected snapshots and all
                display states.

        Returns:
            None.
        """
        if not nodes:
            return

        states: list[SearchNodeState] = []
        for key, node in nodes.items():
            heuristic = self.heuristic(node.x, node.y, node.yaw)
            states.append(
                SearchNodeState(
                    node=node,
                    heuristic=heuristic,
                    total_estimate=node.cost + self.heuristic_weight * heuristic,
                    # With two queues, a state is globally closed only after
                    # both action sets expanded its current best representative.
                    # In fine-only mode, fine CLOSED is the ordinary CLOSED set.
                    closed=(key in fine_closed and (not self.use_two_queues or key in coarse_closed)),
                )
            )

        state_tuple = tuple(states)
        open_states = tuple(state for state in state_tuple if not state.closed)
        if not open_states:
            return
        best_total = min(open_states, key=lambda state: state.total_estimate)
        best_heuristic = min(state_tuple, key=lambda state: state.heuristic)
        progress_callback(
            self.expansion_count,
            self._snapshot(best_total.node, best_total.heuristic),
            self._snapshot(best_heuristic.node, best_heuristic.heuristic),
            state_tuple,
        )

    def report_goal(
        self,
        label: str,
        node: Node,
        expansion: int,
        goal: tuple[float, float, float],
    ) -> None:
        """Print one accepted terminal's cost, path length, and pose error.

        Args:
            label: Prefix identifying an accepted or final-best terminal.
            node: Accepted terminal node to report.
            expansion: Global expansion count at which the node was accepted.
            goal: Nominal rear-axle ``(x, y, yaw)`` goal pose.
        """
        path, _, _ = self.reconstruct(node)
        sampled_path_length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
        exact_path_length = self.exact_path_length(node)
        position_error = math.hypot(node.x - goal[0], node.y - goal[1])
        yaw_error = abs(wrap(node.yaw - goal[2]))
        print(
            f"\n{label} goal: expansion={expansion}; cost={node.cost:.2f}; "
            f"exact path length={exact_path_length:.6f} m; "
            f"sampled path length={sampled_path_length:.6f} m; "
            f"position error={position_error:.3f} m; "
            f"yaw error={math.degrees(yaw_error):.2f} deg",
            flush=True,
        )

    def _finish_search(
        self,
        best_terminal: Node,
        best_terminal_expansion: int,
        goal: tuple[float, float, float],
        fine_closed: set[StateKey],
        coarse_closed: set[StateKey],
        nodes: dict[StateKey, Node],
        live_plot_every: int,
        progress_callback: Optional[ProgressCallback],
        expansion_callback: Optional[ExpansionCallback],
        *,
        announce_bound: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Node]:
        """Publish final state, report the selected terminal, and reconstruct it.

        Args:
            best_terminal: Lowest-cost terminal node found by the search.
            best_terminal_expansion: Expansion count when the terminal was accepted.
            goal: Nominal rear-axle ``(x, y, yaw)`` goal pose.
            fine_closed: State keys expanded with the fine action set.
            coarse_closed: State keys expanded with the coarse action set.
            nodes: Current best-known node for every generated state key.
            live_plot_every: Positive when final search state should be published.
            progress_callback: Optional consumer of search-progress snapshots.
            expansion_callback: Optional consumer of the final expansion count.
            announce_bound: Whether to report an admissible-bound certificate.

        Returns:
            The reconstructed path, directions, steering values, and terminal node.
        """
        if expansion_callback is not None:
            expansion_callback(self.expansion_count)
        if progress_callback is not None and live_plot_every > 0:
            self._publish_search_state(
                fine_closed,
                coarse_closed,
                nodes,
                progress_callback,
            )
        if announce_bound:
            print(
                "Bound condition satisfied at expansion "
                f"{self.expansion_count}: best goal cost <= minimum lower bound",
                flush=True,
            )
        self.report_goal("Best", best_terminal, best_terminal_expansion, goal)
        path, directions, steers = self.reconstruct(best_terminal)
        return path, directions, steers, best_terminal

    def stop_condition(
        self,
        best_terminal: Optional[Node],
        first_goal_expansion: Optional[int],
        use_admissible_bound: bool,
        fine_queue_is_lower_bound: bool,
        fine_queue: list[OpenEntry],
        lower_bound_queue: list[OpenEntry],
        fine_closed: set[StateKey],
        nodes: dict[StateKey, Node],
        post_goal_expansions: int,
    ) -> bool:
        """Return whether the active termination policy allows returning.

        Fine-only cost search uses a proof: every live OPEN state has an
        admissible lower bound ``g + h_lb`` on any solution through it. Once
        the incumbent goal cost is no larger than the minimum such bound, no
        cheaper goal can remain.

        When the proof is disabled, including two-queue search, termination
        instead uses the configured post-goal expansion budget. The two policies
        are mutually exclusive so a zero post-goal budget cannot override an
        enabled admissible certificate.

        Args:
            best_terminal: Lowest-cost terminal node found by the search.
            first_goal_expansion: Expansion count at which the first goal was accepted.
            use_admissible_bound: Whether to use the cost-optimality certificate.
            fine_queue_is_lower_bound: Whether the fine queue is the bound queue.
            fine_queue: Fine OPEN heap.
            lower_bound_queue: Optional separate admissible-bound OPEN heap.
            fine_closed: State keys expanded with the fine action set.
            nodes: Current best-known node for every generated state key.
            post_goal_expansions: Required expansions after the first goal otherwise.

        Returns:
            Whether the active termination policy is satisfied.
        """
        if best_terminal is None or first_goal_expansion is None:
            return False

        if use_admissible_bound:
            bound_queue = fine_queue if fine_queue_is_lower_bound else lower_bound_queue
            minimum_lower_bound = self._peek_best_open_priority(
                bound_queue,
                fine_closed,
                nodes,
            )
            return minimum_lower_bound is None or (best_terminal.cost <= minimum_lower_bound + 1e-9)
        return self.expansion_count >= first_goal_expansion + post_goal_expansions

    def plan(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        max_expansions: int,
        live_plot_every: int = 0,
        progress_callback: Optional[ProgressCallback] = None,
        expansion_callback: Optional[ExpansionCallback] = None,
        post_goal_expansions: int = 0,
        enable_admissible_bound: bool = False,
        max_consecutive_coarse_expansions: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Node]:
        """Search with a fine queue and an optional coarse acceleration queue.

        The fine queue is globally available and expands five steering values.
        The optional coarse queue expands longer edges with ``{-max, 0, +max}``
        steering. The coarse queue is eligible when its current minimum satisfies
        ``min_coarse <= queue_beta * min_fine``. In two-queue mode, each generated
        node is promoted multiplicatively in the queue matching its generating
        action set and penalized in the other queue. While the fine queue is
        non-empty, no more than max_consecutive_coarse_expansions
         consecutive coarse expansions are allowed;
        Separate CLOSED sets allow the same state to receive both action sets.

        Args:
            start: Initial rear-axle ``(x, y, yaw)`` pose.
            goal: Nominal rear-axle ``(x, y, yaw)`` goal pose.
            max_expansions: Maximum number of fine-plus-coarse action-set expansions.
            live_plot_every: Publish live state every N action-set expansions;
                zero disables it.
            progress_callback: Optional consumer of full live-search snapshots.
            expansion_callback: Optional lightweight expansion-count callback.
            post_goal_expansions: Number of additional action-set expansions after
                the first accepted goal when the admissible cost certificate is disabled.
            enable_admissible_bound: Enable the admissible lower-bound certificate
                for fine-only cost search.
            max_consecutive_coarse_expansions: Maximum coarse expansions allowed while
                a fine state remains available; zero disables coarse selection in that
                situation.

        Returns:
            The reconstructed path, per-sample directions, per-sample steering
            angles, and selected terminal node.

        Raises:
            ValueError: If arguments are invalid or start/goal is in collision.
            RuntimeError: If no accepted terminal is found before termination.
        """
        validate_search_inputs(
            max_expansions,
            live_plot_every,
            post_goal_expansions,
            max_consecutive_coarse_expansions,
        )
        start = _validated_pose("start", start)
        goal = _validated_pose("goal", goal)

        # Reset per-search state and invalidate any Dijkstra map built for a
        # previous goal.
        self.goal = goal
        self._dijkstra_cost_to_goal = None
        self.expanded = []
        self.expansion_count = 0
        self.fine_expansion_count = 0
        self.coarse_expansion_count = 0
        self.unique_expanded_state_count = 0
        self.last_expansion_queue = "fine"
        if self.collides(*start):
            raise ValueError("Start pose (including safety margin) is in collision")
        if self.collides(*goal):
            raise ValueError("Goal pose (including safety margin) is in collision")

        # Keep one lowest-g representative for every configured state key.
        # Both queues share this table, but each queue has its own CLOSED set so
        # the representative can still receive both fine and coarse actions.
        start_node = Node(
            *start,
            0.0,
            None,
            1,
            int(np.argmin(np.abs(self.steers))),
            0.0,
        )
        start_key = self._state_key(start_node, initial=True)
        nodes: dict[StateKey, Node] = {start_key: start_node}
        fine_closed: set[StateKey] = set()
        coarse_closed: set[StateKey] = set()

        # Both search heaps store the same current node representatives. Queue-
        # specific CLOSED sets determine whether each action set remains pending.
        start_heuristic = self.heuristic(*start)
        fine_start_entry: OpenEntry = (
            start_node.cost + self.heuristic_weight * start_heuristic,
            start_heuristic,
            -start_node.cost,
            0,
            start_key,
            start_node,
        )
        fine_queue: list[OpenEntry] = [fine_start_entry]

        # A cost-optimality certificate is enabled only for fine-only search.
        # It is deliberately disabled for two-queue search because each queue has
        # a different pending action set and the current scheduler is not an anchor-
        # queue algorithm with a proved global lower-bound invariant.
        use_admissible_bound = enable_admissible_bound and not self.use_two_queues

        # The fine queue itself is a valid lower-bound queue only for ordinary A*
        # ordered by the admissible tolerance heuristic with unit weight. Any other
        # fine-only ordering (Weighted A*, default heuristic, Dijkstra ordering,
        # distance ordering, etc.) gets a separate anchor heap scored with the
        # tolerance-aware admissible lower bound.
        fine_queue_is_lower_bound = (
            use_admissible_bound and self.heuristic_mode == "tolerance" and self.heuristic_weight == 1.0
        )
        use_lower_bound_queue = use_admissible_bound and not fine_queue_is_lower_bound
        lower_bound_queue: list[OpenEntry] = []
        if use_lower_bound_queue:
            start_lower_bound = self._tolerance_aware_lower_bound(*start)
            lower_bound_queue.append(
                (
                    start_node.cost + start_lower_bound,
                    start_lower_bound,
                    -start_node.cost,
                    0,
                    start_key,
                    start_node,
                )
            )
        coarse_queue: list[OpenEntry] = (
            [
                (
                    start_node.cost + self.coarse_heuristic_weight * start_heuristic,
                    start_heuristic,
                    -start_node.cost,
                    0,
                    start_key,
                    start_node,
                )
            ]
            if self.use_two_queues
            else []
        )
        serial = 0
        best_terminal: Optional[Node] = None
        best_terminal_expansion: Optional[int] = None
        first_goal_expansion: Optional[int] = None

        # Bound coarse-queue bursts so the globally available fine action set cannot be starved.
        # After max_consecutive_coarse_expansions consecutive coarse expansions, force fine;
        max_coarse_streak = max_consecutive_coarse_expansions
        coarse_streak = 0
        expanded_state_keys: set[StateKey] = set()
        open_exhausted = False

        while self.expansion_count < max_expansions:
            if self.stop_condition(
                best_terminal,
                first_goal_expansion,
                use_admissible_bound,
                fine_queue_is_lower_bound,
                fine_queue,
                lower_bound_queue,
                fine_closed,
                nodes,
                post_goal_expansions,
            ):
                assert best_terminal is not None
                assert best_terminal_expansion is not None
                return self._finish_search(
                    best_terminal,
                    best_terminal_expansion,
                    goal,
                    fine_closed,
                    coarse_closed,
                    nodes,
                    live_plot_every,
                    progress_callback,
                    expansion_callback,
                    announce_bound=use_admissible_bound,
                )

            fine_min = self._peek_best_open_priority(fine_queue, fine_closed, nodes)
            coarse_min = (
                self._peek_best_open_priority(coarse_queue, coarse_closed, nodes)
                if self.use_two_queues
                else None
            )
            if fine_min is None and coarse_min is None:
                open_exhausted = True
                break

            coarse_is_preferred = (
                self.use_two_queues
                and coarse_min is not None
                and (fine_min is None or coarse_min <= self.queue_beta * fine_min)
            )
            use_coarse = coarse_is_preferred and (fine_min is None or coarse_streak < max_coarse_streak)
            if use_coarse:
                active_queue = coarse_queue
                active_closed = coarse_closed
                primitive_length = self.coarse_primitive_length
                collision_distances = self._coarse_collision_distances
                steer_indices = self.coarse_steer_indices
                queue_name = "coarse"
            else:
                active_queue = fine_queue
                active_closed = fine_closed
                primitive_length = self.fine_primitive_length
                collision_distances = self._fine_collision_distances
                steer_indices = self.fine_steer_indices
                queue_name = "fine"

            current_entry = self._pop_best_open(active_queue, active_closed, nodes)
            if current_entry is None:
                continue
            current_key, current = current_entry
            active_closed.add(current_key)
            expanded_state_keys.add(current_key)
            self.unique_expanded_state_count = len(expanded_state_keys)
            self.last_expansion_queue = queue_name
            self.expansion_count += 1
            if queue_name == "coarse":
                coarse_streak += 1
                self.coarse_expansion_count += 1
            else:
                coarse_streak = 0
                self.fine_expansion_count += 1

            if expansion_callback is not None and self.expansion_count % 100 == 0:
                expansion_callback(self.expansion_count)

            # A node is terminal only when both configured pose tolerances hold.
            position_error = math.hypot(current.x - goal[0], current.y - goal[1])
            yaw_error = abs(wrap(current.yaw - goal[2]))
            if self.expansion_count % 10 == 0:
                self.expanded.append((current.x, current.y))

            if position_error <= self.position_tolerance and yaw_error <= self.yaw_tolerance:
                if best_terminal is None or current.cost < best_terminal.cost:
                    best_terminal = current
                    best_terminal_expansion = self.expansion_count
                    self.report_goal("Accepted", current, self.expansion_count, goal)
                if first_goal_expansion is None:
                    first_goal_expansion = self.expansion_count

                # A cost-terminal is an absorbing search state. All outgoing edges
                # have non-negative cost, so expanding either action set cannot
                # produce a cheaper terminal. Closing both queue views also removes
                # the terminal from the remaining OPEN lower-bound frontier.
                fine_closed.add(current_key)
                coarse_closed.add(current_key)
                continue

            # Expand only the action set associated with the selected queue.
            for direction in (1, -1):
                for steer_index in steer_indices:
                    endpoint = self.check_primitive(
                        current,
                        direction,
                        steer_index,
                        collision_distances,
                    )
                    if endpoint is None:
                        continue
                    successor = Node(
                        *endpoint,
                        self._successor_cost(
                            current,
                            direction,
                            steer_index,
                            primitive_length,
                        ),
                        current,
                        direction,
                        steer_index,
                        primitive_length,
                    )
                    successor_key = self._state_key(successor)
                    incumbent = nodes.get(successor_key)
                    if incumbent is None or successor.cost + 1e-9 < incumbent.cost:
                        nodes[successor_key] = successor

                        # An improved shared state must be reconsidered with both
                        # action sets, even if an older representative was closed.
                        fine_closed.discard(successor_key)
                        coarse_closed.discard(successor_key)
                        serial += 1
                        heuristic = self.heuristic(
                            successor.x,
                            successor.y,
                            successor.yaw,
                        )
                        fine_priority = successor.cost + self.heuristic_weight * heuristic
                        coarse_priority = successor.cost + self.coarse_heuristic_weight * heuristic
                        if self.use_two_queues:
                            # Lower heap values are expanded first. Fine-generated
                            # nodes are penalized in the coarse queue. Each queue
                            # still retains its own configured heuristic weight.
                            if queue_name != "coarse":
                                coarse_priority *= self.origin_priority_factor
                        else:
                            # Preserve the original fine-only queue ordering.
                            coarse_priority = fine_priority

                        fine_entry: OpenEntry = (
                            fine_priority,
                            heuristic,
                            -successor.cost,
                            serial,
                            successor_key,
                            successor,
                        )
                        heapq.heappush(fine_queue, fine_entry)
                        if use_lower_bound_queue:
                            lower_bound_heuristic = self._tolerance_aware_lower_bound(
                                successor.x,
                                successor.y,
                                successor.yaw,
                            )
                            lower_bound_entry: OpenEntry = (
                                successor.cost + lower_bound_heuristic,
                                lower_bound_heuristic,
                                -successor.cost,
                                serial,
                                successor_key,
                                successor,
                            )
                            heapq.heappush(lower_bound_queue, lower_bound_entry)
                        if self.use_two_queues:
                            coarse_entry: OpenEntry = (
                                coarse_priority,
                                heuristic,
                                -successor.cost,
                                serial,
                                successor_key,
                                successor,
                            )
                            heapq.heappush(coarse_queue, coarse_entry)

            # Periodically publish the union of states still open in either queue.
            if (
                progress_callback is not None
                and live_plot_every > 0
                and self.expansion_count % live_plot_every == 0
            ):
                self._publish_search_state(
                    fine_closed,
                    coarse_closed,
                    nodes,
                    progress_callback,
                )

        policy_satisfied = self.stop_condition(
            best_terminal,
            first_goal_expansion,
            use_admissible_bound,
            fine_queue_is_lower_bound,
            fine_queue,
            lower_bound_queue,
            fine_closed,
            nodes,
            post_goal_expansions,
        )

        if best_terminal is not None:
            if use_admissible_bound and not policy_satisfied and not open_exhausted:
                print(
                    "\nWarning: the search limit was reached before the "
                    "admissible lower-bound certificate proved the "
                    "incumbent cost optimal.",
                    flush=True,
                )
            elif not use_admissible_bound and not policy_satisfied and not open_exhausted:
                print(
                    "\nWarning: the search limit was reached before the "
                    "requested post-goal expansion budget was completed.",
                    flush=True,
                )

            assert best_terminal_expansion is not None
            return self._finish_search(
                best_terminal,
                best_terminal_expansion,
                goal,
                fine_closed,
                coarse_closed,
                nodes,
                live_plot_every,
                progress_callback,
                expansion_callback,
                announce_bound=use_admissible_bound and policy_satisfied,
            )
        raise RuntimeError(
            f"No path found after {self.expansion_count} action-set expansions "
            f"({self.fine_expansion_count} fine, {self.coarse_expansion_count} coarse; "
            f"two_queues={self.use_two_queues})"
        )


def make_environment(name: str, planner_overrides: dict[str, float]) -> Environment:
    """Construct one named demonstration scene with planner overrides.

    Args:
        name: Identifier of a built-in demonstration environment.
        planner_overrides: Explicit planner values replacing environment defaults.

    Returns:
        The requested immutable Environment.

    Raises:
        ValueError: If the environment name is unknown.
    """
    if name == "walls":
        return Environment(
            name="walls",
            title="Alternating-wall environment",
            width=30.0,
            height=20.0,
            obstacles=(
                Obstacle(7.0, 9.0, 0.0, 11.5),
                Obstacle(7.0, 9.0, 15.0, 20.0),
                Obstacle(14.0, 16.0, 8.5, 20.0),
                Obstacle(14.0, 16.0, 0.0, 4.5),
                Obstacle(21.0, 23.0, 0.0, 11.0),
                Obstacle(21.0, 23.0, 14.5, 20.0),
            ),
            start=(3.0, 3.0, 0.0),
            goal=(26.5, 16.0, math.radians(90.0)),
            planner=planner_overrides,
        )
    if name == "parking":
        return Environment(
            name="parking",
            title="Parallel-parking environment",
            width=30.0,
            height=12.0,
            obstacles=(
                Obstacle(0.0, 30.0, 0.0, 2.0, "curb"),
                Obstacle(3.0, 8.0, 2.2, 4.4, "parked_car"),
                Obstacle(15.2, 20.2, 2.2, 4.4, "parked_car"),
            ),
            start=(14.5, 7.2, 0.0),
            goal=(10.4, 3.2, 0.0),
            planner=planner_overrides,
        )
    if name == "parking2":
        return Environment(
            name="parking2",
            title="Parallel-parking environment v2",
            width=30.0,
            height=12.0,
            obstacles=(
                Obstacle(0.0, 30.0, 0.0, 2.0, "curb"),
                Obstacle(3.0, 8.0, 2.2, 4.4, "parked_car"),
                Obstacle(15.2, 20.2, 2.2, 4.4, "parked_car"),
            ),
            start=(18.5, 6.2, 0.0),
            goal=(10.4, 3.3, 0.0),
            planner=planner_overrides,
        )
    if name == "parking4":
        return Environment(
            name="parking4",
            title="Parallel-parking environment v4",
            width=30.0,
            height=14.0,
            obstacles=(
                Obstacle(0.0, 30.0, 0.0, 2.0, "curb"),
                Obstacle(3.0, 8.9, 2.2, 9, "parked_car"),
                Obstacle(14.3, 20.2, 2.2, 9, "parked_car"),
            ),
            start=(18.5, 11, 0.0),
            goal=(10.4, 3.3, 0.0),
            planner=planner_overrides,
        )
    if name == "parking2_2":
        return Environment(
            name="parking2_2",
            title="Parallel-parking environment v2.2",
            width=30.0,
            height=12.0,
            obstacles=(
                Obstacle(0.0, 30.0, 0.0, 2.0, "curb"),
                Obstacle(3.6, 9.1, 2.2, 4.4, "parked_car"),
                Obstacle(14.1, 20.0, 2.2, 4.4, "parked_car"),
            ),
            start=(18.5, 6.2, 0.0),
            goal=(10.4, 3.3, 0.0),
            planner=planner_overrides,
        )
    if name == "parking3":
        return Environment(
            name="parking3",
            title="Parallel-parking environment v3",
            width=30.0,
            height=12.0,
            obstacles=(
                Obstacle(0.0, 30.0, 0.0, 2.0, "curb"),
                Obstacle(3.0, 8.0, 2.2, 4.4, "parked_car"),
                Obstacle(15.2, 20.2, 2.2, 4.4, "parked_car"),
            ),
            start=(5.0, 6.2, 0.0),
            goal=(10.4, 3.3, 0.0),
            planner=planner_overrides,
        )
    raise ValueError(f"Unknown environment: {name}")


def draw_environment(ax: plt.Axes, env: Environment) -> None:
    """Draw world bounds, obstacles, labels, axes, and grid.

    Args:
        ax: Matplotlib axes receiving the generated artists.
        env: Environment providing bounds, obstacles, and start/goal poses.

    Returns:
        None.
    """
    for obstacle in env.obstacles:
        hatch = "xx" if obstacle.kind == "parked_car" else "///"
        ax.add_patch(
            Rectangle(
                (obstacle.xmin, obstacle.ymin),
                obstacle.xmax - obstacle.xmin,
                obstacle.ymax - obstacle.ymin,
                alpha=0.45,
                hatch=hatch,
            )
        )
        if obstacle.kind == "parked_car":
            ax.text(
                (obstacle.xmin + obstacle.xmax) / 2.0,
                (obstacle.ymin + obstacle.ymax) / 2.0,
                "parked car",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set(xlim=(0.0, env.width), ylim=(0.0, env.height))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
    )


def draw_goal_region(ax: plt.Axes, planner: HybridAStar, env: Environment) -> None:
    """Draw the rear-axle position-tolerance disk around the goal.

    Args:
        ax: Matplotlib axes receiving the generated artists.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.

    Returns:
        None.
    """
    ax.add_patch(
        Circle(
            (env.goal[0], env.goal[1]),
            planner.position_tolerance,
            facecolor="tab:green",
            edgecolor="tab:green",
            alpha=0.16,
            linestyle="--",
            linewidth=1.5,
            label="Rear-axle position tolerance",
            zorder=2,
        )
    )


def draw_goal_pose(ax: plt.Axes, planner: HybridAStar, env: Environment) -> None:
    """Draw the nominal vehicle footprint at the exact goal pose.

    Args:
        ax: Matplotlib axes receiving the generated artists.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.

    Returns:
        None.
    """
    ax.add_patch(
        Polygon(
            vehicle_polygon(*env.goal, planner.vehicle),
            fill=False,
            linestyle="--",
            linewidth=2.0,
            label="Goal pose",
            zorder=4,
        )
    )


def draw_start_pose(
    ax: plt.Axes,
    planner: HybridAStar,
    env: Environment,
    *,
    show_vehicle: bool = True,
) -> None:
    """Draw the initial vehicle footprint and rear-axle marker.

    Args:
        ax: Matplotlib axes receiving the generated artists.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.
        show_vehicle: Whether to draw the vehicle footprint at the start pose.

    Returns:
        None.
    """
    if show_vehicle:
        ax.add_patch(
            Polygon(
                vehicle_polygon(*env.start, planner.vehicle),
                fill=False,
                linestyle="-.",
                linewidth=1.6,
                alpha=0.8,
                label="Start pose",
                zorder=4,
            )
        )
    ax.scatter([env.start[0]], [env.start[1]], s=32, marker="o", zorder=5)


def draw_scene_background(
    ax: plt.Axes,
    planner: HybridAStar,
    env: Environment,
    *,
    show_start_vehicle: bool = False,
) -> None:
    """Draw static scene elements shared by all visualizations.

    Args:
        ax: Matplotlib axes receiving the generated artists.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.
        show_start_vehicle: Whether to draw the static vehicle footprint at the start pose.
            Defaults to false so visualizations only show the start rear-axle marker.

    Returns:
        None.
    """
    draw_environment(ax, env)
    draw_goal_region(ax, planner, env)
    draw_goal_pose(ax, planner, env)
    draw_start_pose(ax, planner, env, show_vehicle=show_start_vehicle)


class LiveStateView:
    """Own and update artists for one score-colored search panel."""

    def __init__(
        self,
        ax: plt.Axes,
        planner: HybridAStar,
        env: Environment,
        trajectory_label: str,
        state_label: str,
        score_attribute: Literal["total_estimate", "heuristic"],
        score_label: str,
    ) -> None:
        """Create reusable artists for one live-search score panel.

        Args:
            ax: Matplotlib axes receiving the generated artists.
            planner: Planner providing geometry, tolerances, and completed search data.
            env: Environment providing bounds, obstacles, and start/goal poses.
            trajectory_label: Legend label for the selected node's reconstructed path.
            state_label: Legend label for the selected node's vehicle outline.
            score_attribute: SearchNodeState field used to color open states.
            score_label: Human-readable colorbar label for the selected score.

        Returns:
            None.
        """
        self.ax = ax
        self.planner = planner
        self.env = env
        self.score_attribute = score_attribute
        draw_scene_background(ax, planner, env, show_start_vehicle=False)

        # Only open states participate in the heatmap and its normalization.
        # Closed states remain visible as fixed gray dots beneath open states.
        self.score_norm = Normalize(vmin=0.0, vmax=1.0)
        self.score_mappable = ScalarMappable(norm=self.score_norm, cmap="viridis")
        self.score_mappable.set_array(np.empty(0, dtype=float))
        self.open_points = ax.scatter(
            [],
            [],
            c=np.empty(0, dtype=float),
            s=12,
            marker="o",
            cmap="viridis",
            norm=self.score_norm,
            linewidths=0.0,
            label="Open states",
            zorder=4,
        )
        self.closed_points = ax.scatter(
            [],
            [],
            color="0.55",
            s=12,
            marker="o",
            linewidths=0.0,
            label="Closed states",
            zorder=3,
        )
        # Per-node heading arrows are intentionally omitted because dense
        # searches become unreadable; the selected state's full vehicle remains.
        self.colorbar = ax.figure.colorbar(self.score_mappable, ax=ax, pad=0.02)
        self.colorbar.set_label(score_label)

        # The trajectory and full vehicle box identify the globally best state
        # according to this panel's score.
        (self.trajectory,) = ax.plot(
            [env.start[0]],
            [env.start[1]],
            linewidth=2.2,
            label=trajectory_label,
            zorder=5,
        )
        self.safety_envelope = Polygon(
            vehicle_polygon(*env.start, planner.vehicle, margin=planner.safety_margin),
            fill=False,
            linestyle=":",
            linewidth=2.2,
            alpha=0.0,
            label="Safety envelope",
            zorder=6,
        )
        self.car = Polygon(
            vehicle_polygon(*env.start, planner.vehicle),
            fill=False,
            linewidth=2.0,
            label=state_label,
            zorder=7,
        )
        self.tires = [
            Polygon(tire, closed=True, facecolor="black", edgecolor="black", zorder=8)
            for tire in vehicle_tire_polygons(*env.start, 0.0, planner.vehicle)
        ]
        arrow_start, arrow_tip = vehicle_heading_arrow(*env.start, planner.vehicle)
        self.heading_arrow = FancyArrowPatch(
            arrow_start,
            arrow_tip,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2.0,
            color="tab:blue",
            zorder=8,
        )
        ax.add_patch(self.safety_envelope)
        ax.add_patch(self.car)
        for tire in self.tires:
            ax.add_patch(tire)
        ax.add_patch(self.heading_arrow)

    def _score_arrays(
        self,
        states: tuple[SearchNodeState, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert state scores into finite display values and color limits.

        Args:
            states: Current best-known generated states, including open and closed.

        Returns:
            Display scores and two-element color limits.
        """
        raw_scores = np.asarray(
            [getattr(state, self.score_attribute) for state in states],
            dtype=float,
        )
        open_mask = np.asarray([not state.closed for state in states], dtype=bool)
        finite_scores = raw_scores[open_mask & np.isfinite(raw_scores)]
        if finite_scores.size == 0:
            return np.zeros_like(raw_scores), np.asarray([0.0, 1.0])

        minimum = float(finite_scores.min())
        maximum = float(finite_scores.max())
        display_scores = np.where(np.isfinite(raw_scores), raw_scores, maximum)
        if maximum - minimum < 1e-12:
            padding = max(0.5, abs(minimum) * 0.01)
            limits = np.asarray([minimum - padding, maximum + padding])
        else:
            limits = np.asarray([minimum, maximum])
        return display_scores, limits

    @staticmethod
    def _node_arrays(
        states: tuple[SearchNodeState, ...],
        scores: np.ndarray,
        *,
        closed: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract positions and scores for an open or closed subset.

        Args:
            states: Current best-known generated states, including open and closed.
            scores: Display scores aligned with ``states``.
            closed: Whether to select globally closed states or union-open states.

        Returns:
            The selected ``N x 2`` rear-axle positions and matching score array.
            Both arrays are empty when the subset has no states.
        """
        indices = [index for index, state in enumerate(states) if state.closed is closed]
        if not indices:
            return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
        points = np.asarray(
            [[states[index].node.x, states[index].node.y] for index in indices],
            dtype=float,
        )
        return points, scores[np.asarray(indices, dtype=int)]

    def update(
        self,
        expansion_count: int,
        snapshot: SearchSnapshot,
        states: tuple[SearchNodeState, ...],
        selection_name: str,
    ) -> None:
        """Refresh state populations and move the selected vehicle outline.

        Args:
            expansion_count: Number of action-set expansions performed so far.
            snapshot: Path and score information for the selected node.
            states: Current best-known generated states, including open and closed.
            selection_name: Title describing the panel's selection rule.

        Returns:
            None.
        """
        scores, limits = self._score_arrays(states)
        self.score_norm.vmin = float(limits[0])
        self.score_norm.vmax = float(limits[1])
        self.score_mappable.set_clim(*limits)
        self.colorbar.update_normal(self.score_mappable)

        open_points, open_scores = self._node_arrays(
            states,
            scores,
            closed=False,
        )
        closed_points = self._node_arrays(
            states,
            scores,
            closed=True,
        )[0]
        self.open_points.set_offsets(open_points)
        self.open_points.set_array(open_scores)
        self.closed_points.set_offsets(closed_points)

        node = snapshot.node
        self.trajectory.set_data(snapshot.path[:, 0], snapshot.path[:, 1])
        pose = (node.x, node.y, node.yaw)
        self.safety_envelope.set_xy(
            vehicle_polygon(*pose, self.planner.vehicle, margin=self.planner.safety_margin)
        )
        self.car.set_xy(vehicle_polygon(*pose, self.planner.vehicle))
        for tire, footprint in zip(
            self.tires,
            vehicle_tire_polygons(
                *pose,
                float(self.planner.steers[node.steer_index]),
                self.planner.vehicle,
            ),
        ):
            tire.set_xy(footprint)
        arrow_start, arrow_tip = vehicle_heading_arrow(*pose, self.planner.vehicle)
        self.heading_arrow.set_positions(arrow_start, arrow_tip)

        # Report both search scores, raw distance, and population counts.
        distance = math.hypot(node.x - self.env.goal[0], node.y - self.env.goal[1])
        yaw_error = abs(wrap(node.yaw - self.env.goal[2]))
        open_count = len(open_points)
        closed_count = len(closed_points)
        self.ax.set_title(
            f"{selection_name} expansion {expansion_count:} "
            f"[{self.planner.last_expansion_queue}]\n"
            f"g={node.cost:.2f}; h={snapshot.heuristic:.2f}; "
            f"g+weight*h={snapshot.total_estimate:.2f}; distance={distance:.2f} m\n"
            f"yaw err={math.degrees(yaw_error):.1f} deg "
            f"open={open_count:}; closed={closed_count:}",
        )


class LiveSearchPlot:
    """Compare the minimum open total score and minimum all-state heuristic."""

    def __init__(self, planner: HybridAStar, env: Environment) -> None:
        """Create synchronized minimum-open-total and minimum-all-state-h panels.

        Args:
            planner: Planner providing geometry, tolerances, and completed search data.
            env: Environment providing bounds, obstacles, and start/goal poses.

        Returns:
            None.
        """
        plt.ion()
        self.fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharex=True,
            sharey=True,
        )
        self.best_total = LiveStateView(
            axes[0],
            planner,
            env,
            trajectory_label="Minimum g+weight*h trajectory",
            state_label="Minimum g+weight*h state",
            score_attribute="total_estimate",
            score_label="g+weight*h",
        )
        self.best_heuristic = LiveStateView(
            axes[1],
            planner,
            env,
            trajectory_label="Minimum h trajectory",
            state_label="Minimum h state",
            score_attribute="heuristic",
            score_label="h",
        )
        self.fig.suptitle(env.title)
        self.fig.tight_layout()

        # Non-interactive backends such as Agg can render files but cannot display
        # or pause an on-screen window.
        self._interactive_canvas = self.fig.canvas.__class__.__module__ != "matplotlib.backends.backend_agg"
        if self._interactive_canvas:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def update(
        self,
        expansion_count: int,
        best_total: SearchSnapshot,
        best_heuristic: SearchSnapshot,
        states: tuple[SearchNodeState, ...],
    ) -> None:
        """Refresh both live-search panels from the current frontier.

        Args:
            expansion_count: Number of action-set expansions performed so far.
            best_total: Snapshot of the open node with minimum ``g+weight*h``.
            best_heuristic: Snapshot of the open or closed node with minimum ``h``.
            states: Current best-known generated states, including open and closed.

        Returns:
            None.
        """
        if not plt.fignum_exists(self.fig.number):
            return
        self.best_total.update(
            expansion_count,
            best_total,
            states,
            "Minimum g+weight*h open state",
        )
        self.best_heuristic.update(
            expansion_count,
            best_heuristic,
            states,
            "Minimum heuristic state (open and closed)",
        )
        if self._interactive_canvas:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)


def save_plot(output: Path, planner: HybridAStar, env: Environment, path: np.ndarray) -> None:
    """Render and save a static summary of the search and final path.

    Args:
        output: Destination path for the generated image.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.
        path: Sampled ``N x 3`` solution path.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_scene_background(ax, planner, env)
    if planner.expanded:
        expanded = np.asarray(planner.expanded)
        ax.scatter(expanded[:, 0], expanded[:, 1], s=4, alpha=0.12, label="Expanded")
    ax.plot(path[:, 0], path[:, 1], linewidth=2.2, label="Hybrid A* path")

    # Draw a bounded number of intermediate footprints to show orientation
    # without overcrowding the path plot.
    indices = np.linspace(0, len(path) - 1, min(14, len(path)), dtype=int)
    for index in indices:
        ax.add_patch(
            Polygon(
                vehicle_polygon(*path[index], planner.vehicle),
                fill=False,
                alpha=0.65,
            )
        )
    ax.add_patch(
        Polygon(
            vehicle_polygon(*path[-1], planner.vehicle, planner.safety_margin),
            fill=False,
            linestyle=":",
            linewidth=2.0,
            alpha=0.0,
            label="Safety envelope",
        )
    )
    arrow_start, arrow_tip = vehicle_heading_arrow(*path[-1], planner.vehicle)
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_tip,
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2.0,
            color="tab:blue",
            label="Vehicle front",
            zorder=7,
        )
    )
    position_error = math.hypot(path[-1, 0] - env.goal[0], path[-1, 1] - env.goal[1])
    yaw_error = math.degrees(abs(wrap(path[-1, 2] - env.goal[2])))
    ax.set_title(
        f"{env.title}\nHybrid A*; margin={planner.safety_margin:.2f} m; "
        f"terminal error={position_error:.3f} m / {yaw_error:.2f} deg"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_animation(
    output: Path,
    planner: HybridAStar,
    env: Environment,
    path: np.ndarray,
    directions: np.ndarray,
    steers: np.ndarray,
) -> None:
    """Render and save a GIF of the vehicle following the planned path.

    Args:
        output: Destination path for the generated GIF.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.
        path: Sampled ``N x 3`` solution path.
        directions: Per-sample travel directions aligned with ``path``.
        steers: Per-sample steering angles aligned with ``path``.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_scene_background(ax, planner, env)
    ax.plot(path[:, 0], path[:, 1], alpha=0.3)
    (trace,) = ax.plot([], [], linewidth=2.0)
    safety_envelope = Polygon(
        vehicle_polygon(*path[0], planner.vehicle, margin=planner.safety_margin),
        fill=False,
        linestyle=":",
        linewidth=2.2,
        alpha=0.0,
        zorder=5,
    )
    car = Polygon(
        vehicle_polygon(*path[0], planner.vehicle),
        fill=False,
        linewidth=2.0,
        zorder=6,
    )
    tires = [
        Polygon(tire, closed=True, facecolor="black", edgecolor="black", zorder=7)
        for tire in vehicle_tire_polygons(*path[0], steers[0], planner.vehicle)
    ]
    arrow_start, arrow_tip = vehicle_heading_arrow(*path[0], planner.vehicle)
    heading_arrow = FancyArrowPatch(
        arrow_start,
        arrow_tip,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=2.0,
        color="tab:blue",
        zorder=7,
    )
    ax.add_patch(safety_envelope)
    ax.add_patch(car)
    for tire in tires:
        ax.add_patch(tire)
    ax.add_patch(heading_arrow)

    # Render every reconstructed sample so each motion primitive is visible.
    frames = np.arange(len(path), dtype=int)
    frames = np.append(frames, [len(path) - 1] * 4)

    def update(frame_number: int):
        """Move animation artists to one selected path sample.

        Args:
            frame_number: Index into the reduced animation-frame sequence.

        Returns:
            The modified Matplotlib artists for the frame.
        """
        index = int(frames[frame_number])
        trace.set_data(path[: index + 1, 0], path[: index + 1, 1])
        safety_envelope.set_xy(
            vehicle_polygon(
                *path[index],
                planner.vehicle,
                margin=planner.safety_margin,
            )
        )
        car.set_xy(vehicle_polygon(*path[index], planner.vehicle))
        for tire, footprint in zip(
            tires,
            vehicle_tire_polygons(*path[index], steers[index], planner.vehicle),
        ):
            tire.set_xy(footprint)
        arrow_start, arrow_tip = vehicle_heading_arrow(*path[index], planner.vehicle)
        heading_arrow.set_positions(arrow_start, arrow_tip)
        gear = "forward" if directions[index] > 0 else "reverse"
        ax.set_title(
            f"{env.title} — {gear}; margin={planner.safety_margin:.2f} m "
            f"({frame_number + 1}/{len(frames)})"
        )
        return trace, safety_envelope, car, *tires, heading_arrow

    animation = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    animation.save(output, writer=PillowWriter(fps=12))
    plt.close(fig)


def positive_float(value: str) -> float:
    """Parse an argparse value as a finite strictly positive float.

    Args:
        value: Command-line token to parse and validate.

    Returns:
        The parsed positive floating-point value.

    Raises:
        ValueError: If conversion to float fails.
        argparse.ArgumentTypeError: If the value is non-finite or not positive.
    """
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    """Parse an argparse value as a finite non-negative float.

    Args:
        value: Command-line token to parse and validate.

    Returns:
        The parsed non-negative floating-point value.

    Raises:
        ValueError: If conversion to float fails.
        argparse.ArgumentTypeError: If the value is non-finite or negative.
    """
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return number


def nonnegative_int(value: str) -> int:
    """Parse an argparse value as a non-negative integer.

    Args:
        value: Command-line token to parse and validate.

    Returns:
        The parsed non-negative integer.

    Raises:
        ValueError: If conversion to int fails.
        argparse.ArgumentTypeError: If the value is negative.
    """
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def parse_args() -> argparse.Namespace:
    """Define the command-line interface and validate parsed arguments.

    Returns:
        The populated argparse Namespace.

    Raises:
        SystemExit: When argparse rejects command-line input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=("walls", "parking", "parking2", "parking2_2", "parking3", "parking4"),
        default="parking",
    )
    parser.add_argument(
        "--safety_margin",
        type=nonnegative_float,
        default=0.20,
        help="Clearance added on every side of the ego vehicle [m].",
    )
    parser.add_argument(
        "--integration_step",
        type=positive_float,
        default=0.1,
        help="Spacing of reconstructed path/animation samples [m].",
    )
    parser.add_argument(
        "--collision_check_step",
        type=positive_float,
        default=0.05,
        help="Independent swept-path collision sampling interval [m].",
    )
    parser.add_argument(
        "--xy_resolution",
        type=positive_float,
        default=0.15,
        help="Hybrid A* x/y grid resolution [m]",
    )
    parser.add_argument(
        "--yaw_resolution_deg",
        type=positive_float,
        default=1.0,
        help="Hybrid A* heading grid resolution [deg]",
    )
    parser.add_argument(
        "--primitive_length",
        type=positive_float,
        default=0.2,
        help="Length of each motion primitive [m]",
    )
    parser.add_argument(
        "--two_queues",
        action="store_true",
        help=(
            "Enable the coarse acceleration queue. Without this flag, use the " "original fine-only search."
        ),
    )
    parser.add_argument(
        "--coarse_primitive_mult",
        type=int,
        default=4,
        help=(
            "Integer multiplier for the coarse queue primitive length. The length is "
            "coarse_primitive_mult * primitive_length. The coarse queue always "
            "uses 3 steering values: -max, 0, +max."
        ),
    )
    parser.add_argument(
        "--queue_beta",
        type=positive_float,
        default=1.5,
        help="Expand coarse when min_coarse <= beta * min_fine; beta must be at least 1.",
    )
    parser.add_argument(
        "--origin_priority_factor",
        type=positive_float,
        default=2.0,
        help=(
            "Multiplicative preference for a node's generating queue: divide "
            "its home-queue priority by this factor and multiply its other-queue "
            "priority by this factor; 1 disables the preference."
        ),
    )
    parser.add_argument(
        "--position_tolerance",
        type=positive_float,
        default=0.2,
        help="Allowed terminal position error [m]",
    )
    parser.add_argument(
        "--yaw_tolerance_deg",
        type=positive_float,
        default=1.5,
        help="Allowed terminal heading error [deg]",
    )
    parser.add_argument(
        "--reverse_multiplier",
        type=positive_float,
        default=1.0,
        help="Cost multiplier for reverse motion",
    )
    parser.add_argument(
        "--gear_change_penalty",
        type=nonnegative_float,
        default=0.0,
        help="Additional cost when changing direction",
    )
    parser.add_argument(
        "--steering_change_penalty",
        type=nonnegative_float,
        default=0.0,
        help="Steering-change event cost on the original fine-edge scale",
    )
    parser.add_argument(
        "--state_key_mode",
        choices=("pose", "pose_control"),
        default="pose",
        help=(
            "Use a legacy compact pose key, or include incoming direction and "
            "steering. pose_control is recommended when change penalties are nonzero."
        ),
    )
    parser.add_argument(
        "--heuristic",
        choices=("distance", "default", "defaultw1", "tolerance", "dijkstra"),
        default="default",
        help=(
            "Heuristic used for queue priorities: distance-only; legacy distance plus "
            "heading; tolerance-aware kinematic lower bound; or the maximum of "
            "that bound and a relaxed 2D obstacle-grid estimate."
        ),
    )
    parser.add_argument(
        "--heuristic_weight",
        type=nonnegative_float,
        default=1.0,
        help="Multiplier for h in total priority g+weight*h.",
    )
    parser.add_argument(
        "--coarse_heuristic_weight",
        type=nonnegative_float,
        default=None,
        help=("Optional coarse-queue multiplier for h in g+weight*h; by default, " "use --heuristic_weight."),
    )
    parser.add_argument(
        "--post_goal_expansions",
        type=nonnegative_int,
        default=0,
        help=(
            "Fallback after the first accepted goal when the admissible cost bound is "
            "disabled, including when --enable_admissible_bound is omitted or two "
            "queues are enabled. Fine-only search with the bound enabled instead "
            "stops when its admissible lower-bound certificate is met."
        ),
    )
    parser.add_argument(
        "--enable_admissible_bound",
        action="store_true",
        help=(
            "For fine-only cost search, stop only once an admissible lower-bound "
            "certificate proves the incumbent goal cost optimal."
        ),
    )
    parser.add_argument("--max_expansions", type=int, default=1_000_000)
    parser.add_argument(
        "--max_consecutive_coarse_expansions",
        type=nonnegative_int,
        default=10,
        help=(
            "Maximum consecutive coarse action-set expansions while a fine state "
            "remains available; 0 disables coarse selection in that situation."
        ),
    )
    parser.add_argument(
        "--live_plot_every",
        type=nonnegative_int,
        default=100000,
        help=(
            "Update a two-panel interactive plot every X action-set expansions showing "
            "open states as score heatmaps and closed states as gray dots, with "
            "full vehicle boxes on the open-set minimum-g+weight*h state and the "
            "minimum-h state across open and closed states; "
            "0 disables live plotting."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--no_animation",
        action="store_true",
        help="Skip PNG and GIF output generation",
    )
    args = parser.parse_args()
    if args.max_expansions <= 0:
        parser.error("--max_expansions must be greater than zero")
    if args.two_queues and args.coarse_primitive_mult < 1:
        parser.error("--coarse_primitive_mult must be at least 1")
    if args.queue_beta < 1.0:
        parser.error("--queue_beta must be at least 1.0")
    if args.origin_priority_factor < 1.0:
        parser.error("--origin_priority_factor must be at least 1.0")
    return args


def main(args: argparse.Namespace) -> dict[str, object]:
    """Run the selected scenario and write visual outputs and metrics.

    Args:
        args: Complete planner configuration, normally produced by ``parse_args``.

    Returns:
        Metrics and output paths for the completed planning run.

    Raises:
        ValueError: If start or goal is invalid.
        RuntimeError: If planning fails within the expansion limit.
    """
    # Collect the complete planner configuration from the supplied arguments.
    planner_options: dict[str, float] = {
        "xy_resolution": args.xy_resolution,
        "primitive_length": args.primitive_length,
        "position_tolerance": args.position_tolerance,
        "reverse_multiplier": args.reverse_multiplier,
        "gear_change_penalty": args.gear_change_penalty,
        "steering_change_penalty": args.steering_change_penalty,
    }

    # Convert degree-based CLI values to the radians used internally.
    planner_options["yaw_resolution"] = math.radians(args.yaw_resolution_deg)
    planner_options["yaw_tolerance"] = math.radians(args.yaw_tolerance_deg)
    env = make_environment(args.env, planner_options)

    # Construct the planner and optional live visualization, then execute search.
    planner = HybridAStar(
        environment=env,
        vehicle=Vehicle(),
        safety_margin=args.safety_margin,
        integration_step=args.integration_step,
        collision_check_step=args.collision_check_step,
        heuristic_mode=args.heuristic,
        state_key_mode=args.state_key_mode,
        heuristic_weight=args.heuristic_weight,
        use_two_queues=args.two_queues,
        coarse_heuristic_weight=args.coarse_heuristic_weight,
        coarse_primitive_mult=args.coarse_primitive_mult,
        queue_beta=args.queue_beta,
        origin_priority_factor=args.origin_priority_factor,
    )
    live_plot = LiveSearchPlot(planner, env) if args.live_plot_every > 0 else None
    progress_bar = tqdm(
        total=args.max_expansions,
        desc="Expansions",
        unit="action-set",
        dynamic_ncols=True,
    )

    def update_progress(count: int) -> None:
        """Advance the progress bar to the reported expansion count.

        Args:
            count: Current total number of fine-plus-coarse action-set expansions.

        Returns:
            None.
        """
        progress_bar.update(count - progress_bar.n)

    try:
        path, directions, steers, terminal = planner.plan(
            env.start,
            env.goal,
            args.max_expansions,
            live_plot_every=args.live_plot_every,
            progress_callback=live_plot.update if live_plot is not None else None,
            expansion_callback=update_progress,
            post_goal_expansions=args.post_goal_expansions,
            enable_admissible_bound=args.enable_admissible_bound,
            max_consecutive_coarse_expansions=args.max_consecutive_coarse_expansions,
        )
    finally:
        # Keep the progress display consistent even when planning raises.
        progress_bar.update(planner.expansion_count - progress_bar.n)
        progress_bar.close()

    # Create output paths and save visual outputs unless explicitly disabled.
    env_output_dir = args.output_dir / env.name
    env_output_dir.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now().astimezone()
    timestamp = run_time.strftime("%Y_%m_%d_%H_%M_%S")
    output_stem = f"{planner.heuristic_mode}_{timestamp}"
    plot_path = env_output_dir / f"{output_stem}_path.png"
    animation_path = env_output_dir / f"{output_stem}_animation.gif"
    if not args.no_animation:
        # Live plotting enables pyplot's interactive mode. Temporarily disable it so
        # the save-only PNG and GIF figures are rendered without flashing extra GUI
        # windows; the existing live-search figure remains open.
        with plt.ioff():
            save_plot(plot_path, planner, env, path)
            save_animation(animation_path, planner, env, path, directions, steers)

    # Compute exact primitive-arc length and the reconstructed sampled-polyline length.
    sampled_path_length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
    exact_path_length = planner.exact_path_length(terminal)
    position_error = math.hypot(path[-1, 0] - env.goal[0], path[-1, 1] - env.goal[1])
    yaw_error = math.degrees(abs(wrap(path[-1, 2] - env.goal[2])))
    result_path = env_output_dir / f"{output_stem}.json"
    arguments = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    result = {
        "environment": env.name,
        "timestamp": run_time.isoformat(),
        "arguments": arguments,
        "action_set_expansions": planner.expansion_count,
        "unique_expanded_state_keys": planner.unique_expanded_state_count,
        "fine_expansions": planner.fine_expansion_count,
        "coarse_expansions": planner.coarse_expansion_count,
        "path_samples": len(path),
        "path_length_m": exact_path_length,
        "sampled_path_length_m": sampled_path_length,
        "search_cost": terminal.cost,
        "terminal_error_m": position_error,
        "terminal_error_deg": yaw_error,
    }
    if not args.no_animation:
        result["plot"] = str(plot_path.resolve())
        result["animation"] = str(animation_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if __name__ == "__main__":
        # Print the run configuration, search outcome, and generated file locations.
        print(f"Environment: {env.name}")
        print(f"Action-set expansions: {planner.expansion_count}")
        print(f"Unique expanded state keys: {planner.unique_expanded_state_count}")
        print(f"Two queues: {planner.use_two_queues}")
        print(f"Fine expansions: {planner.fine_expansion_count}")
        print(f"Coarse expansions: {planner.coarse_expansion_count}")
        print(f"Fine primitive length: {planner.fine_primitive_length:.3f} m")
        print(f"Coarse primitive length: {planner.coarse_primitive_length:.3f} m")
        print(f"Queue beta: {planner.queue_beta:.3f}")
        print(f"Origin priority factor: {planner.origin_priority_factor:.3f}")
        print(f"Path samples: {len(path)}")
        print(f"Path length: {exact_path_length:.6f} m")
        print(f"Sampled polyline length: {sampled_path_length:.6f} m")
        print(f"Search cost: {terminal.cost:.2f}")
        print(f"Terminal error: {position_error:.3f} m / {yaw_error:.2f} deg")
        print(f"Safety margin: {planner.safety_margin:.3f} m")
        print(f"Integration step: {planner.integration_step:.3f} m")
        print(f"Collision-check step: {planner.collision_check_step:.3f} m")
        print(f"Heuristic: {planner.heuristic_mode}")
        print(f"Heuristic weight: {planner.heuristic_weight:.3f}")
        print(f"Coarse heuristic weight: {planner.coarse_heuristic_weight:.3f}")
        print(f"State key mode: {planner.state_key_mode}")
        print(f"Post-goal action-set expansions: {args.post_goal_expansions}")
        print(f"Enable admissible bound: {args.enable_admissible_bound}")
        print(f"Max consecutive coarse expansions: {args.max_consecutive_coarse_expansions}")
        print(f"Live plot interval: {args.live_plot_every} action-set expansions")
        if not args.no_animation:
            print(f"Plot: {plot_path.resolve()}")
            print(f"Animation: {animation_path.resolve()}")
        print(f"Results JSON: {result_path.resolve()}")
    return result


if __name__ == "__main__":
    main(parse_args())
