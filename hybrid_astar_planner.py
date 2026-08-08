#!/usr/bin/env python3
"""Core geometry, environments, and single-queue Hybrid A* search.

The planner keeps continuous rear-axle poses connected by exact, constant-
steering bicycle-model arcs. It discretizes those poses only to form finite
OPEN/CLOSED keys; the generated motion itself is never snapped to the grid.
For each key, the lowest-cost continuous representative found so far replaces
any more expensive representative.

For each expanded nonterminal state, the single-queue search attempts forward
and reverse primitives at five steering angles. Each primitive has a fixed arc
length, is sampled independently for collision checking and path reconstruction,
and is discarded if a collision sample or corridor sample is invalid. Queue
ordering uses the configured ``g + weight * h`` score. Search stops after the
configured post-goal expansion budget, when OPEN is exhausted, or at the global
expansion limit.

``TwoQueueHybridAStar`` builds on these shared mechanics in
``hybrid_astar_two_queues.py``; coarse corridor construction lives in
``hybrid_astar_corridor.py``.
"""

import heapq
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Literal, Optional

import numpy as np
from numba import njit

from hybrid_astar_corridor import CoarsePathCorridor, _arc_is_inside_corridor


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
        x0: Rear-axle x-coordinate at the start of the primitive, in meters.
        y0: Rear-axle y-coordinate at the start of the primitive, in meters.
        yaw0: Vehicle heading at the start of the primitive, in radians.
        direction: Travel direction: ``1`` for forward or ``-1`` for reverse.
        steer: Constant front-wheel steering angle, in radians.
        wheelbase: Distance between the front and rear axles, in meters.
        collision_distances: Arc-length samples used for swept collision checks.
        half_length: Half of the safety-inflated vehicle length, in meters.
        half_width: Half of the safety-inflated vehicle width, in meters.
        center_offset: Longitudinal offset from the rear axle to the box center.
        world_width: Width of the rectangular planning world, in meters.
        world_height: Height of the rectangular planning world, in meters.
        obstacle_boxes: Obstacle rows formatted as ``[xmin, xmax, ymin, ymax]``.

    Returns:
        A ``(collision_free, x, y, yaw)`` tuple. The pose is the primitive
        endpoint when collision-free and the input pose after a collision.
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

    wheelbase: float = 2.6  # Distance between rear and front axles, in meters.
    length: float = 4.4  # Overall body length, in meters.
    width: float = 1.8  # Overall body width, in meters.
    rear_overhang: float = 1.0  # Distance from rear bumper to rear axle, in meters.
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
    """Describe a complete planning scene and its planner configuration."""

    name: str  # Stable scene identifier.
    title: str  # Human-readable figure title.
    width: float  # World width, in meters.
    height: float  # World height in meters.
    obstacles: tuple[Obstacle, ...]  # Axis-aligned obstacles in the scene.
    start: tuple[float, float, float]  # Initial rear-axle pose.
    goal: tuple[float, float, float]  # Nominal rear-axle goal pose.
    planner: dict[str, float] = field(default_factory=dict)  # Planner configuration.


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

    Returns:
        None.

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
    """Validate the required planner configuration values.

    Args:
        options: Complete planner configuration supplied by an environment.

    Returns:
        None.

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
) -> None:
    """Validate public fine-search control arguments.

    Args:
        max_expansions: Maximum number of action-set expansions.
        live_plot_every: Live-search update interval; zero disables updates.
        post_goal_expansions: Additional expansions after the first accepted goal.

    Returns:
        None.

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


# Search bookkeeping identifies a state by discretized rear-axle pose, optionally
# augmented with the incoming direction and steering index. Direction and previous
# steering affect gear-change and steering-change costs, so ``pose_control`` is the
# Markov representation for this objective. ``pose`` is retained only as the legacy
# compact mode; it can merge arrivals with different continuation costs.
PoseKey = tuple[int, int, int]
StateKey = tuple[int, ...]
StateKeyMode = Literal["pose", "pose_control"]
HeuristicMode = Literal[
    "distance",
    "default",
    "defaultw1",
    "tolerance",
    "dijkstra",
]


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
    total_estimate: float  # Displayed main-queue priority.


@dataclass(frozen=True)
class SearchNodeState:
    """Store the best node and display scores for one state key."""

    node: Node  # Selected/current search node.
    heuristic: float  # Heuristic score h.
    total_estimate: float  # Displayed main-queue priority.
    closed: bool  # Whether the state is closed in all enabled queue views.


# Heap entries store the total priority, tie breaks, state key, and exact Node
# object. Object identity rejects stale heap entries.
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
        x: Rear-axle or grid-point x-coordinate in world meters.
        y: Rear-axle or grid-point y-coordinate in world meters.
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
        x: Rear-axle or grid-point x-coordinate in world meters.
        y: Rear-axle or grid-point y-coordinate in world meters.
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
        x: Rear-axle or grid-point x-coordinate in world meters.
        y: Rear-axle or grid-point y-coordinate in world meters.
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
        length: Total segment length in meters.
        step: Maximum nominal spacing between samples, in meters.

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
        x0: Rear-axle x-coordinate at the start of the primitive, in meters.
        y0: Rear-axle y-coordinate at the start of the primitive, in meters.
        yaw0: Vehicle heading at the start of the primitive, in radians.
        signed_distance: Travel distance in meters; negative values mean reverse.
        steer: Constant front-wheel steering angle, in radians.
        wheelbase: Distance between the front and rear axles, in meters.

    Returns:
        The terminal ``(x, y, yaw)`` pose with wrapped yaw.
    """
    curvature = math.tan(steer) / wheelbase
    if abs(curvature) < 1e-12:
        # Zero steering is the limiting straight-line case.
        return (
            x0 + signed_distance * math.cos(yaw0),
            y0 + signed_distance * math.sin(yaw0),
            wrap(yaw0),
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
        corridor_width: Optional[float] = None,
        coarse_resolution: float = 1.0,
    ) -> None:
        """Configure planner geometry, discretization, costs, and search ordering.

        Args:
            environment: Planning bounds, obstacles, poses, and planner configuration.
            vehicle: Vehicle dimensions, wheelbase, and steering limits.
            safety_margin: Clearance added to every side during collision checking.
            integration_step: Spacing of samples retained for path output and animation.
            collision_check_step: Independent spacing of swept-path collision samples.
            heuristic_mode: Heuristic name: ``distance``, ``default``,
                ``defaultw1``, ``tolerance``, or ``dijkstra``.
            state_key_mode: ``pose_control`` for a Markov control-aware state, or
                ``pose`` for the legacy pose-only state.
            heuristic_weight: Non-negative multiplier applied to ``h`` for total priority.
            corridor_width: Optional rear-axle radius around a coarse 2D A* path.
            coarse_resolution: Grid spacing used by coarse 2D A* in meters.

        Returns:
            None.

        Raises:
            ValueError: If any environment, vehicle, planner, sampling, or corridor
                configuration is invalid.
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
        if heuristic_mode not in {
            "distance",
            "default",
            "defaultw1",
            "tolerance",
            "dijkstra",
        }:
            raise ValueError(
                "heuristic_mode must be 'distance', 'default', 'defaultw1', "
                "'tolerance', or 'dijkstra'"
            )
        if state_key_mode not in {"pose", "pose_control"}:
            raise ValueError("state_key_mode must be 'pose' or 'pose_control'")
        if not math.isfinite(heuristic_weight) or heuristic_weight < 0.0:
            raise ValueError("heuristic_weight must be a finite non-negative number")
        self.heuristic_mode = heuristic_mode
        self.state_key_mode = state_key_mode
        self.heuristic_weight = heuristic_weight
        if corridor_width is not None and (not math.isfinite(corridor_width) or corridor_width <= 0.0):
            raise ValueError("corridor_width must be a finite positive number or None")
        if not math.isfinite(coarse_resolution) or coarse_resolution <= 0.0:
            raise ValueError("coarse_resolution must be a finite positive number")
        self.corridor_width = corridor_width
        self.corridor_grid_resolution = coarse_resolution

        # Read the complete planner configuration stored with the environment.
        options = environment.planner
        validate_planner_options(options)

        self.xy_resolution = options["xy_resolution"]
        requested_yaw_resolution = options["yaw_resolution"]

        self.yaw_bin_count = max(1, int(round(math.tau / requested_yaw_resolution)))
        # Use an exact divisor of one revolution so every cyclic yaw bin has the
        # same width, including the bin crossing the -pi/+pi boundary.
        self.yaw_resolution = math.tau / self.yaw_bin_count
        self.primitive_length = options["primitive_length"]
        if self.primitive_length < self.xy_resolution:
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
        self.steer_indices = tuple(range(len(self.steers)))

        # Precompute collision and reconstruction samples so the search does not
        # repeatedly allocate identical arrays for every generated primitive.
        self._collision_distances = np.asarray(
            sample_distances(self.primitive_length, collision_check_step), dtype=float
        )
        self._integration_distances_by_length = {
            self.primitive_length: np.asarray(
                sample_distances(self.primitive_length, integration_step), dtype=float
            )
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
        self.unique_expanded_state_count = 0
        self._dijkstra_cost_to_goal: Optional[np.ndarray] = None
        self.corridor: Optional[CoarsePathCorridor] = None
        self._corridor_path = np.empty((0, 2), dtype=float)
        self._corridor_radius_squared = -1.0

    def _prepare_corridor(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> None:
        """Build the optional coarse path corridor for one search.

        Args:
            start: Validated exact start rear-axle pose.
            goal: Validated exact goal rear-axle pose.

        Returns:
            None.
        """
        self.corridor = None
        self._corridor_path = np.empty((0, 2), dtype=float)
        self._corridor_radius_squared = -1.0
        if self.corridor_width is None:
            return
        corridor = CoarsePathCorridor(
            self.environment.width,
            self.environment.height,
            tuple(
                (obstacle.xmin, obstacle.xmax, obstacle.ymin, obstacle.ymax)
                for obstacle in self.environment.obstacles
            ),
            self.corridor_grid_resolution,
            self.corridor_width,
            obstacle_clearance=self._half_width,
        )
        corridor.build(start[:2], goal[:2])
        self.corridor = corridor
        self._corridor_path = corridor.coarse_path
        self._corridor_radius_squared = corridor.corridor_width**2

    def key(self, x: float, y: float, yaw: float) -> PoseKey:
        """Quantize a continuous pose into geometric lattice coordinates.

        The yaw index is cyclic, so equivalent orientations on opposite sides of
        the ``-pi``/``+pi`` boundary share one state key.

        Args:
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.
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
            and the node's discrete steering index in control-aware mode.
        """
        return self._state_key_from_values(
            node.x,
            node.y,
            node.yaw,
            node.direction,
            node.steer_index,
            initial=initial,
        )

    def _state_key_from_values(
        self,
        x: float,
        y: float,
        yaw: float,
        direction: int,
        steer_index: int,
        *,
        initial: bool = False,
    ) -> StateKey:
        """Return a state key without requiring a temporary node allocation.

        Args:
            x: Rear-axle x-coordinate.
            y: Rear-axle y-coordinate.
            yaw: Vehicle heading in radians.
            direction: Incoming travel direction.
            steer_index: Incoming discrete steering index.
            initial: Whether the state is the control-free start state.

        Returns:
            A pose-only key or a pose-and-incoming-control key.
        """
        pose_key = self.key(x, y, yaw)
        if self.state_key_mode == "pose":
            return pose_key
        key_direction = 0 if initial else int(direction)
        return (*pose_key, key_direction, steer_index)

    def collides(self, x: float, y: float, yaw: float) -> bool:
        """Test the safety-inflated vehicle box against bounds and obstacles.

        Args:
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.
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
        """Return the cheapest configured translational cost for one meter.

        Returns:
            The minimum forward/reverse per-meter multiplier.
        """
        return min(1.0, self.reverse_multiplier)

    def _tolerance_aware_lower_bound(self, x: float, y: float, yaw: float) -> float:
        """Lower-bound the cost needed to enter the accepted goal tolerances.

        Args:
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.
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
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.

        Returns:
            ``True`` when the point lies inside or on the boundary of an obstacle.
        """
        # This is deliberately a point-robot relaxation: it ignores the ego footprint,
        # safety margin, heading, steering, and corner-cutting constraints.
        return any(
            obstacle.xmin <= x <= obstacle.xmax and obstacle.ymin <= y <= obstacle.ymax
            for obstacle in self.environment.obstacles
        )

    def _build_2d_dijkstra_to_goal_region(self) -> np.ndarray:
        """Build an eight-connected cost-to-go grid from the goal region.

        When corridor search is active, only grid vertices inside the built
        corridor participate in the relaxed search. The returned array keeps its
        world-sized shape for coordinate lookup and plotting, with vertices outside
        the corridor left at infinity.

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
        corridor = self.corridor
        blocked = np.ones((ny, nx), dtype=bool) if corridor is not None else np.zeros((ny, nx), dtype=bool)

        # Restrict corridor-mode rasterization to the centerline's radius-expanded
        # bounding box. Exact geometric containment below removes the remaining
        # cells in the bounding-box corners.
        x_indices = range(nx)
        y_indices = range(ny)
        if corridor is not None:
            radius = corridor.corridor_width
            minimum_x = max(
                0,
                int(math.floor((np.min(corridor.coarse_path[:, 0]) - radius) / resolution)),
            )
            maximum_x = min(
                nx - 1,
                int(math.ceil((np.max(corridor.coarse_path[:, 0]) + radius) / resolution)),
            )
            minimum_y = max(
                0,
                int(math.floor((np.min(corridor.coarse_path[:, 1]) - radius) / resolution)),
            )
            maximum_y = min(
                ny - 1,
                int(math.ceil((np.max(corridor.coarse_path[:, 1]) + radius) / resolution)),
            )
            x_indices = range(minimum_x, maximum_x + 1)
            y_indices = range(minimum_y, maximum_y + 1)

        # Rasterize closed obstacles onto the point-robot grid so walls touching a
        # world boundary cannot create a zero-width route along their shared edge.
        # In corridor mode, out-of-corridor cells remain blocked without requiring
        # obstacle checks across the rest of the world.
        for iy in y_indices:
            y = iy * resolution
            for ix in x_indices:
                x = ix * resolution
                blocked[iy, ix] = self._point_is_inside_obstacle(x, y) or (
                    corridor is not None and not corridor.contains(x, y)
                )

        # Multi-source Dijkstra starts from every free cell inside the accepted
        # positional goal tolerance.
        gx, gy, _ = self.goal
        queue: list[tuple[float, int, int]] = []
        for iy in y_indices:
            y = iy * resolution
            for ix in x_indices:
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
            for iy in y_indices:
                y = iy * resolution
                for ix in x_indices:
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
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.

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
            x: Rear-axle or grid-point x-coordinate in world meters.
            y: Rear-axle or grid-point y-coordinate in world meters.
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
            # The obstacle-aware queue score combines the kinematic lower bound with
            # a raster-grid estimate. The grid estimate guides ordering but is not an
            # admissible continuous-path bound because eight-connected movement can
            # exceed the corresponding straight-line distance.
            obstacle_distance = self._dijkstra_distance_to_goal_region(x, y)
            if not math.isfinite(obstacle_distance):
                # The relaxed raster grid may report disconnection because of coarse
                # discretization even when the continuous search may still succeed.
                # Keep the state eligible by reverting to the finite kinematic estimate.
                return tolerance_lower_bound
            obstacle_grid_estimate = self._minimum_cost_per_metre() * obstacle_distance
            return max(tolerance_lower_bound, obstacle_grid_estimate)

        raise AssertionError(f"Unhandled validated heuristic mode: {self.heuristic_mode}")

    def check_primitive(
        self,
        node: Node,
        direction: int,
        steer_index: int,
        collision_distances: np.ndarray,
    ) -> Optional[tuple[float, float, float]]:
        """Return the endpoint of one valid motion primitive.

        Args:
            node: Search node supplying the primitive's initial pose.
            direction: Travel direction: ``1`` for forward or ``-1`` for reverse.
            steer_index: Index of the constant front-wheel steering angle.
            collision_distances: Positive arc-length samples ending at the exact
                primitive endpoint.

        Returns:
            The terminal ``(x, y, yaw)`` pose, or ``None`` if any swept sample
            collides with an obstacle or the world boundary, or if the primitive
            leaves the optional path corridor.
        """
        steer = float(self.steers[steer_index])
        collision_free, x, y, yaw = _sample_collision_free_primitive(
            node.x,
            node.y,
            node.yaw,
            direction,
            steer,
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
        if self._corridor_radius_squared >= 0.0 and not _arc_is_inside_corridor(
            node.x,
            node.y,
            node.yaw,
            direction,
            steer,
            self.vehicle.wheelbase,
            collision_distances,
            self._corridor_path,
            self._corridor_radius_squared,
        ):
            return None
        return x, y, yaw

    def _successor_cost(
        self,
        node: Node,
        direction: int,
        steer_index: int,
        primitive_length: float,
    ) -> float:
        """Calculate the successor cost for a selected motion primitive.

        Travel cost scales with edge length. Steering and gear changes are event
        costs. The configured primitive length preserves the original event-cost
        scale even when a derived planner uses a different edge length.

        Args:
            node: Parent node supplying accumulated cost and incoming control.
            direction: Successor travel direction: ``1`` forward or ``-1`` reverse.
            steer_index: Index of the successor steering angle.
            primitive_length: Length of the selected edge [m].

        Returns:
            Accumulated path cost at the successor.
        """
        steer = float(self.steers[steer_index])
        previous_steer = float(self.steers[node.steer_index])
        reverse = self.reverse_multiplier if direction < 0 else 1.0
        travel_cost = primitive_length * reverse
        steering_change_cost = (
            self.primitive_length
            * self.steering_change_penalty
            * abs(steer - previous_steer)
            / self.vehicle.max_steer
        )
        gear_change_cost = (
            self.gear_change_penalty if node.parent is not None and direction != node.direction else 0.0
        )
        return node.cost + travel_cost + steering_change_cost + gear_change_cost

    def _primitive_actions(self) -> tuple[tuple[float, np.ndarray], ...]:
        """Return edge-length and collision-sample specifications for expansion.

        Returns:
            The standard planner's single edge specification. ``plan`` combines
            it with both directions and every steering index.
        """
        return ((self.primitive_length, self._collision_distances),)

    def _main_queue_priority(self, node: Node, heuristic: float) -> float:
        """Return the priority stored in the main OPEN heap.

        Args:
            node: Node whose incoming action may affect search ordering.
            heuristic: Heuristic value for ``node``.

        Returns:
            The main-queue priority for ``node``.
        """
        return node.cost + self.heuristic_weight * heuristic

    def _main_queue_priority_label(self) -> str:
        """Return a human-readable label for the main OPEN priority.

        Returns:
            The score label used by live-search visualization.
        """
        return "g+weight*h"

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
            Exact path length in meters.
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
        key, queued_node = entry[-2:]
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
                key, node = entry[-2:]
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
            total_estimate=self._main_queue_priority(node, heuristic),
        )

    def _publish_search_state(
        self,
        closed: set[StateKey],
        nodes: dict[StateKey, Node],
        progress_callback: ProgressCallback,
    ) -> None:
        """Publish OPEN/CLOSED state and selected search snapshots.

        Args:
            closed: State keys expanded with the planner's action set.
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
                    total_estimate=self._main_queue_priority(node, heuristic),
                    closed=key in closed,
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

        Returns:
            None.
        """
        exact_path_length = self.exact_path_length(node)
        position_error = math.hypot(node.x - goal[0], node.y - goal[1])
        yaw_error = abs(wrap(node.yaw - goal[2]))
        print(
            f"\n{label} goal: expansion={expansion}; cost={node.cost:.2f}; "
            f"exact path length={exact_path_length:.6f} m; "
            f"position error={position_error:.3f} m; "
            f"yaw error={math.degrees(yaw_error):.2f} deg",
            flush=True,
        )

    def _finish_search(
        self,
        best_terminal: Node,
        best_terminal_expansion: int,
        goal: tuple[float, float, float],
        closed: set[StateKey],
        nodes: dict[StateKey, Node],
        live_plot_every: int,
        progress_callback: Optional[ProgressCallback],
        expansion_callback: Optional[ExpansionCallback],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Node]:
        """Publish final state, report the selected terminal, and reconstruct it.

        Args:
            best_terminal: Lowest-cost terminal node found by the search.
            best_terminal_expansion: Expansion count when the terminal was accepted.
            goal: Nominal rear-axle ``(x, y, yaw)`` goal pose.
            closed: State keys expanded with the standard action set.
            nodes: Current best-known node for every generated state key.
            live_plot_every: Positive when final search state should be published.
            progress_callback: Optional consumer of search-progress snapshots.
            expansion_callback: Optional consumer of the final expansion count.

        Returns:
            The reconstructed path, directions, steering values, and terminal node.
        """
        if expansion_callback is not None:
            expansion_callback(self.expansion_count)
        if progress_callback is not None and live_plot_every > 0:
            self._publish_search_state(closed, nodes, progress_callback)
        self.report_goal("Best", best_terminal, best_terminal_expansion, goal)
        path, directions, steers = self.reconstruct(best_terminal)
        return path, directions, steers, best_terminal

    def _post_goal_budget_complete(
        self,
        first_goal_expansion: Optional[int],
        post_goal_expansions: int,
    ) -> bool:
        """Return whether the post-goal expansion budget is complete.

        Args:
            first_goal_expansion: Expansion count at which the first goal was accepted.
            post_goal_expansions: Required expansions after the first goal.

        Returns:
            Whether a goal exists and the requested expansion budget is complete.
        """
        return first_goal_expansion is not None and (
            self.expansion_count >= first_goal_expansion + post_goal_expansions
        )

    def plan(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        max_expansions: int,
        live_plot_every: int = 0,
        progress_callback: Optional[ProgressCallback] = None,
        expansion_callback: Optional[ExpansionCallback] = None,
        post_goal_expansions: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Node]:
        """Search with the standard five-steer action set.

        Args:
            start: Initial rear-axle ``(x, y, yaw)`` pose.
            goal: Nominal rear-axle goal pose.
            max_expansions: Maximum number of action-set expansions.
            live_plot_every: Progress-publication interval; zero disables it.
            progress_callback: Optional consumer of live-search snapshots.
            expansion_callback: Optional consumer of expansion counts.
            post_goal_expansions: Additional expansions after the first goal.

        Returns:
            Reconstructed path, directions, steering values, and terminal node.

        Raises:
            ValueError: If search controls are invalid or start/goal is in collision.
            RuntimeError: If no accepted terminal is found before termination.
        """
        validate_search_inputs(max_expansions, live_plot_every, post_goal_expansions)
        start = _validated_pose("start", start)
        goal = _validated_pose("goal", goal)

        self.goal = goal
        self._dijkstra_cost_to_goal = None
        self.expanded = []
        self.expansion_count = 0
        self.unique_expanded_state_count = 0
        if self.collides(*start):
            raise ValueError("Start pose (including safety margin) is in collision")
        if self.collides(*goal):
            raise ValueError("Goal pose (including safety margin) is in collision")
        self._prepare_corridor(start, goal)

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
        closed: set[StateKey] = set()

        start_heuristic = self.heuristic(*start)
        queue: list[OpenEntry] = [
            (
                self._main_queue_priority(start_node, start_heuristic),
                start_heuristic,
                -start_node.cost,
                0,
                start_key,
                start_node,
            )
        ]

        serial = 0
        best_terminal: Optional[Node] = None
        best_terminal_expansion: Optional[int] = None
        first_goal_expansion: Optional[int] = None
        expanded_state_keys: set[StateKey] = set()
        open_exhausted = False

        while self.expansion_count < max_expansions:
            if self._post_goal_budget_complete(
                first_goal_expansion,
                post_goal_expansions,
            ):
                assert best_terminal is not None
                assert best_terminal_expansion is not None
                return self._finish_search(
                    best_terminal,
                    best_terminal_expansion,
                    goal,
                    closed,
                    nodes,
                    live_plot_every,
                    progress_callback,
                    expansion_callback,
                )

            if self._peek_best_open_priority(queue, closed, nodes) is None:
                open_exhausted = True
                break
            current_entry = self._pop_best_open(queue, closed, nodes)
            if current_entry is None:
                continue
            current_key, current = current_entry
            closed.add(current_key)
            expanded_state_keys.add(current_key)
            self.unique_expanded_state_count = len(expanded_state_keys)
            self.expansion_count += 1

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
                # have non-negative cost, so expanding it cannot produce a cheaper
                # terminal. Keeping it closed also removes it from OPEN.
                closed.add(current_key)
                continue

            for primitive_length, collision_distances in self._primitive_actions():
                for direction in (1, -1):
                    for steer_index in self.steer_indices:
                        endpoint = self.check_primitive(
                            current,
                            direction,
                            steer_index,
                            collision_distances,
                        )
                        if endpoint is None:
                            continue
                        successor_cost = self._successor_cost(
                            current,
                            direction,
                            steer_index,
                            primitive_length,
                        )
                        successor_key = self._state_key_from_values(
                            *endpoint,
                            direction,
                            steer_index,
                        )
                        incumbent = nodes.get(successor_key)
                        if incumbent is None or successor_cost + 1e-9 < incumbent.cost:
                            successor = Node(
                                *endpoint,
                                successor_cost,
                                current,
                                direction,
                                steer_index,
                                primitive_length,
                            )
                            nodes[successor_key] = successor

                            # An improved state must be reconsidered even if an older
                            # representative with the same key was already closed.
                            closed.discard(successor_key)
                            serial += 1
                            heuristic = self.heuristic(successor.x, successor.y, successor.yaw)
                            heapq.heappush(
                                queue,
                                (
                                    self._main_queue_priority(successor, heuristic),
                                    heuristic,
                                    -successor.cost,
                                    serial,
                                    successor_key,
                                    successor,
                                ),
                            )
            if (
                progress_callback is not None
                and live_plot_every > 0
                and self.expansion_count % live_plot_every == 0
            ):
                self._publish_search_state(closed, nodes, progress_callback)

        post_goal_budget_complete = self._post_goal_budget_complete(
            first_goal_expansion,
            post_goal_expansions,
        )
        if best_terminal is not None:
            if not post_goal_budget_complete and not open_exhausted:
                print(
                    "\nWarning: the search limit was reached before the requested "
                    "post-goal expansion budget was completed.",
                    flush=True,
                )
            assert best_terminal_expansion is not None
            return self._finish_search(
                best_terminal,
                best_terminal_expansion,
                goal,
                closed,
                nodes,
                live_plot_every,
                progress_callback,
                expansion_callback,
            )
        corridor_detail = ""
        if self.corridor is not None:
            corridor_detail = (
                f" inside corridor width {self.corridor_width:.3f} m at coarse "
                f"resolution {self.corridor_grid_resolution:.3f} m"
            )
        raise RuntimeError(
            f"No path found after {self.expansion_count} action-set expansions"
            f"{corridor_detail} ({self.expansion_count} fine, 0 coarse; two_queue=False)"
        )


_MAZE_SVG_SCALE = 0.25
_MAZE_FRAME_LEFT = 20.410
_MAZE_FRAME_RIGHT = 457.459
_MAZE_FRAME_TOP = 35.983
_MAZE_FRAME_BOTTOM = 263.197


def _maze_obstacles() -> tuple[Obstacle, ...]:
    """Convert the source SVG wall strokes into planning obstacles.

    The source uses 10-unit, butt-capped strokes with miter joins. Each source
    box below is one axis-aligned stroke segment, extended at polyline joins so
    the union of boxes preserves the SVG corners. The one nearly vertical source
    segment is represented by its tight axis-aligned bounds.

    Returns:
        Axis-aligned wall rectangles in maze world coordinates.
    """
    source_boxes = (
        # Left and lower outer wall polyline.
        (20.410, 30.410, 35.983, 263.197),
        (20.410, 414.754, 253.197, 263.197),
        # Left-facing middle branch and lower vertical branch.
        (25.410, 69.672, 169.590, 179.590),
        (190.082, 200.082, 212.295, 258.197),
        # Upper and right outer wall polyline.
        (62.295, 457.459, 35.983, 45.983),
        (447.459, 457.459, 35.983, 263.197),
        # Upper short vertical.
        (276.147, 286.147, 40.983, 86.885),
        # Left and center branching polyline.
        (64.672, 112.295, 81.885, 91.885),
        (64.672, 74.672, 81.885, 135.328),
        (64.672, 200.082, 125.328, 135.328),
        (190.082, 200.082, 125.328, 176.311),
        (148.279, 200.082, 166.311, 176.311),
        (148.279, 158.279, 166.311, 217.295),
        (69.672, 158.279, 207.295, 217.295),
        # Remaining left-side verticals.
        (104.017, 114.017, 130.328, 174.590),
        (149.099, 159.099, 40.983, 130.328),
        # Right middle branch.
        (404.836, 414.836, 125.328, 174.590),
        (404.836, 452.459, 125.328, 135.328),
        (366.394, 452.459, 212.213, 222.213),
        # Upper-right and center branching polyline.
        (195.082, 244.344, 81.885, 91.885),
        (234.344, 244.344, 81.885, 135.328),
        (234.344, 286.147, 125.328, 135.328),
        (276.147, 286.147, 125.328, 179.590),
        (276.147, 371.394, 169.590, 179.590),
        (361.394, 371.394, 81.885, 179.590),
        (319.590, 371.394, 81.885, 91.885),
        (318.772, 329.590, 86.791, 130.422),
        (366.394, 414.754, 81.885, 91.885),
        # Lower-center U-shaped polyline and vertical branch.
        (234.344, 244.344, 174.590, 217.295),
        (234.344, 329.590, 207.295, 217.295),
        (319.590, 329.590, 174.590, 217.295),
        (276.967, 286.967, 212.295, 258.197),
    )
    scale = _MAZE_SVG_SCALE
    return tuple(
        Obstacle(
            (xmin - _MAZE_FRAME_LEFT) * scale,
            (xmax - _MAZE_FRAME_LEFT) * scale,
            (_MAZE_FRAME_BOTTOM - ymax) * scale,
            (_MAZE_FRAME_BOTTOM - ymin) * scale,
        )
        for xmin, xmax, ymin, ymax in source_boxes
    )


def make_environment(name: str, planner_overrides: dict[str, float]) -> Environment:
    """Construct one named demonstration scene with planner configuration.

    Args:
        name: Identifier of a built-in demonstration environment.
        planner_overrides: Complete planner configuration to store in the scene.

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
    if name == "maze":
        return Environment(
            name="maze",
            title="Classic branching maze environment",
            width=(_MAZE_FRAME_RIGHT - _MAZE_FRAME_LEFT) * _MAZE_SVG_SCALE,
            height=(_MAZE_FRAME_BOTTOM - _MAZE_FRAME_TOP) * _MAZE_SVG_SCALE,
            obstacles=_maze_obstacles(),
            start=((46.540 - _MAZE_FRAME_LEFT) * _MAZE_SVG_SCALE, 51.5, math.radians(-90.0)),
            goal=((431.147 - _MAZE_FRAME_LEFT) * _MAZE_SVG_SCALE, 6.0, math.radians(-90.0)),
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
    if name == "parking2_hard":
        return Environment(
            name="parking2_hard",
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
