#!/usr/bin/env python3
"""Build and enforce a coarse path corridor for Hybrid A*.

Corridor mode first inflates obstacles and the world boundary by a configured
rear-axle clearance, then runs eight-connected point-robot A* on a coarse
position grid. The exact continuous start and goal are attached to valid
corners of their containing grid cells with collision-checked line segments,
so the stored centerline includes both requested endpoints. Diagonal moves
cannot squeeze between blocked orthogonal neighbors, and every grid edge is
also checked geometrically to catch thin obstacles between free vertices.

The corridor is the closed geometric tube whose radius is ``corridor_width``
around that complete polyline. Hybrid A* still performs its normal rectangular
vehicle collision checks; the corridor is only an additional restriction on
sampled rear-axle positions. It guides the detailed search into one coarse
route and can greatly reduce work, but an overly narrow tube or coarse grid can
exclude a feasible vehicle path.
"""

import heapq
import math
from typing import Sequence

import numpy as np
from numba import njit


@njit(cache=True)
def _segment_avoids_boxes(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    boxes: np.ndarray,
) -> bool:
    """Return whether a closed segment avoids every closed axis-aligned box.

    Args:
        x0: First endpoint x-coordinate.
        y0: First endpoint y-coordinate.
        x1: Second endpoint x-coordinate.
        y1: Second endpoint y-coordinate.
        boxes: Box rows formatted as ``[xmin, xmax, ymin, ymax]``.

    Returns:
        Whether the segment is disjoint from every box.
    """
    dx, dy = x1 - x0, y1 - y0
    for box in boxes:
        minimum_t, maximum_t = 0.0, 1.0
        separated = False
        for origin, delta, minimum, maximum in (
            (x0, dx, box[0], box[1]),
            (y0, dy, box[2], box[3]),
        ):
            if abs(delta) <= 1e-15:
                if origin < minimum or origin > maximum:
                    separated = True
                    break
                continue
            first_t = (minimum - origin) / delta
            second_t = (maximum - origin) / delta
            entry_t = min(first_t, second_t)
            exit_t = max(first_t, second_t)
            minimum_t = max(minimum_t, entry_t)
            maximum_t = min(maximum_t, exit_t)
            if minimum_t > maximum_t:
                separated = True
                break
        if not separated:
            return False
    return True


@njit(cache=True)
def _point_is_inside_corridor(
    x: float,
    y: float,
    path: np.ndarray,
    radius_squared: float,
) -> bool:
    """Return whether a point lies in the closed tube around a polyline.

    Args:
        x: Point x-coordinate.
        y: Point y-coordinate.
        path: Corridor centerline coordinates.
        radius_squared: Squared corridor radius.

    Returns:
        Whether the point lies within the corridor radius of any segment.
    """
    for index in range(len(path) - 1):
        ax, ay = path[index, 0], path[index, 1]
        bx, by = path[index + 1, 0], path[index + 1, 1]
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-24:
            projection = 0.0
        else:
            projection = ((x - ax) * dx + (y - ay) * dy) / length_squared
            projection = min(1.0, max(0.0, projection))
        nearest_x = ax + projection * dx
        nearest_y = ay + projection * dy
        error_x, error_y = x - nearest_x, y - nearest_y
        if error_x * error_x + error_y * error_y <= radius_squared + 1e-12:
            return True
    return False


@njit(cache=True)
def _arc_is_inside_corridor(
    x0: float,
    y0: float,
    yaw0: float,
    direction: int,
    steer: float,
    wheelbase: float,
    sample_distances: np.ndarray,
    path: np.ndarray,
    radius_squared: float,
) -> bool:
    """Sample one bicycle arc and require every rear-axle point in the corridor.

    Args:
        x0: Arc start x-coordinate.
        y0: Arc start y-coordinate.
        yaw0: Arc start heading.
        direction: Travel direction, either ``1`` or ``-1``.
        steer: Constant steering angle.
        wheelbase: Vehicle wheelbase.
        sample_distances: Positive arc-length samples.
        path: Corridor centerline coordinates.
        radius_squared: Squared corridor radius.

    Returns:
        Whether every sampled rear-axle position lies inside the corridor.
    """
    curvature = math.tan(steer) / wheelbase
    straight = abs(curvature) < 1e-12
    for distance in sample_distances:
        signed_distance = direction * distance
        if straight:
            x = x0 + signed_distance * math.cos(yaw0)
            y = y0 + signed_distance * math.sin(yaw0)
        else:
            yaw = yaw0 + signed_distance * curvature
            x = x0 + (math.sin(yaw) - math.sin(yaw0)) / curvature
            y = y0 - (math.cos(yaw) - math.cos(yaw0)) / curvature
        if not _point_is_inside_corridor(x, y, path, radius_squared):
            return False
    return True


class CoarsePathCorridor:
    """Create a radius-bounded corridor around an eight-connected 2D A* path."""

    def __init__(
        self,
        world_width: float,
        world_height: float,
        obstacle_boxes: Sequence[tuple[float, float, float, float]],
        coarse_resolution: float,
        corridor_width: float,
        obstacle_clearance: float = 0.0,
    ) -> None:
        """Configure the occupancy grid and corridor geometry.

        Args:
            world_width: Planning-world width in meters.
            world_height: Planning-world height in meters.
            obstacle_boxes: Closed boxes formatted as ``xmin, xmax, ymin, ymax``.
            coarse_resolution: Spacing of the coarse point-robot grid in meters.
            corridor_width: Maximum rear-axle distance from the coarse path in meters.
            obstacle_clearance: Distance added around obstacles and inward from
                world boundaries, in meters.

        Returns:
            None.

        Raises:
            ValueError: If a dimension, resolution, corridor width, or clearance
                is invalid.
        """
        for name, value in (
            ("world_width", world_width),
            ("world_height", world_height),
            ("coarse_resolution", coarse_resolution),
            ("corridor_width", corridor_width),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(obstacle_clearance) or obstacle_clearance < 0.0:
            raise ValueError("obstacle_clearance must be a finite non-negative number")
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.coarse_resolution = float(coarse_resolution)
        self.corridor_width = float(corridor_width)
        self.obstacle_clearance = float(obstacle_clearance)
        self.obstacle_boxes = tuple(tuple(float(value) for value in box) for box in obstacle_boxes)
        self._inflated_obstacle_boxes = np.asarray(self.obstacle_boxes, dtype=float).reshape((-1, 4)).copy()
        if len(self._inflated_obstacle_boxes) > 0:
            self._inflated_obstacle_boxes[:, (0, 2)] -= self.obstacle_clearance
            self._inflated_obstacle_boxes[:, (1, 3)] += self.obstacle_clearance
        self.nx = int(math.floor(self.world_width / self.coarse_resolution + 1e-9)) + 1
        self.ny = int(math.floor(self.world_height / self.coarse_resolution + 1e-9)) + 1
        self.occupancy = self._build_occupancy()
        self.coarse_path = np.empty((0, 2), dtype=float)
        self._corridor_radius_squared = self.corridor_width**2

    def _build_occupancy(self) -> np.ndarray:
        """Rasterize inflated obstacles and world boundaries as point occupancy.

        Returns:
            A boolean array indexed as ``[y_index, x_index]``.
        """
        x_coordinates = np.arange(self.nx, dtype=float) * self.coarse_resolution
        y_coordinates = np.arange(self.ny, dtype=float) * self.coarse_resolution
        occupancy = (
            (x_coordinates[np.newaxis, :] <= self.obstacle_clearance)
            | (x_coordinates[np.newaxis, :] >= self.world_width - self.obstacle_clearance)
            | (y_coordinates[:, np.newaxis] <= self.obstacle_clearance)
            | (y_coordinates[:, np.newaxis] >= self.world_height - self.obstacle_clearance)
        )
        for xmin, xmax, ymin, ymax in self._inflated_obstacle_boxes:
            occupancy |= (
                (x_coordinates[np.newaxis, :] >= xmin)
                & (x_coordinates[np.newaxis, :] <= xmax)
                & (y_coordinates[:, np.newaxis] >= ymin)
                & (y_coordinates[:, np.newaxis] <= ymax)
            )
        return occupancy

    def _point_violates_clearance(self, point: tuple[float, float]) -> bool:
        """Check inflated obstacles and inward-inflated world boundaries.

        Args:
            point: Continuous world ``(x, y)`` coordinate.

        Returns:
            Whether the point is in the configured obstacle or boundary clearance.
        """
        x, y = point
        clearance = self.obstacle_clearance
        if (
            x <= clearance
            or x >= self.world_width - clearance
            or y <= clearance
            or y >= self.world_height - clearance
        ):
            return True
        return any(
            xmin - clearance <= x <= xmax + clearance and ymin - clearance <= y <= ymax + clearance
            for xmin, xmax, ymin, ymax in self.obstacle_boxes
        )

    @staticmethod
    def _segment_intersects_box(
        first: tuple[float, float],
        second: tuple[float, float],
        box: tuple[float, float, float, float],
    ) -> bool:
        """Test a closed line segment against a closed axis-aligned box.

        Args:
            first: First segment endpoint.
            second: Second segment endpoint.
            box: Closed box formatted as ``xmin, xmax, ymin, ymax``.

        Returns:
            Whether the segment touches or enters the box.
        """
        xmin, xmax, ymin, ymax = box
        x0, y0 = first
        dx, dy = second[0] - x0, second[1] - y0
        minimum_t, maximum_t = 0.0, 1.0
        for origin, delta, minimum, maximum in (
            (x0, dx, xmin, xmax),
            (y0, dy, ymin, ymax),
        ):
            if abs(delta) <= 1e-15:
                if origin < minimum or origin > maximum:
                    return False
                continue
            first_t = (minimum - origin) / delta
            second_t = (maximum - origin) / delta
            entry_t, exit_t = min(first_t, second_t), max(first_t, second_t)
            minimum_t = max(minimum_t, entry_t)
            maximum_t = min(maximum_t, exit_t)
            if minimum_t > maximum_t:
                return False
        return True

    def _connector_is_clear(
        self,
        point: tuple[float, float],
        index: tuple[int, int],
    ) -> bool:
        """Check the segment attaching an exact endpoint to one grid vertex.

        Coarse A* searches only discrete grid vertices, while the requested start
        and goal are continuous world coordinates that normally lie between those
        vertices. A connector is the straight segment from such an exact start or
        goal point to a candidate grid vertex. It becomes the first or last segment
        of the complete coarse path and must therefore avoid inflated obstacles.

        Args:
            point: Exact continuous start or goal world coordinate.
            index: Integer index of the candidate grid vertex to attach to.

        Returns:
            Whether the closed point-to-vertex segment avoids every inflated obstacle.
        """
        vertex = (
            index[0] * self.coarse_resolution,
            index[1] * self.coarse_resolution,
        )
        return bool(
            _segment_avoids_boxes(
                point[0],
                point[1],
                vertex[0],
                vertex[1],
                self._inflated_obstacle_boxes,
            )
        )

    def _grid_edge_is_clear(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> bool:
        """Check one coarse graph edge against inflated obstacle boxes.

        Raster occupancy alone can miss an obstacle that lies between two free
        grid vertices. Testing the continuous edge prevents the coarse centerline
        and its resulting corridor from crossing such an obstacle.

        Args:
            first: Integer index of the first grid vertex.
            second: Integer index of the second grid vertex.

        Returns:
            Whether the closed vertex-to-vertex segment avoids every inflated
            obstacle.
        """
        resolution = self.coarse_resolution
        first_point = (first[0] * resolution, first[1] * resolution)
        second_point = (second[0] * resolution, second[1] * resolution)
        return bool(
            _segment_avoids_boxes(
                first_point[0],
                first_point[1],
                second_point[0],
                second_point[1],
                self._inflated_obstacle_boxes,
            )
        )

    def _connector_candidates(
        self,
        name: str,
        point: tuple[float, float],
    ) -> tuple[tuple[float, int, int], ...]:
        """Find valid grid attachments for an exact start or goal point.

        The coarse search graph contains grid vertices only; it does not contain
        the exact continuous start and goal coordinates. Each exact endpoint is
        therefore attached to one of the corners of the grid cell containing it.
        The straight point-to-corner segment is called a connector.

        This method considers the containing cell's floor/ceil corner indices and
        retains only vertices that are free and whose connector does not touch an
        inflated obstacle. The returned connector length is included in A*'s path
        cost. Keeping every valid corner lets A* choose the best attachment instead
        of prematurely rounding the endpoint to one grid vertex.

        Args:
            name: Human-readable endpoint name used in error messages.
            point: Exact continuous start or goal world coordinate.

        Returns:
            Candidate ``(connector_length, x_index, y_index)`` tuples, sorted by
            connector length and then by deterministic grid-index tie breaks.

        Raises:
            ValueError: If the exact point violates inflated clearance.
            RuntimeError: If no nearby grid vertex has a clear connector.
        """
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Coarse corridor {name} must contain finite coordinates")
        if self._point_violates_clearance(point):
            raise ValueError(
                f"Coarse corridor {name} violates the {self.obstacle_clearance:.3f} m "
                "obstacle or world-boundary clearance"
            )

        resolution = self.coarse_resolution
        x_indices = sorted(
            index
            for index in {
                int(math.floor(x / resolution)),
                int(math.ceil(x / resolution)),
            }
            if 0 <= index < self.nx
        )
        y_indices = sorted(
            index
            for index in {
                int(math.floor(y / resolution)),
                int(math.ceil(y / resolution)),
            }
            if 0 <= index < self.ny
        )
        candidates: list[tuple[float, int, int]] = []
        for iy in y_indices:
            grid_y = iy * resolution
            for ix in x_indices:
                if self.occupancy[iy, ix]:
                    continue
                grid_x = ix * resolution
                distance = math.hypot(x - grid_x, y - grid_y)
                if not self._connector_is_clear(point, (ix, iy)):
                    continue
                candidates.append((distance, ix, iy))
        if not candidates:
            raise RuntimeError(
                f"Coarse corridor {name} has no nearby free grid vertex with a "
                "collision-free connector; reduce --coarse_resolution"
            )
        return tuple(sorted(candidates))

    @staticmethod
    def _octile_distance(first: tuple[int, int], second: tuple[int, int], resolution: float) -> float:
        """Return the exact empty-grid cost for eight-connected movement.

        Args:
            first: First integer grid index.
            second: Second integer grid index.
            resolution: Grid spacing in meters.

        Returns:
            Octile distance between the indices in meters.
        """
        dx = abs(first[0] - second[0])
        dy = abs(first[1] - second[1])
        diagonal = min(dx, dy)
        straight = max(dx, dy) - diagonal
        return resolution * (math.sqrt(2.0) * diagonal + straight)

    @classmethod
    def _goal_candidate_heuristic(
        cls,
        index: tuple[int, int],
        goal_candidates: tuple[tuple[float, int, int], ...],
        resolution: float,
    ) -> float:
        """Lower-bound travel from a grid vertex to the exact continuous goal.

        The exact goal may have several valid connector vertices around its
        containing grid cell. For each candidate, the estimate is the empty-grid
        octile distance from ``index`` to that vertex plus the straight connector
        length from the vertex to the exact goal. The minimum candidate estimate
        guides A* without committing the search to one rounded goal vertex.

        Args:
            index: Current grid index.
            goal_candidates: Goal connector lengths and grid indices.
            resolution: Grid spacing in meters.

        Returns:
            Minimum grid-distance lower bound plus exact goal-connector length.
        """
        return min(
            cls._octile_distance(index, (goal_ix, goal_iy), resolution) + connector_length
            for connector_length, goal_ix, goal_iy in goal_candidates
        )

    def build(self, start: tuple[float, float], goal: tuple[float, float]) -> np.ndarray:
        """Run coarse A* and store its complete world-coordinate polyline.

        The search uses valid connector vertices for both continuous endpoints.
        Its output begins at the exact start, follows the selected start connector,
        traverses the eight-connected grid path, follows the selected goal
        connector, and ends at the exact goal.

        Args:
            start: Exact start rear-axle coordinate.
            goal: Exact goal rear-axle coordinate.

        Returns:
            A copied ``N x 2`` coarse polyline including exact start and goal.

        Raises:
            ValueError: If an exact endpoint violates obstacle or boundary clearance.
            RuntimeError: If no coarse point-robot route exists.
        """
        start_candidates = self._connector_candidates("start", start)
        goal_candidates = self._connector_candidates("goal", goal)
        goal_connector_costs = {(ix, iy): connector_length for connector_length, ix, iy in goal_candidates}
        queue: list[tuple[float, float, int, int]] = []
        costs: dict[tuple[int, int], float] = {}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        for connector_length, ix, iy in start_candidates:
            index = (ix, iy)
            if connector_length + 1e-12 >= costs.get(index, math.inf):
                continue
            costs[index] = connector_length
            priority = connector_length + self._goal_candidate_heuristic(
                index,
                goal_candidates,
                self.coarse_resolution,
            )
            heapq.heappush(queue, (priority, connector_length, ix, iy))
        moves = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        best_goal_index: tuple[int, int] | None = None
        best_total_cost = math.inf
        while queue:
            priority, cost, ix, iy = heapq.heappop(queue)
            current = (ix, iy)
            if cost > costs.get(current, math.inf) + 1e-12:
                continue
            if priority >= best_total_cost - 1e-12:
                break
            if current in goal_connector_costs:
                total_cost = cost + goal_connector_costs[current]
                if total_cost + 1e-12 < best_total_cost:
                    best_total_cost = total_cost
                    best_goal_index = current
            for dx, dy, multiplier in moves:
                next_ix, next_iy = ix + dx, iy + dy
                if not (0 <= next_ix < self.nx and 0 <= next_iy < self.ny):
                    continue
                if self.occupancy[next_iy, next_ix]:
                    continue
                if dx != 0 and dy != 0 and (self.occupancy[iy, next_ix] or self.occupancy[next_iy, ix]):
                    continue
                neighbor = (next_ix, next_iy)
                if not self._grid_edge_is_clear(current, neighbor):
                    continue
                candidate = cost + multiplier * self.coarse_resolution
                if candidate + 1e-12 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = candidate
                parents[neighbor] = current
                priority = candidate + self._goal_candidate_heuristic(
                    neighbor,
                    goal_candidates,
                    self.coarse_resolution,
                )
                heapq.heappush(queue, (priority, candidate, next_ix, next_iy))

        if best_goal_index is None:
            raise RuntimeError(
                "Coarse 2D A* found no path; reduce --coarse_resolution or disable corridor mode"
            )

        indices = [best_goal_index]
        while indices[-1] in parents:
            indices.append(parents[indices[-1]])
        indices.reverse()
        points = [start]
        points.extend((ix * self.coarse_resolution, iy * self.coarse_resolution) for ix, iy in indices)
        points.append(goal)
        deduplicated: list[tuple[float, float]] = []
        for point in points:
            point = (float(point[0]), float(point[1]))
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)
        if len(deduplicated) == 1:
            deduplicated.append(deduplicated[0])
        self.coarse_path = np.asarray(deduplicated, dtype=float)
        return self.coarse_path.copy()

    @property
    def path_length(self) -> float:
        """Return the coarse polyline length in meters.

        Returns:
            Sum of the coarse polyline segment lengths.
        """
        if len(self.coarse_path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.coarse_path, axis=0), axis=1).sum())

    def contains(self, x: float, y: float) -> bool:
        """Check whether a rear-axle point lies inside the built corridor.

        Args:
            x: Rear-axle x-coordinate.
            y: Rear-axle y-coordinate.

        Returns:
            Whether the point lies inside the corridor.

        Raises:
            RuntimeError: If the coarse path has not been built.
        """
        if len(self.coarse_path) < 2:
            raise RuntimeError("Corridor path has not been built")
        return bool(
            _point_is_inside_corridor(
                float(x),
                float(y),
                self.coarse_path,
                self._corridor_radius_squared,
            )
        )

    def contains_arc(
        self,
        x0: float,
        y0: float,
        yaw0: float,
        direction: int,
        steer: float,
        wheelbase: float,
        sample_distances: np.ndarray,
    ) -> bool:
        """Check every sampled rear-axle point of a bicycle arc.

        Args:
            x0: Arc start x-coordinate.
            y0: Arc start y-coordinate.
            yaw0: Arc start heading.
            direction: Travel direction, either ``1`` or ``-1``.
            steer: Constant steering angle in radians.
            wheelbase: Vehicle wheelbase in meters.
            sample_distances: Positive arc distances ending at the primitive endpoint.

        Returns:
            Whether every sample is within the corridor.
        """
        return bool(
            _arc_is_inside_corridor(
                x0,
                y0,
                yaw0,
                direction,
                steer,
                wheelbase,
                sample_distances,
                self.coarse_path,
                self._corridor_radius_squared,
            )
        )
