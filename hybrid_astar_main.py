#!/usr/bin/env python3
"""Hybrid A* for a car-like robot without analytic Reeds-Shepp expansion.

Examples:
    python hybrid_astar_main.py
    python hybrid_astar_main.py --env walls
    python hybrid_astar_main.py --env parking

The sampling/safety arguments have the same defaults for every environment:

    margin=0.20 m, integration=0.10 m, collision=0.05 m

``integration_step`` controls reconstructed path/animation sampling.
``collision_check_step`` independently controls swept-path collision sampling.
Motion primitives use exact constant-curvature bicycle arcs, so changing either
sampling step does not change a primitive's endpoint.

After a solution is found, a timestamped path PNG is saved and the animation is
played in a blocking Matplotlib window by default. Use ``--save_video`` to also
save the animation, and ``--no_animation_plot`` to disable playback independently.

``heuristic`` selects the distance-only, legacy distance-and-heading,
tolerance-aware, or obstacle-aware estimate used for queue ordering and the
live-search comparison. Standard open states are expanded by the weighted total
estimate ``g+weight*h``.

``post_goal_expansions`` continues search for a requested number of additional
action-set expansions after the first accepted goal while retaining the
lowest-cost terminal seen.

The optional live plot colors every open state by its single/fine-queue priority
or ``h``. Closed states use gray dots underneath the open-state dots. The left
panel keeps a full vehicle box on its minimum-score open state, while the second
panel selects the minimum-heuristic state across both open and closed states.
When the Dijkstra heuristic is enabled, the right panel shows its relaxed 2D
cost-to-go grid. Per-state heading arrows are omitted because dense searches
become unreadable.

This variant optionally uses two OPEN queues over one shared state graph:

* the fine queue expands base-length primitives with all five steering values;
* the coarse queue expands coarse-length primitives with three steering values
  ``{-max, 0, +max}``.

Enable the second queue with ``--two_queues``. Set its edge length with
``--coarse_primitive_mult``; it defaults to four times the fine
``--primitive_length``. ``--coarse_heuristic_weight`` can give the coarse
queue a different heuristic multiplier; when omitted, it uses
``--heuristic_weight``. At each iteration the coarse queue is eligible when
``min_coarse_priority <= queue_beta * min_fine_priority``. A multiplicative
origin-priority factor penalizes fine-generated nodes in the coarse queue; it
does not affect coarse-generated nodes. To prevent the fine queue from being
starved, ``--max_consecutive_coarse_expansions`` limits each coarse burst while
a fine state remains available; the next expansion is then forced from the fine
queue. Fine actions therefore remain available at every nonterminal state, but
are not generated on every coarse expansion. Both primitive sets use the same
swept collision-check spacing. Without ``--two_queues`` the planner runs the
original single-queue search and does not apply the origin-priority factor.
"""

import argparse
import json
import math
import subprocess
import warnings
from datetime import datetime
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from hybrid_astar_planner import (
    Environment,
    HybridAStar,
    SearchNodeState,
    SearchSnapshot,
    Vehicle,
    make_environment,
    vehicle_heading_arrow,
    vehicle_polygon,
    vehicle_tire_polygons,
    wrap,
)
from hybrid_astar_two_queues import TwoQueueHybridAStar

plt.rcParams.update({"font.size": 8})


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
        planner: Planner providing the configured position tolerance.
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
        planner: Planner providing vehicle geometry.
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
        planner: Planner providing vehicle geometry.
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
        planner: Planner providing vehicle geometry and goal tolerances.
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
            planner: Planner providing geometry, tolerances, and live search state.
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
        self.score_label = score_label
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
            s=5,
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

        # The trajectory and full vehicle box identify the state selected by this
        # panel's score rule.
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
            f"[{getattr(self.planner, 'last_expansion_queue', 'fine')}]\n"
            f"g={node.cost:.2f}; h={snapshot.heuristic:.2f}; "
            f"{self.score_label}={snapshot.total_estimate:.2f}; distance={distance:.2f} m\n"
            f"yaw err={math.degrees(yaw_error):.1f} deg "
            f"open={open_count:}; closed={closed_count:}",
        )


class DijkstraCostToGoalView:
    """Own the relaxed two-dimensional Dijkstra cost-to-go heatmap."""

    def __init__(self, ax: plt.Axes, planner: HybridAStar, env: Environment) -> None:
        """Create the Dijkstra heatmap panel.

        Args:
            ax: Matplotlib axes receiving the generated artists.
            planner: Planner that lazily caches the relaxed cost-to-go grid.
            env: Environment providing static scene geometry.

        Returns:
            None.
        """
        self.ax = ax
        self.planner = planner
        draw_scene_background(ax, planner, env, show_start_vehicle=False)
        self.cost_norm = Normalize(vmin=0.0, vmax=1.0)
        self.cost_mappable = ScalarMappable(norm=self.cost_norm, cmap="magma")
        self.cost_mappable.set_array(np.empty(0, dtype=float))
        self.heatmap = ax.imshow(
            np.ma.masked_all((1, 1), dtype=float),
            origin="lower",
            extent=(0.0, env.width, 0.0, env.height),
            cmap="magma",
            norm=self.cost_norm,
            interpolation="nearest",
            alpha=0.82,
            zorder=0,
        )
        self.colorbar = ax.figure.colorbar(self.cost_mappable, ax=ax, pad=0.02)
        self.colorbar.set_label("Relaxed cost to goal [m]")
        ax.set_title("Relaxed Dijkstra cost to goal\nwaiting for dijkstra heuristic")

    def update(self, expansion_count: int) -> None:
        """Refresh the heatmap from the planner's lazily cached Dijkstra grid.

        Args:
            expansion_count: Number of action-set expansions performed so far.

        Returns:
            None.
        """
        costs = self.planner._dijkstra_cost_to_goal
        if costs is None:
            self.ax.set_title(
                f"Relaxed Dijkstra cost to goal expansion {expansion_count:}\n"
                "unavailable outside dijkstra heuristic"
            )
            return

        finite_costs = costs[np.isfinite(costs)]
        if finite_costs.size == 0:
            self.ax.set_title(
                f"Relaxed Dijkstra cost to goal expansion {expansion_count:}\nno reachable grid cells"
            )
            return

        minimum = float(finite_costs.min())
        maximum = float(finite_costs.max())
        if maximum - minimum < 1e-12:
            padding = max(0.5, abs(minimum) * 0.01)
            minimum -= padding
            maximum += padding
        self.cost_norm.vmin = minimum
        self.cost_norm.vmax = maximum
        self.cost_mappable.set_clim(minimum, maximum)
        self.heatmap.set_data(np.ma.masked_invalid(costs))
        ny, nx = costs.shape
        half_cell = 0.5 * self.planner.xy_resolution
        self.heatmap.set_extent(
            (
                -half_cell,
                (nx - 0.5) * self.planner.xy_resolution,
                -half_cell,
                (ny - 0.5) * self.planner.xy_resolution,
            )
        )
        self.colorbar.update_normal(self.cost_mappable)
        self.ax.set_title(f"Relaxed Dijkstra cost to goal expansion {expansion_count:}")


class LiveSearchPlot:
    """Compare live state scores and optionally show Dijkstra cost-to-go."""

    def __init__(self, planner: HybridAStar, env: Environment) -> None:
        """Create state-score panels and, for Dijkstra, its cost-to-go heatmap.

        Args:
            planner: Planner providing geometry, tolerances, and live search state.
            env: Environment providing bounds, obstacles, and start/goal poses.

        Returns:
            None.
        """
        plt.ion()
        show_dijkstra_cost = planner.heuristic_mode == "dijkstra"
        panel_count = 3 if show_dijkstra_cost else 2
        self.fig, axes = plt.subplots(
            1,
            panel_count,
            figsize=(6 * panel_count, 5),
            sharex=True,
            sharey=True,
        )
        self.planner = planner
        self.env = env
        self.panel_axes = tuple(np.atleast_1d(axes))
        self.corridor_overlays: list[object] = []
        self._corridor_overlay_added = False
        self.priority_label = planner._main_queue_priority_label()
        self.best_total = LiveStateView(
            axes[0],
            planner,
            env,
            trajectory_label=f"Minimum {self.priority_label} trajectory",
            state_label=f"Minimum {self.priority_label} state",
            score_attribute="total_estimate",
            score_label=self.priority_label,
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
        self.dijkstra_cost_to_goal = (
            DijkstraCostToGoalView(axes[2], planner, env) if show_dijkstra_cost else None
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

    def _add_corridor_overlay(self) -> None:
        """Shade live-panel space outside the built path corridor.

        Returns:
            None.
        """
        if self._corridor_overlay_added:
            return
        corridor = getattr(self.planner, "corridor", None)
        if corridor is None or len(corridor.coarse_path) < 2:
            return

        # Rasterize the exact polyline-distance test once at display resolution.
        # This keeps live updates cheap while giving smooth enough corridor edges.
        maximum_cells = 320
        cell_size = max(self.env.width, self.env.height) / maximum_cells
        nx = max(1, int(math.ceil(self.env.width / cell_size)))
        ny = max(1, int(math.ceil(self.env.height / cell_size)))
        xs = (np.arange(nx, dtype=float) + 0.5) * self.env.width / nx
        ys = (np.arange(ny, dtype=float) + 0.5) * self.env.height / ny
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = np.zeros((ny, nx), dtype=bool)
        radius_squared = corridor.corridor_width**2
        for first, second in zip(corridor.coarse_path[:-1], corridor.coarse_path[1:]):
            delta = second - first
            length_squared = float(np.dot(delta, delta))
            if length_squared <= 1e-24:
                projection = np.zeros_like(grid_x)
            else:
                projection = np.clip(
                    ((grid_x - first[0]) * delta[0] + (grid_y - first[1]) * delta[1]) / length_squared,
                    0.0,
                    1.0,
                )
            nearest_x = first[0] + projection * delta[0]
            nearest_y = first[1] + projection * delta[1]
            inside |= (grid_x - nearest_x) ** 2 + (grid_y - nearest_y) ** 2 <= radius_squared

        rgba = np.zeros((ny, nx, 4), dtype=float)
        rgba[~inside] = (0.35, 0.35, 0.35, 0.42)
        for ax in self.panel_axes:
            overlay = ax.imshow(
                rgba,
                origin="lower",
                extent=(0.0, self.env.width, 0.0, self.env.height),
                interpolation="nearest",
                zorder=0.8,
            )
            self.corridor_overlays.append(overlay)
            ax.plot(
                corridor.coarse_path[:, 0],
                corridor.coarse_path[:, 1],
                linestyle="--",
                linewidth=1.4,
                color="tab:cyan",
                label="Coarse 2D A* path",
                zorder=2,
            )
        self._corridor_overlay_added = True

    def update(
        self,
        expansion_count: int,
        best_total: SearchSnapshot,
        best_heuristic: SearchSnapshot,
        states: tuple[SearchNodeState, ...],
    ) -> None:
        """Refresh state-score panels and the relaxed Dijkstra heatmap.

        Args:
            expansion_count: Number of action-set expansions performed so far.
            best_total: Snapshot of the open node with minimum main-queue priority.
            best_heuristic: Snapshot of the open or closed node with minimum ``h``.
            states: Current best-known generated states, including open and closed.

        Returns:
            None.
        """
        if not plt.fignum_exists(self.fig.number):
            return
        self._add_corridor_overlay()
        self.best_total.update(
            expansion_count,
            best_total,
            states,
            f"Minimum {self.priority_label} open state",
        )
        self.best_heuristic.update(
            expansion_count,
            best_heuristic,
            states,
            "Minimum heuristic state (open and closed)",
        )
        if self.dijkstra_cost_to_goal is not None:
            self.dijkstra_cost_to_goal.update(expansion_count)
        if self._interactive_canvas:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)


def save_plot(output: Path, planner: HybridAStar, env: Environment, path: np.ndarray) -> None:
    """Render and save a static summary of the search and final path.

    Args:
        output: Destination path for the generated image.
        planner: Planner providing vehicle geometry, tolerances, and completed search data.
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


def render_animation(
    output: Path,
    planner: HybridAStar,
    env: Environment,
    path: np.ndarray,
    directions: np.ndarray,
    steers: np.ndarray,
    save_video: bool = False,
    show: bool = True,
) -> None:
    """Render the vehicle following the planned path.

    Saving and interactive playback are independent. If saving fails, the error
    is reported and playback still occurs when ``show`` is true.

    Args:
        output: Destination path used when ``save_video`` is true.
        planner: Planner providing geometry, tolerances, and completed search data.
        env: Environment providing bounds, obstacles, and start/goal poses.
        path: Sampled ``N x 3`` solution path.
        directions: Per-sample travel directions aligned with ``path``.
        steers: Per-sample steering angles aligned with ``path``.
        save_video: Whether to save the animation as MP4 or GIF.
        show: Whether to play the animation in a blocking interactive window.

    Returns:
        None.
    """
    destination = f" to {output}" if save_video else ""
    print(f"Rendering animation{destination} ...")
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

    # Render every reconstructed sample to preserve the full sampled path motion.
    frames = np.arange(len(path), dtype=int)
    frames = np.append(frames, [len(path) - 1] * 4)

    def update(frame_number: int):
        """Move animation artists to one selected path sample.

        Args:
            frame_number: Index into ``frames``, including repeated final frames.

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
            f"{env.title} - {gear}; margin={planner.safety_margin:.2f} m "
            f"({frame_number + 1}/{len(frames)})"
        )
        return trace, safety_envelope, car, *tires, heading_arrow

    animation = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    if save_video:
        start_time = time.time()
        try:
            writer, encoder = animation_writer(output.suffix.lower())
            animation.save(output, writer=writer)
        except Exception:
            warnings.warn(
                "Saving the animation failed.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            print(f"Animation saved in {time.time() - start_time:.2f} s " f"to {output} using {encoder}.")
    if show:
        print("You can close the windows to exit")
        plt.show(block=True)
    plt.close(fig)


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """Return whether FFmpeg can initialize the NVIDIA H.264 encoder.

    An encoder listed by FFmpeg may still be unusable when its driver or GPU is
    unavailable, so this performs a one-frame initialization probe.

    Returns:
        True when a high-quality NVENC probe completes successfully.
    """
    if not FFMpegWriter.isAvailable():
        return False
    command = [
        FFMpegWriter.bin_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=640x480:rate=1:duration=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def animation_writer(
    suffix: str,
    *,
    use_nvenc: bool | None = None,
) -> tuple[PillowWriter | FFMpegWriter, str]:
    """Create the high-quality writer for an animation output suffix.

    Args:
        suffix: Output suffix, including the leading period.
        use_nvenc: Whether to use NVIDIA NVENC for MP4 output. When omitted,
            availability is detected automatically.

    Returns:
        The configured Matplotlib writer and its encoder name.

    Raises:
        ValueError: If the output suffix is unsupported.
        RuntimeError: If MP4 output is requested without FFmpeg.
    """
    if suffix == ".gif":
        return PillowWriter(fps=12), "pillow"
    if suffix != ".mp4":
        raise ValueError(f"unsupported animation format: {suffix}")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("MP4 output requires FFmpeg; install FFmpeg or use --animation_format gif")
    if use_nvenc is None:
        use_nvenc = nvenc_available()
    common_args = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if use_nvenc:
        return (
            FFMpegWriter(
                fps=12,
                codec="h264_nvenc",
                extra_args=[
                    "-preset",
                    "p7",
                    "-tune",
                    "hq",
                    "-profile:v",
                    "high",
                    "-rc",
                    "vbr",
                    "-cq",
                    "16",
                    "-b:v",
                    "0",
                    "-multipass",
                    "fullres",
                    "-spatial-aq",
                    "1",
                    "-temporal-aq",
                    "1",
                    *common_args,
                ],
            ),
            "h264_nvenc",
        )
    return (
        FFMpegWriter(
            fps=12,
            codec="libx264",
            extra_args=["-preset", "slow", "-crf", "16", *common_args],
        ),
        "libx264",
    )


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
        choices=("walls", "maze", "parking", "parking2", "parking2_hard", "parking3", "parking4"),
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
        "--corridor_width",
        type=positive_float,
        default=None,
        help=(
            "Enable corridor-guided search and constrain the rear axle to this "
            "radius around a coarse 2D A* path [m]."
        ),
    )
    parser.add_argument(
        "--coarse_resolution",
        type=positive_float,
        default=1.0,
        help="Point-robot grid spacing used to build the coarse 2D A* path [m].",
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
        help="Motion-primitive length [m].",
    )
    parser.add_argument(
        "--two_queues",
        action="store_true",
        help=(
            "Enable the coarse acceleration queue. Without this flag, use the "
            "original single-queue search."
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
            "Multiplicative penalty applied to fine-generated nodes in the coarse "
            "queue; coarse-generated nodes keep their base priority. A value of 1 "
            "disables the bias."
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
            "Heuristic used for queue priorities: distance-only; distance plus heading "
            "(default/defaultw1); tolerance-aware kinematic lower bound; or the maximum "
            "of that bound and a relaxed 2D obstacle-grid estimate."
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
            "Additional action-set expansions after the first accepted goal while "
            "retaining the lowest-cost goal found."
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
            "Update an interactive plot every X action-set expansions showing open "
            "states as score heatmaps and closed states as gray dots, with full "
            "vehicle boxes on the open-set minimum-priority state and the minimum-h "
            "state across open and closed states. A third panel shows the relaxed "
            "Dijkstra cost-to-go grid only when --heuristic dijkstra is selected; "
            "0 disables live plotting."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--animation_format",
        choices=("mp4", "gif"),
        default="mp4",
        help=(
            "Video format used with --save_video. MP4 is the default and "
            "automatically uses NVIDIA NVENC when available; GIF uses Pillow."
        ),
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help=(
            "Save the final animation using --animation_format. Playback is "
            "controlled independently by --no_animation_plot."
        ),
    )
    parser.add_argument(
        "--no_animation_plot",
        action="store_true",
        help=(
            "Do not open the final animation playback window. Video saving is "
            "controlled independently by --save_video."
        ),
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
    """Run the selected scenario, write its outputs, and optionally play the animation.

    Args:
        args: Complete planner configuration, normally produced by ``parse_args``.

    Returns:
        Metrics and output paths for the completed planning run.

    Raises:
        ValueError: If start or goal is invalid.
        RuntimeError: If planning fails within the expansion limit.
    """
    # Collect the complete planner configuration stored with the environment.
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
    planner_kwargs = dict(
        environment=env,
        vehicle=Vehicle(),
        safety_margin=args.safety_margin,
        integration_step=args.integration_step,
        collision_check_step=args.collision_check_step,
        heuristic_mode=args.heuristic,
        state_key_mode=args.state_key_mode,
        heuristic_weight=args.heuristic_weight,
        corridor_width=getattr(args, "corridor_width", None),
        coarse_resolution=getattr(args, "coarse_resolution", 1.0),
    )
    if args.two_queues:
        planner = TwoQueueHybridAStar(
            **planner_kwargs,
            coarse_heuristic_weight=args.coarse_heuristic_weight,
            coarse_primitive_mult=args.coarse_primitive_mult,
            queue_beta=args.queue_beta,
            origin_priority_factor=args.origin_priority_factor,
        )
    else:
        planner = HybridAStar(**planner_kwargs)
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
        search_kwargs = dict(
            live_plot_every=args.live_plot_every,
            progress_callback=live_plot.update if live_plot is not None else None,
            expansion_callback=update_progress,
            post_goal_expansions=args.post_goal_expansions,
        )
        if isinstance(planner, TwoQueueHybridAStar):
            search_kwargs["max_consecutive_coarse_expansions"] = args.max_consecutive_coarse_expansions
        path, directions, steers, terminal = planner.plan(
            env.start,
            env.goal,
            args.max_expansions,
            **search_kwargs,
        )
    finally:
        # Keep the progress display consistent even when planning raises.
        progress_bar.update(planner.expansion_count - progress_bar.n)
        progress_bar.close()

    # Create output paths and always save the static path summary.
    env_output_dir = args.output_dir / env.name
    env_output_dir.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now().astimezone()
    timestamp = run_time.strftime("%Y_%m_%d_%H_%M_%S")
    output_stem = f"{planner.heuristic_mode}_{timestamp}"
    plot_path = env_output_dir / f"{output_stem}_path.png"
    animation_path = env_output_dir / f"{output_stem}_animation.{args.animation_format}"
    # Live search may have enabled pyplot's interactive mode. Disable it while
    # creating output figures so the final animation does not appear prematurely.
    # Video saving and blocking playback are controlled independently.
    with plt.ioff():
        save_plot(plot_path, planner, env, path)
        if args.save_video or not args.no_animation_plot:
            render_animation(
                animation_path,
                planner,
                env,
                path,
                directions,
                steers,
                save_video=args.save_video,
                show=not args.no_animation_plot,
            )

    # Compute exact primitive-arc length and the reconstructed sampled-polyline length.
    sampled_path_length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
    exact_path_length = planner.exact_path_length(terminal)
    position_error = math.hypot(path[-1, 0] - env.goal[0], path[-1, 1] - env.goal[1])
    yaw_error = math.degrees(abs(wrap(path[-1, 2] - env.goal[2])))
    result_path = env_output_dir / f"{output_stem}.json"
    arguments = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    two_queue_enabled = isinstance(planner, TwoQueueHybridAStar)
    fine_expansions = planner.fine_expansion_count if two_queue_enabled else planner.expansion_count
    coarse_expansions = planner.coarse_expansion_count if two_queue_enabled else 0
    corridor = getattr(planner, "corridor", None)
    result = {
        "environment": env.name,
        "timestamp": run_time.isoformat(),
        "arguments": arguments,
        "action_set_expansions": planner.expansion_count,
        "unique_expanded_state_keys": planner.unique_expanded_state_count,
        "fine_expansions": fine_expansions,
        "coarse_expansions": coarse_expansions,
        "path_samples": len(path),
        "path_length_m": exact_path_length,
        "sampled_path_length_m": sampled_path_length,
        "search_cost": terminal.cost,
        "terminal_error_m": position_error,
        "terminal_error_deg": yaw_error,
        "corridor_enabled": corridor is not None,
        "corridor_width_m": corridor.corridor_width if corridor is not None else None,
        "coarse_resolution_m": corridor.coarse_resolution if corridor is not None else None,
        "coarse_obstacle_clearance_m": (corridor.obstacle_clearance if corridor is not None else None),
        "coarse_path_nodes": len(corridor.coarse_path) if corridor is not None else 0,
        "coarse_path_length_m": corridor.path_length if corridor is not None else None,
    }
    result["plot"] = str(plot_path.resolve())
    if args.save_video:
        result["animation"] = str(animation_path.resolve())
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if __name__ == "__main__":
        # Print the run configuration, search outcome, and generated file locations.
        print(f"Environment: {env.name}")
        print(f"Action-set expansions: {planner.expansion_count}")
        print(f"Unique expanded state keys: {planner.unique_expanded_state_count}")
        print(f"Two queues: {two_queue_enabled}")
        print(f"Fine expansions: {fine_expansions}")
        print(f"Coarse expansions: {coarse_expansions}")
        print(f"Primitive length: {planner.primitive_length:.3f} m")
        if two_queue_enabled:
            print(f"Coarse primitive length: {planner.coarse_primitive_length:.3f} m")
            print(f"Queue beta: {planner.queue_beta:.3f}")
            print(f"Origin priority factor: {planner.origin_priority_factor:.3f}")
        print(f"Path samples: {len(path)}")
        print(f"Path length: {exact_path_length:.6f} m")
        print(f"Search cost: {terminal.cost:.2f}")
        print(f"Terminal error: {position_error:.3f} m / {yaw_error:.2f} deg")
        print(f"Safety margin: {planner.safety_margin:.3f} m")
        print(f"Integration step: {planner.integration_step:.3f} m")
        print(f"Collision-check step: {planner.collision_check_step:.3f} m")
        print(f"Corridor enabled: {corridor is not None}")
        if corridor is not None:
            print(f"Corridor width: {corridor.corridor_width:.3f} m")
            print(f"Coarse resolution: {corridor.coarse_resolution:.3f} m")
            print(f"Coarse obstacle clearance: {corridor.obstacle_clearance:.3f} m")
            print(f"Coarse path nodes: {len(corridor.coarse_path)}")
            print(f"Coarse path length: {corridor.path_length:.3f} m")
        print(f"Heuristic: {planner.heuristic_mode}")
        print(f"Heuristic weight: {planner.heuristic_weight:.3f}")
        if two_queue_enabled:
            print(f"Coarse heuristic weight: {planner.coarse_heuristic_weight:.3f}")
        print(f"State key mode: {planner.state_key_mode}")
        print(f"Post-goal action-set expansions: {args.post_goal_expansions}")
        print(f"Max consecutive coarse expansions: {args.max_consecutive_coarse_expansions}")
        print(f"Live plot interval: {args.live_plot_every} action-set expansions")
        print(f"Plot: {plot_path.resolve()}")
        if args.save_video:
            print(f"Animation: {animation_path.resolve()}")
        print(f"Results JSON: {result_path.resolve()}")
    return result


if __name__ == "__main__":
    main(parse_args())
