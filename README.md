# Path Planning Project

This project explores path-planning algorithms for a car-like robot in a 2D
environment. With the help of GPT 5.6, I implemented Hybrid A* with single-queue and optional two-queue search, collision-checked bicycle-model motion primitives, corridor-guided
search, plotting, animation, and interactive driving demos.


If you find an issue or have a question, please open an issue.

We first describe the project setup and usage, then explain the algorithms and implementation details.

## Quick start

Using Python 3.10 or newer, create and activate a virtual environment, then
install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Saving MP4 animations also requires FFmpeg.

## Usage

Run the planner:

```bash
python hybrid_astar_main.py --env parking
```

To favor a smoother path with fewer steering changes and gear shifts, enable
the corresponding penalties and use the control-aware state key:

```bash
python hybrid_astar_main.py --env parking --gear_change_penalty 0.2 --steering_change_penalty 0.2 --state_key_mode pose_control
```

To use a smaller state space at the cost of possibly merging different
control histories, omit the control-aware state key:

```bash
python hybrid_astar_main.py --env parking --gear_change_penalty 0.2 --steering_change_penalty 0.2
```

`pose_control` is recommended whenever gear-change or steering-change
penalties are nonzero (as explained later), but it can take more time and
memory.

To find a solution for `parking2_hard`, use a zero safety margin and a smaller
`xy_resolution`:

```bash
python hybrid_astar_main.py --env parking2_hard --safety_margin 0.0 --xy_resolution 0.1
```

For a smoother path, use `--steering_change_penalty 0.5`:

```bash
python hybrid_astar_main.py --env parking2_hard --safety_margin 0.0 --xy_resolution 0.1 --steering_change_penalty 0.5
```


For the maze, corridor guidance can substantially reduce search time:

```bash
python hybrid_astar_main.py --env maze --corridor_width 3 --heuristic dijkstra --heuristic_weight 1.3 --integration_step 0.2
```

A `heuristic_weight` greater than `1.0` can find a maze solution faster, but
the resulting search is not guaranteed to return the cheapest path.

The fine/coarse two-queue scheduler can also accelerate the maze search:

```bash
python hybrid_astar_main.py --env maze --corridor_width 3 --heuristic dijkstra --heuristic_weight 1.3 --integration_step 0.2 --two_queues
```

Gear-change and steering-change penalties can favor paths with fewer control
changes, possibly at the cost of a longer geometric path. This accelerated
configuration does not provide an optimality guarantee:

```bash
python hybrid_astar_main.py --env maze --corridor_width 3 --heuristic dijkstra --heuristic_weight 1.3 --integration_step 0.2 --gear_change_penalty 0.2 --steering_change_penalty 0.2
```

Note that `--state_key_mode pose_control` can make the planner slower by
retaining more control-specific states, so this speed-oriented example does
not use it.

After a successful search, the planner saves a timestamped path PNG and plays
the path animation in a blocking Matplotlib window. Close the window to let the
command finish. `--save_video` additionally saves the animation as MP4 (or GIF
with `--animation_format gif`). `--no_animation_plot` independently suppresses
the playback window. The `--live_plot_every` option is separate and controls
visualization while the search is running.

Run the interactive Python game with `python hybrid_astar_game.py` and try to
reach the goal manually with the arrow keys. Alternatively, open
`hybrid_astar_game_standalone.html` directly in a browser. Configure parameter
sweeps in `run_hybrid_astar_batch.py`. Use `search_json_arguments.py` to find
result JSON files that contain particular stored argument values.

## A*

A* is a graph-search algorithm that finds a path from a start state to a goal.
It assigns each candidate state a score `f = g + h`, where `g` is the cost of
the path from the start and `h` is a heuristic estimate of the remaining cost
to the goal. OPEN contains discovered states that are waiting to be expanded,
while CLOSED contains states that have already been expanded. Initially, OPEN
contains only the start state with `g = 0` and `f = h(start)`, and CLOSED is
empty. The algorithm repeatedly removes the state with the lowest score from
OPEN, expands its neighbors, and updates their best-known costs. The expanded
state is then placed in CLOSED, while newly discovered or improved neighbors
are placed in OPEN.

An **admissible** heuristic never overestimates the true remaining cost. A
**consistent** heuristic additionally satisfies `h(n) <= c(n, n') + h(n')`
for every edge from `n` to `n'`, with `h(goal) = 0` and `c(n, n')` the cost of the transition from `n` to `n'` (edge cost). Consistency means that
`f` cannot decrease along a path. Every consistent heuristic is admissible,
but an admissible heuristic is not necessarily consistent.

Assuming nonnegative edge costs and correct best-known-cost bookkeeping, A*
returns a lowest-cost path in the searched graph when the goal is removed from
OPEN if either:

- the heuristic is consistent; or
- the heuristic is admissible and a CLOSED state is reopened whenever a
  cheaper path to it is found.

The usual termination assumptions must also hold, such as searching a finite
graph, or having finite branching and edge costs bounded away from zero.

Grid-based A* searches discrete positions and is well suited to a point robot
that can move directly between neighboring cells. A car must also respect its
heading and steering constraints, so a position-only path may be impossible to
drive.

## Hybrid A*

Hybrid A* extends grid-based A* by propagating continuous, kinematically
feasible vehicle poses while discretizing the state space for OPEN and CLOSED
bookkeeping. A state usually includes `(x, y, yaw)` and may include additional
control history. As in A*, `g` is the accumulated path cost and `h` estimates
the remaining cost to the goal (heuristic). Successors are generated using feasible vehicle motions and checked for collisions with obstacles and planning boundaries.

## State-space discretization

In Hybrid A*, each successor is a continuous state produced by applying a
feasible motion from its parent state. Discretization maps each continuous
state to a discrete search key used for OPEN and CLOSED bookkeeping.

In a typical implementation, the world is divided into position cells and the
heading is divided into angular bins. A continuous pose `(x, y, yaw)` is mapped
to the corresponding `(x_index, y_index, yaw_index)` key. Heading bins are
cyclic because orientations separated by a full revolution are equivalent.
Implementations may use rounding, flooring, or other quantization rules, and
may add variables such as travel direction or steering history to the key.

Several continuous states can therefore share one search key. When that
happens, the search normally retains the best representative found for the
discretized state identified by that key. This merging keeps OPEN and CLOSED
finite and prevents the search from generating an unbounded number of nearly
identical continuous states. It also makes resolution important: smaller bins
preserve more distinctions between paths but require more memory and
expansions, while larger bins search faster but may merge states whose future
motions differ.

Discretization is used to organize the search: continuous poses that fall into
the same position and heading bins are treated as the same search state in OPEN
and CLOSED. The vehicle's state itself is still continuous and not restricted
to the grid. Successors are continuous, kinematically feasible motion
primitives that are sampled for collision checking. The resulting path follows
these continuous primitives rather than moving from one grid cell to another.

## Implementation in this project

This planner generates continuous successors using fixed-length,
constant-steering bicycle-model motion primitives. It reconstructs paths from
continuous poses and collision-checks samples along each primitive.

For a continuous state `(x, y, yaw)`, the planner rounds `x / xy_resolution`
and `y / xy_resolution` to integer position-bin indices. It wraps the heading
into `[-pi, pi)` and maps it to a cyclic yaw-bin index. The requested yaw
resolution is adjusted slightly, when needed, so an integer number of
equal-width bins covers exactly one full revolution. Equivalent headings on
opposite sides of the `-pi`/`pi` boundary therefore map consistently into the
cyclic key space.

When continuous states share a key, this planner retains the lowest-cost
representative found for that discretized state and discards or replaces the
others. The actual OPEN heap, or heaps in two-queue mode, can temporarily retain
older entries for a replaced representative. The planner recognizes and skips
these stale entries when inspecting or popping a queue.

In `pose` mode, the key contains only the discretized `(x, y, yaw)` indices. In
`pose_control` mode, it additionally contains the incoming travel direction and
steering index. This makes it a Markov representation when gear-change or
steering-change penalties are enabled: the stored state contains all the
information needed to calculate the cost of the next motion, without needing
any earlier history. In particular, change penalties depend on the direction
and steering with which the vehicle entered the pose.

The tradeoff is search size. `pose` keeps at most one current representative
for each discretized pose, while `pose_control` can keep separate
representatives for two travel directions and five steering values. It can
therefore generate and expand more states, use more memory, and take longer.
Without `pose_control`, a search with change penalties can still find a valid
path, but it may merge arrivals with different future change costs and discard
a control history that would have produced a cheaper or smoother continuation.

## Car-like robot

A car-like robot cannot translate sideways or rotate in place. It moves along
its current heading, changes heading by steering while moving, and has a
minimum turning radius. This is a nonholonomic motion constraint: the vehicle's
instantaneous velocity is constrained by its heading, so it cannot move
directly between arbitrary nearby poses. Its position and heading must instead
evolve together along a feasible forward or reverse curve. This planner
supports both directions and can assign extra costs to reversing, changing
gear, or changing steering. The goal therefore includes both a rear-axle
position and a vehicle heading.

## Simplified bicycle model

The kinematic bicycle model replaces the left and right wheels on each axle
with one rear wheel and one steerable front wheel. The vehicle pose is measured
at the center of the rear axle. The wheelbase is the distance between the front
and rear axles. For the same steering angle, a vehicle with a longer wheelbase
makes a wider turn, while one with a shorter wheelbase makes a tighter turn.
Holding the steering angle constant produces an exact
straight line or constant-curvature arc.

![Kinematic bicycle model showing the wheelbase and steering angle](https://av1tenth-docs.readthedocs.io/en/latest/_images/bicyle_diagram.png)

*Diagram source: [AV1Tenth Bicycle Kinematic Model](https://av1tenth-docs.readthedocs.io/en/latest/information/theoryinfo/cyckinem.html).*

This is a geometric, low-speed model: it ignores acceleration, tire slip,
weight transfer, suspension, and other vehicle dynamics. Collision checking
uses a rectangular vehicle body, expanded by the configured safety margin, at
samples along each motion primitive.

## Single-queue search

The standard planner has one OPEN queue containing states waiting to be
expanded. An action is a choice of travel direction and steering angle.
Applying an action to the current state produces a motion primitive: a short,
kinematically feasible straight line or curved arc that the vehicle can follow.

The planner has ten actions, formed by combining two travel directions (forward
and reverse) with five evenly spaced steering angles (full right, intermediate
right, straight, intermediate left, and full left). Each resulting motion
primitive has `primitive_length` meters of arc length. Its endpoint is
calculated exactly; `collision_check_step` controls only how densely the arc is
checked, and `integration_step` controls only how densely an accepted path is
reconstructed for visualization.

OPEN is ordered by `g + heuristic_weight * h`, where `g` includes travel,
reverse, gear-change, and steering-change costs. At each iteration, the planner
removes and processes the state with the lowest value of that priority. If both
position and yaw are within their tolerances, the state is accepted as a goal
and no successors are generated from it. After the first accepted goal, the
planner performs `post_goal_expansions` additional counted state expansions,
meaning valid states popped from OPEN, and returns the cheapest goal found
during that budget. With the default value of zero, it returns the first
accepted goal.

Whenever a successor has a lower `g` than the current representative for its
discrete key, the planner replaces that representative, removes the key from
CLOSED, and inserts the improved state into OPEN. Thus a previously expanded
key is reopened when a cheaper route to it is found.

In single-queue mode, selecting an admissible heuristic with weight `1`, for
example `--heuristic tolerance --heuristic_weight 1`, gives ordinary A*
ordering by `g + h`. Because the planner expands the lowest-priority state and
reopens a CLOSED state when a cheaper route to it is found, ordinary A* would
return a lowest-cost path on a fixed graph under the assumptions described
above.

Hybrid A* adds an important limitation to that result: several continuous
poses can map to the same discrete state key, and the planner keeps only the
cheapest pose found for each key. A discarded pose might allow a cheaper
continuation later. The result is therefore lowest-cost only within the
planner's retained discrete search under this representative-merging
assumption; it is not a proof of global optimality over every possible
continuous vehicle trajectory.

Including incoming direction and steering in the key with `pose_control` is
necessary to represent gear-change and steering-change costs correctly. If
either control-change penalty is nonzero and the A* lowest-cost argument above
is desired, use `--state_key_mode pose_control`. This makes those transition
costs Markov, but it does not eliminate the approximation caused by pose
discretization.

## Two-queue search

`--two_queues` adds a coarse action set to accelerate exploration without
removing the fine actions:

- The fine queue uses `primitive_length`, five steering values, and both travel
  directions, so a fine expansion attempts ten actions.
- The coarse queue uses
  `coarse_primitive_mult * primitive_length`, full-right/straight/full-left
  steering, and both directions, so a coarse expansion attempts six actions.

The queues are two priority views of the same generated states; they do not
partition states by generation origin. In particular, the coarse queue does
not contain only states produced by coarse actions. The planner keeps one
shared best node for each discretized state key, including its cost, parent,
and incoming action. Whenever either a fine or coarse expansion produces a
collision-free successor that is new or cheaper than the stored node for its
key, the planner:

1. updates the shared best-node mapping;
2. inserts that same node into the fine OPEN queue with its fine priority; and
3. inserts that same node into the coarse OPEN queue with its coarse priority.

A generated successor that collides or does not improve its state key is not
inserted.

The two queues have separate CLOSED sets because they represent pending action
sets, not separate state graphs. Popping a node from the fine queue marks only
its fine view closed and expands it with the ten fine actions. Its coarse view
remains available for a later expansion with the six coarse actions. Popping
from the coarse queue behaves symmetrically. If a cheaper node replaces a
state, both CLOSED marks are cleared and the replacement is inserted into both
OPEN queues. An accepted goal is instead closed in both views because it is not
expanded further.

For any current best representative of a state key, each queue view can be
expanded at most once: once with fine actions and once with coarse actions.
Either expansion may never occur if the search terminates first, and an
accepted goal receives neither successor expansion. Finding a cheaper
representative resets both views, so the state key may be expanded again. The
code and result fields call each valid state popped from either OPEN queue an
“action-set expansion”, including an accepted terminal pop for which no
successor actions are attempted. `fine_expansions` and `coarse_expansions`
identify which queue supplied those pops.

Both queues remain available throughout the search, but each iteration selects
only one of them and performs one action-set expansion. The coarse queue can be
selected when:

```text
minimum coarse priority <= queue_beta * minimum fine priority
```

If this condition is false, the fine queue is selected. If it is true, the
coarse queue is normally selected; however, the coarse-burst limit described
below can force that iteration to use the fine queue instead. If the fine queue
is empty and the coarse queue is not, the coarse queue is selected. Selecting
one queue does not discard or disable the other queue: its states remain
available for later iterations.

The fine priority is `g + heuristic_weight * h`. The coarse priority uses
`coarse_heuristic_weight`, or the fine weight when that option is omitted.
When a node was generated by a fine action, its coarse-queue priority is also
multiplied by `origin_priority_factor`; coarse-generated nodes are not
penalized. This encourages the long-action frontier to develop without making
coarse search follow every fine branch. To prevent fine work from starving,
`max_consecutive_coarse_expansions` forces the next iteration to select the fine
queue after the selected number of consecutive coarse expansions whenever a
fine state remains.

Two-queue mode uses the same `post_goal_expansions` termination policy as
single-queue mode.

## Corridor-guided search

`--corridor_width WIDTH` adds a four-step guide:

1. Obstacles and world boundaries are inflated by half the vehicle width plus
   `safety_margin`, and an eight-connected point-robot A* searches a grid
   spaced by `coarse_resolution`.
2. The exact start and goal are connected to valid corners of their containing
   coarse cells. Connectors and grid edges are checked against the inflated
   obstacle boxes, and diagonal moves cannot cut between blocked neighbors.
3. The resulting coarse-grid path, including its exact start and goal
   connectors, becomes the centerline of a closed tube with radius
   `corridor_width`.
4. The detailed Hybrid A* search rejects a primitive if any sampled rear-axle
   position leaves that tube. With the Dijkstra heuristic, its relaxed grid is
   restricted to the same tube.

The coarse route is a guide, not a drivable solution: it ignores heading,
steering, and the vehicle's longitudinal footprint. Hybrid A* still performs
the normal safety-inflated rectangular collision checks. A narrow corridor can
therefore exclude a valid maneuver, while a coarse grid can fail to find a
connector or route even when one exists. In those cases, increase
`corridor_width`, reduce `coarse_resolution`, or disable corridor mode.

## `hybrid_astar_main.py` options

All distances are in meters and all CLI angles are in degrees. The planner
converts angles to radians internally.

### Scene, geometry, and sampling

| Option | Default | Meaning |
| --- | ---: | --- |
| `-h`, `--help` | N/A | Print the complete command help and exit. |
| `--env` | `parking` | Scene name: `walls`, `maze`, `parking`, `parking2`, `parking2_hard`, `parking3`, or `parking4`. |
| `--safety_margin` | `0.20` | Clearance added to every side of the rectangular vehicle for collision checks. |
| `--integration_step` | `0.10` | Spacing of reconstructed path and animation samples. It does not change primitive endpoints. |
| `--collision_check_step` | `0.05` | Spacing of swept-path collision samples. Smaller values check an arc more densely. |
| `--xy_resolution` | `0.15` | Position-bin resolution used in state keys. |
| `--yaw_resolution_deg` | `1.0` | Requested heading-bin resolution. It is adjusted slightly so a whole number of equal bins covers 360 degrees. |
| `--primitive_length` | `0.20` | Arc length of every fine motion primitive. It must be at least `xy_resolution`. |
| `--position_tolerance` | `0.20` | Maximum rear-axle distance from the nominal goal. |
| `--yaw_tolerance_deg` | `1.5` | Maximum absolute wrapped heading error at the goal. |

### Cost, state, and heuristic

| Option | Default | Meaning |
| --- | ---: | --- |
| `--reverse_multiplier` | `1.0` | Per-meter reverse cost multiplier; forward motion has multiplier 1. |
| `--gear_change_penalty` | `0.0` | Event cost added when consecutive edges change travel direction. |
| `--steering_change_penalty` | `0.0` | Event cost for changing steering, normalized to the fine primitive scale. The initial steering state is straight. |
| `--state_key_mode` | `pose` | `pose` keys use discretized `(x, y, yaw)` only. `pose_control` also stores incoming direction and steering, making control-change costs Markov and avoiding the merging of arrivals with different future change costs. |
| `--heuristic` | `default` | Queue estimate; modes are explained below. |
| `--heuristic_weight` | `1.0` | Multiplier on `h` in the fine/main priority `g + weight * h`. Zero gives cost-only ordering. |

The heuristic choices are:

- `distance`: Euclidean rear-axle distance to the nominal goal.
- `default`: distance plus `0.5 * wheelbase * heading_error`.
- `defaultw1`: distance plus `wheelbase * heading_error`.
- `tolerance`: an admissible, tolerance-aware lower bound based on the remaining
  position and heading error.
- `dijkstra`: the maximum of the tolerance bound and an obstacle-aware,
  eight-connected point-robot cost-to-go grid. It falls back to the tolerance
  bound where the raster grid appears disconnected.

These heuristics control queue ordering only. The `tolerance` heuristic is
admissible; `distance`, `default`, `defaultw1`, and `dijkstra` are useful
ordering estimates but are not guaranteed to be admissible for every goal
region and cost configuration supported by the planner.

### Two-queue and corridor options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--two_queues` | off | Enable the fine/coarse scheduler described above. |
| `--coarse_primitive_mult` | `4` | Integer coarse/fine primitive-length ratio. Used only in two-queue mode. |
| `--queue_beta` | `1.5` | Coarse eligibility factor; it must be at least 1. |
| `--origin_priority_factor` | `2.0` | Coarse-queue priority multiplier for nodes produced by fine actions. A value of 1 disables this bias. |
| `--coarse_heuristic_weight` | fine weight | Optional `h` multiplier for the coarse priority. |
| `--max_consecutive_coarse_expansions` | `10` | Largest coarse burst while fine work remains. Zero forces fine selection whenever possible. |
| `--corridor_width` | disabled | Enable corridor mode with this rear-axle tube radius. |
| `--coarse_resolution` | `1.0` | Grid spacing for corridor centerline A*. It has no effect when corridor mode is disabled. |

### Termination, visualization, and output

| Option | Default | Meaning |
| --- | ---: | --- |
| `--post_goal_expansions` | `0` | Additional counted state expansions after the first accepted goal while retaining the cheapest goal found. |
| `--max_expansions` | `1000000` | Hard cap on counted state expansions. If a goal exists at the cap it is returned with a warning when the selected termination condition is unfinished. |
| `--live_plot_every` | `100000` | Refresh the interactive search view every N expansions; zero disables it. Dijkstra mode adds a cost-grid panel. |
| `--output_dir` | `./results` | Parent directory; each environment gets its own subdirectory. |
| `--animation_format` | `mp4` | Video format used with `--save_video`: `mp4` uses FFmpeg and prefers NVIDIA NVENC when available; `gif` uses Pillow. |
| `--save_video` | off | Save the final path animation. This does not affect interactive playback. |
| `--no_animation_plot` | off | Do not play the final path animation in a Matplotlib window. This does not affect video saving. |

Each JSON result records the full argument set, expansion counts, exact
primitive path length, sampled path length, search cost, terminal errors, and
corridor metrics. It also records the path PNG location and, when
`--save_video` is enabled, the animation location.
