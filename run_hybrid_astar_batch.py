#!/usr/bin/env python3
"""Run every configured two-queue Hybrid A* parameter combination.

Edit the lists below to define the parameter sweep. Each environment is run for
every Cartesian-product combination of ``PARAMETER_VALUES``.
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from hybrid_astar_demo import main

ENVIRONMENTS = [
    "walls",
    "parking",
    "parking2",
    "parking2_2",
    "parking3",
    "parking4",
]

# Keep every value in a list, even when the sweep currently contains one value.
# Add values to any list to include them in the Cartesian-product sweep.
PARAMETER_VALUES = {
    "safety_margin": [0.0],
    "integration_step": [0.10],
    "collision_check_step": [0.05],
    "xy_resolution": [0.15],
    "yaw_resolution_deg": [1.0],
    "primitive_length": [0.20],
    "two_queues": [False, True],
    "coarse_primitive_mult": [4],
    "queue_beta": [1.0, 1.1],
    "origin_priority_factor": [1.0, 2.0],
    "position_tolerance": [0.20],
    "yaw_tolerance_deg": [1.5],
    "reverse_multiplier": [1.0, 1.1],
    "gear_change_penalty": [0.0, 0.5],
    "steering_change_penalty": [0.0, 0.5],
    "state_key_mode": ["pose"],
    "heuristic": ["default", "tolerance"],
    "heuristic_weight": [1.0, 1.2],
    "coarse_heuristic_weight": [None, 1.6],
    "post_goal_expansions": [200000],
    "enable_admissible_bound": [False, True],
    "max_expansions": [1_000_000],
    "max_consecutive_coarse_expansions": [10],
    "live_plot_every": [0],
    "no_animation": [True],
}
Result = dict[str, object]
BestResults = dict[str, dict[str, list[Result]]]

METRICS = {
    "path_length_m": ("Path length", "m"),
    "search_cost": ("Search cost", ""),
    "action_set_expansions": ("Action-set expansions", ""),
}

RESULT_METRIC_FIELDS = (
    "path_length_m",
    "search_cost",
    "action_set_expansions",
    # "unique_expanded_state_keys",
    # "fine_expansions",
    # "coarse_expansions",
    # "path_samples",
    # "sampled_path_length_m",
    # "terminal_error_m",
    # "terminal_error_deg",
    "error",
)

QUEUE_ONLY_PARAMETERS = (
    "coarse_primitive_mult",
    "queue_beta",
    "origin_priority_factor",
    "coarse_heuristic_weight",
    "max_consecutive_coarse_expansions",
)
FINE_ONLY_PARAMETERS = ("enable_admissible_bound",)


def _uses_first_parameter_value(arguments: argparse.Namespace, name: str) -> bool:
    """Return whether an argument uses the first configured candidate value.

    Args:
        arguments: Candidate batch-run arguments.
        name: Parameter name to inspect.

    Returns:
        Whether the value is first in its configured candidate list.
    """
    values = PARAMETER_VALUES.get(name)
    return values is None or getattr(arguments, name) == values[0]


def is_redundant_combination(arguments: argparse.Namespace) -> bool:
    """Return whether a configuration differs only in inactive parameters.

    Args:
        arguments: Candidate batch-run arguments.

    Returns:
        Whether the candidate would duplicate a run with active parameters unchanged.
    """
    if not getattr(arguments, "two_queues", True) and any(
        not _uses_first_parameter_value(arguments, name) for name in QUEUE_ONLY_PARAMETERS
    ):
        return True
    if getattr(arguments, "two_queues", False) and any(
        not _uses_first_parameter_value(arguments, name) for name in FINE_ONLY_PARAMETERS
    ):
        return True
    return getattr(arguments, "enable_admissible_bound", False) and not _uses_first_parameter_value(
        arguments, "post_goal_expansions"
    )


def argument_combinations() -> Iterator[argparse.Namespace]:
    """Yield non-redundant ``main`` argument namespaces for configured runs.

    Yields:
        Planner configurations for all environments and parameter combinations.
    """
    names = tuple(PARAMETER_VALUES)
    for environment in ENVIRONMENTS:
        for values in product(*(PARAMETER_VALUES[name] for name in names)):
            arguments = argparse.Namespace(env=environment, **dict(zip(names, values)))
            if not is_redundant_combination(arguments):
                yield arguments


def configuration_key(arguments: argparse.Namespace) -> str:
    """Build a stable identity for one configured run.

    Args:
        arguments: Complete batch-run arguments.

    Returns:
        A JSON key containing the environment and swept parameter values.
    """
    configuration = {
        "environment": arguments.env,
        "parameters": {name: getattr(arguments, name) for name in PARAMETER_VALUES},
    }
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def result_configuration_key(result: Result) -> str:
    """Build a configured-run identity from a stored result.

    Args:
        result: Previously saved result dictionary.

    Returns:
        A JSON key containing the environment and swept parameter values.

    Raises:
        ValueError: If the stored result lacks required configuration fields.
    """
    arguments = result.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("stored result is missing an arguments dictionary")
    try:
        configuration = {
            "environment": result["environment"],
            "parameters": {name: arguments[name] for name in PARAMETER_VALUES},
        }
    except KeyError as exc:
        raise ValueError(f"stored result is missing configuration field {exc.args[0]!r}") from exc
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def load_results(path: Path) -> list[Result]:
    """Load completed results when an aggregate file already exists.

    Args:
        path: Aggregate JSON path.

    Returns:
        Previously completed results, or an empty list when the file is absent.

    Raises:
        ValueError: If the aggregate JSON is not a list of dictionaries.
    """
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(isinstance(result, dict) for result in loaded):
        raise ValueError(f"{path} must contain a JSON list of result dictionaries")
    return loaded


def failed_result(arguments: argparse.Namespace, error: Exception) -> Result:
    """Create a persisted result record for a planning run that did not finish.

    Args:
        arguments: Complete arguments passed to the failed planner run.
        error: Planning failure raised by ``main``.

    Returns:
        A result dictionary with unavailable search metrics set to ``None``.
    """
    serialized_arguments = {
        name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()
    }
    return {
        "environment": arguments.env,
        "timestamp": datetime.now().astimezone().isoformat(),
        "arguments": serialized_arguments,
        "action_set_expansions": None,
        "unique_expanded_state_keys": None,
        "fine_expansions": None,
        "coarse_expansions": None,
        "path_samples": None,
        "path_length_m": None,
        "sampled_path_length_m": None,
        "search_cost": None,
        "terminal_error_m": None,
        "terminal_error_deg": None,
        "error": str(error),
    }


def save_results(path: Path, results: list[Result]) -> None:
    """Atomically save all completed batch results.

    Args:
        path: Aggregate JSON path.
        results: Completed result dictionaries.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def save_results_csv(aggregate_path: Path, results: list[Result]) -> Path:
    """Write aggregate results as a CSV alongside the aggregate JSON.

    Metric columns are followed by varying sweep arguments, then arguments with a
    single configured value. This keeps the columns that distinguish runs first
    while retaining the fixed planner configuration.

    Args:
        aggregate_path: Aggregate JSON path whose extension is replaced with ``.csv``.
        results: Completed or failed batch-run result dictionaries.

    Returns:
        The written CSV path.
    """
    csv_path = aggregate_path.with_suffix(".csv")
    varying_arguments = [name for name, values in PARAMETER_VALUES.items() if len(values) > 1]
    fixed_arguments = [name for name, values in PARAMETER_VALUES.items() if len(values) == 1]
    fieldnames = ["environment", *RESULT_METRIC_FIELDS, *varying_arguments, *fixed_arguments]
    environment_order = {environment: index for index, environment in enumerate(ENVIRONMENTS)}

    def csv_sort_key(result: Result) -> tuple[int, str, bool, float, bool, float]:
        """Return the CSV ordering key for a result."""
        environment = str(result.get("environment", ""))
        path_length = result.get("path_length_m")
        is_missing_length = not isinstance(path_length, (int, float)) or not math.isfinite(path_length)
        action_set_expansions = result.get("action_set_expansions")
        is_missing_expansions = not isinstance(action_set_expansions, (int, float)) or not math.isfinite(
            action_set_expansions
        )
        return (
            environment_order.get(environment, len(environment_order)),
            environment,
            is_missing_length,
            float(path_length) if not is_missing_length else math.inf,
            is_missing_expansions,
            float(action_set_expansions) if not is_missing_expansions else math.inf,
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(results, key=csv_sort_key):
            arguments = result.get("arguments")
            row = {
                "environment": result.get("environment"),
                **{field: result.get(field) for field in RESULT_METRIC_FIELDS},
            }
            if isinstance(arguments, dict):
                row.update({name: arguments.get(name) for name in PARAMETER_VALUES})
            writer.writerow(row)
    temporary_path.replace(csv_path)
    return csv_path


def update_best_results(best_results: BestResults, result: Result) -> bool:
    """Update an environment's best results for each tracked metric.

    Args:
        best_results: Mutable best-results mapping grouped by environment and metric.
        result: Newly available run result.

    Returns:
        Whether the result set a new best value or tied an existing best value for
        at least one tracked metric.
    """
    improved = False
    eps = 1e-2
    for metric in METRICS:
        value = result.get(metric)
        if value is None:
            continue
        environment = str(result["environment"])
        environment_bests = best_results.setdefault(environment, {})
        previous = environment_bests.get(metric)
        previous_best_value = (
            min(previous_result[metric] for previous_result in previous) if previous else None
        )
        if previous_best_value is None or value < previous_best_value - eps:
            environment_bests[metric] = [result]
            improved = True
        elif abs(value - previous_best_value) < eps:
            previous.append(result)
            improved = True
        # Remove any previous results that are no longer within ``eps`` of the best value.
        metric_results = environment_bests.get(metric)
        assert metric_results is not None
        best_value = min(best_result[metric] for best_result in metric_results)
        environment_bests[metric] = [
            best_result for best_result in metric_results if best_result[metric] < best_value + eps
        ]
    return improved


def _varying_arguments(result: Result) -> str:
    """Format only arguments that have multiple configured candidate values.

    Args:
        result: Result whose arguments should be formatted.

    Returns:
        Comma-separated varying arguments, or an empty string if none vary.
    """
    arguments = result["arguments"]
    assert isinstance(arguments, dict)
    return " ".join(
        f"{name}={arguments[name]!r}"
        for name, values in PARAMETER_VALUES.items()
        if len(values) > 1
        and (arguments.get("two_queues", True) or name not in QUEUE_ONLY_PARAMETERS)
        and (not arguments.get("two_queues", False) or name not in FINE_ONLY_PARAMETERS)
    )


def print_environment_bests(environment: str, best_results: BestResults, heading: str) -> None:
    """Print all tracked bests for one environment.

    Args:
        environment: Environment whose bests should be printed.
        best_results: Best-result mapping grouped by environment and metric.
        heading: Heading displayed before the metric lines.

    Returns:
        None.
    """
    print(f"{heading}: {environment}")
    for metric, (label, unit) in METRICS.items():
        results = best_results[environment][metric]
        value = min(result[metric] for result in results)
        suffix = f" {unit}" if unit else ""
        run_count = len(results)
        runs_suffix = f" ({run_count} tied runs)" if run_count > 1 else ""
        print(f"  {label}: {value}{suffix}{runs_suffix}")
        for result in results:
            varying_arguments = _varying_arguments(result)
            if varying_arguments:
                print(f"{result[metric]} - {varying_arguments}")


def run_configuration(
    index: int,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> tuple[int, argparse.Namespace, Result]:
    """Run one configuration with a dedicated output directory.

    Args:
        index: One-based position in the complete configuration list.
        arguments: Planner arguments for the configuration.
        output_directory: Root directory for all batch-run output.

    Returns:
        The configuration index, its arguments, and its completed or failed result.
    """
    arguments.output_dir = output_directory / f"run_{index:04d}"
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                result = main(arguments)
    except Exception as error:
        result = failed_result(arguments, error)
    return index, arguments, result


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the batch runner.

    Returns:
        The populated batch-run argument namespace.

    Raises:
        SystemExit: When argparse rejects command-line input.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hybrid_a_star"),
        help="Parent directory for the named batch-run output directory.",
    )
    parser.add_argument(
        "--name",
        default="all_results",
        help="Batch run name, used for the output subdirectory and aggregate filenames.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=10,
        help="Number of independent planner processes to run concurrently",
    )
    parser.add_argument(
        "--read_only",
        action="store_true",
        help="Do not run new planners; regenerate the aggregate CSV from existing JSON.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    """Parse an argparse value as a strictly positive integer.

    Args:
        value: Command-line token to parse and validate.

    Returns:
        The parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the number is not positive.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def run_all(
    output_directory: Path,
    name: str,
    workers: int,
    read_only: bool,
) -> list[Result]:
    """Execute every configured environment and parameter combination.

    Args:
        output_directory: Parent directory for the named batch-run output directory.
        name: Batch run name, used for the output subdirectory and aggregate filename.
        workers: Number of planner processes to run concurrently.

    Returns:
        Metrics and output paths for every completed planning run.
    """
    output_directory = output_directory / name
    aggregate_path = output_directory / f"{name}.json"
    results = load_results(aggregate_path)
    completed_keys = {result_configuration_key(result) for result in results}
    best_results: BestResults = {}
    for result in results:
        update_best_results(best_results, result)

    configurations = list(argument_combinations())
    if results:
        print(f"Loaded {len(results)} completed runs from {aggregate_path.resolve()}")
        print("=== Best so far from loaded results ===")
        for environment in ENVIRONMENTS:
            if environment in best_results:
                print_environment_bests(environment, best_results, "Best so far")
    if read_only:
        print("Read-only mode: no new runs will be executed")
        csv_path = save_results_csv(aggregate_path, results)
        print(f"Aggregate results CSV: {csv_path.resolve()}")
        return results
    pending = [
        (index, arguments)
        for index, arguments in enumerate(configurations, start=1)
        if configuration_key(arguments) not in completed_keys
    ]
    for index, arguments in enumerate(configurations, start=1):
        if configuration_key(arguments) in completed_keys:
            print(f"Skipping completed run {index}/{len(configurations)}: {arguments.env}")

    def record_result(index: int, arguments: argparse.Namespace, result: Result) -> None:
        """Persist and report one completed planner result."""
        if result.get("error"):
            print(f"Planning failed for run {index}/{len(configurations)}: {result['error']}")
        results.append(result)
        completed_keys.add(configuration_key(arguments))
        save_results(aggregate_path, results)
        if update_best_results(best_results, result):
            print()
            print_environment_bests(arguments.env, best_results, "Best so far")

    if pending:
        with tqdm(total=len(pending), desc="Pending runs", unit="run") as progress_bar:
            if workers == 1:
                for index, arguments in pending:
                    print()
                    print(f"=== Run {index}/{len(configurations)}: {arguments.env} ===")
                    _, completed_arguments, result = run_configuration(index, arguments, output_directory)
                    record_result(index, completed_arguments, result)
                    progress_bar.update()
            else:
                print(f"Running {len(pending)} remaining configurations with {workers} worker processes")
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(run_configuration, index, arguments, output_directory): index
                        for index, arguments in pending
                    }
                    for future in as_completed(futures):
                        index, completed_arguments, result = future.result()
                        print()
                        print(
                            f"=== Completed run {index}/{len(configurations)}: "
                            f"{completed_arguments.env} ==="
                        )
                        record_result(index, completed_arguments, result)
                        progress_bar.update()

    csv_path = save_results_csv(aggregate_path, results)
    print()
    print(f"Aggregate results JSON: {aggregate_path.resolve()}")
    print(f"Aggregate results CSV: {csv_path.resolve()}")
    print()
    print("=== Best results by environment ===")
    for environment in ENVIRONMENTS:
        if environment in best_results:
            print_environment_bests(environment, best_results, "Best")
    return results


if __name__ == "__main__":
    args = parse_args()
    run_all(
        output_directory=args.output_dir,
        name=args.name,
        workers=args.workers,
        read_only=args.read_only,
    )
