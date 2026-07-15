import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import run_hybrid_astar_batch as batch


def test_save_results_csv_orders_metrics_and_arguments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        batch,
        "PARAMETER_VALUES",
        {"varied": [1, 2], "fixed": [10]},
    )
    results: list[batch.Result] = [
        {
            "environment": "test_env",
            "arguments": {"varied": 2, "fixed": 10},
            "path_length_m": 1.5,
            "search_cost": 3.0,
            "action_set_expansions": 4,
        }
    ]

    csv_path = batch.save_results_csv(tmp_path / "all_results.json", results)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert reader.fieldnames == [
            "environment",
            *batch.RESULT_METRIC_FIELDS,
            "varied",
            "fixed",
        ]
    assert rows == [
        {
            "environment": "test_env",
            **{
                field: str(results[0][field]) if field in results[0] else ""
                for field in batch.RESULT_METRIC_FIELDS
            },
            "varied": "2",
            "fixed": "10",
        }
    ]
    assert csv_path == tmp_path / "all_results.csv"


def test_save_results_csv_groups_environments_and_sorts_path_length(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(batch, "ENVIRONMENTS", ["parking", "walls"])
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {})
    results: list[batch.Result] = [
        {"environment": "walls", "path_length_m": 2.0, "action_set_expansions": 20},
        {"environment": "parking", "path_length_m": 3.0},
        {"environment": "walls", "path_length_m": 1.0, "action_set_expansions": 20},
        {"environment": "walls", "path_length_m": 1.0, "action_set_expansions": 10},
        {"environment": "walls", "path_length_m": None},
    ]

    csv_path = batch.save_results_csv(tmp_path / "all_results.json", results)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [
        (row["environment"], row["path_length_m"], row["action_set_expansions"])
        for row in rows
    ] == [
        ("parking", "3.0", ""),
        ("walls", "1.0", "10"),
        ("walls", "1.0", "20"),
        ("walls", "2.0", "20"),
        ("walls", "", ""),
    ]


def test_argument_combinations_skip_inactive_queue_and_bound_options(monkeypatch) -> None:
    monkeypatch.setattr(batch, "ENVIRONMENTS", ["test_env"])
    monkeypatch.setattr(
        batch,
        "PARAMETER_VALUES",
        {
            "two_queues": [False, True],
            "coarse_primitive_mult": [4, 8],
            "queue_beta": [1.0, 1.5],
            "origin_priority_factor": [1.0, 2.0],
            "max_consecutive_coarse_expansions": [0, 10],
            "post_goal_expansions": [0, 1],
            "enable_admissible_bound": [False, True],
        },
    )

    combinations = list(batch.argument_combinations())

    fine_only = [arguments for arguments in combinations if not arguments.two_queues]
    two_queue = [arguments for arguments in combinations if arguments.two_queues]
    assert len(fine_only) == 3
    assert len(two_queue) == 32
    assert all(
        arguments.coarse_primitive_mult == 4
        and arguments.queue_beta == 1.0
        and arguments.origin_priority_factor == 1.0
        and arguments.max_consecutive_coarse_expansions == 0
        for arguments in fine_only
    )
    assert all(not arguments.enable_admissible_bound for arguments in two_queue)
    assert all(
        arguments.post_goal_expansions == 0
        for arguments in fine_only
        if arguments.enable_admissible_bound
    )


def test_varying_arguments_omit_queue_options_for_fine_only_results(monkeypatch) -> None:
    monkeypatch.setattr(
        batch,
        "PARAMETER_VALUES",
        {
            "two_queues": [False, True],
            "queue_beta": [1.0, 1.5],
            "origin_priority_factor": [1.0, 2.0],
            "enable_admissible_bound": [False, True],
            "reverse_multiplier": [1.0, 1.1],
        },
    )
    fine_only_result: batch.Result = {
        "arguments": {
            "two_queues": False,
            "queue_beta": 1.0,
            "origin_priority_factor": 1.0,
            "enable_admissible_bound": False,
            "reverse_multiplier": 1.1,
        }
    }

    formatted_arguments = batch._varying_arguments(fine_only_result)

    assert "two_queues=False" in formatted_arguments
    assert "reverse_multiplier=1.1" in formatted_arguments
    assert "queue_beta" not in formatted_arguments
    assert "origin_priority_factor" not in formatted_arguments


def test_varying_arguments_omit_bound_option_for_two_queue_results(monkeypatch) -> None:
    monkeypatch.setattr(
        batch,
        "PARAMETER_VALUES",
        {
            "two_queues": [False, True],
            "enable_admissible_bound": [False, True],
        },
    )
    result: batch.Result = {
        "arguments": {"two_queues": True, "enable_admissible_bound": False}
    }

    formatted_arguments = batch._varying_arguments(result)

    assert "two_queues=True" in formatted_arguments
    assert "enable_admissible_bound" not in formatted_arguments


def test_update_best_results_retains_all_tied_runs_and_prints_them(monkeypatch, capsys) -> None:
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {"variant": ["first", "second"]})
    first: batch.Result = {
        "environment": "test_env",
        "arguments": {"variant": "first"},
        "path_length_m": 1.0005,
        "search_cost": 2.0005,
        "action_set_expansions": 3.0005,
    }
    tied: batch.Result = {
        "environment": "test_env",
        "arguments": {"variant": "second"},
        "path_length_m": 1.0,
        "search_cost": 2.0,
        "action_set_expansions": 3.0,
    }
    best_results: batch.BestResults = {}

    assert batch.update_best_results(best_results, first)
    assert batch.update_best_results(best_results, tied)
    assert best_results["test_env"]["path_length_m"] == [first, tied]

    batch.print_environment_bests("test_env", best_results, "Best")

    output = capsys.readouterr().out
    assert "Path length: 1.0 m (2 tied runs)" in output
    assert "- variant='first'" in output
    assert "- variant='second'" in output


def test_update_best_results_discards_chained_ties_outside_best_tolerance(monkeypatch) -> None:
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {"variant": ["first", "stale", "best"]})
    results = [
        {
            "environment": "test_env",
            "arguments": {"variant": variant},
            "path_length_m": value,
            "search_cost": value,
            "action_set_expansions": value,
        }
        for variant, value in (("first", 1.0), ("stale", 1.009), ("best", 0.991))
    ]
    best_results: batch.BestResults = {}

    for result in results:
        batch.update_best_results(best_results, result)

    for metric in batch.METRICS:
        metric_results = best_results["test_env"][metric]
        best_value = min(result[metric] for result in metric_results)
        assert all(result[metric] < best_value + 1e-2 for result in metric_results)
        assert metric_results == [results[0], results[2]]


def test_run_all_persists_failed_runs_without_updating_bests(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(batch, "ENVIRONMENTS", ["test_env"])
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {"option": [1]})

    def failing_main(arguments: argparse.Namespace) -> batch.Result:
        raise RuntimeError("No path found")

    monkeypatch.setattr(batch, "main", failing_main)

    results = batch.run_all(
        output_directory=tmp_path,
        name="all_results2",
        workers=1,
        read_only=False,
    )

    assert len(results) == 1
    result = results[0]
    assert result["error"] == "No path found"
    assert result["path_length_m"] is None
    assert result["search_cost"] is None
    assert result["action_set_expansions"] is None
    assert batch.load_results(tmp_path / "all_results2" / "all_results2.json") == results
    output = capsys.readouterr().out
    assert "Planning failed for run 1/1: No path found" in output
    assert "Best: test_env" not in output


def test_run_all_saves_each_result_and_resumes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(batch, "ENVIRONMENTS", ["test_env"])
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {"varied": [1, 2], "fixed": [10]})
    calls: list[argparse.Namespace] = []

    def fake_main(arguments: argparse.Namespace) -> batch.Result:
        calls.append(arguments)
        varied = arguments.varied
        return {
            "environment": arguments.env,
            "arguments": {
                "varied": varied,
                "fixed": arguments.fixed,
                "output_dir": str(arguments.output_dir),
            },
            "path_length_m": 3.0 - varied,
            "search_cost": float(varied),
            "action_set_expansions": 30 - varied,
        }

    monkeypatch.setattr(batch, "main", fake_main)

    first_results = batch.run_all(
        output_directory=tmp_path,
        name="all_results2",
        workers=1,
        read_only=False,
    )

    assert len(first_results) == 2
    assert len(calls) == 2
    assert batch.load_results(tmp_path / "all_results2" / "all_results2.json") == first_results
    assert (tmp_path / "all_results2" / "all_results2.csv").exists()
    first_output = capsys.readouterr().out
    assert "varied=" in first_output
    assert "fixed=" not in first_output
    assert "=== Best results by environment ===" in first_output

    second_results = batch.run_all(
        output_directory=tmp_path,
        name="all_results2",
        workers=1,
        read_only=False,
    )

    assert second_results == first_results
    assert len(calls) == 2
    second_output = capsys.readouterr().out
    assert "Loaded 2 completed runs" in second_output
    assert "=== Best so far from loaded results ===" in second_output
    assert second_output.count("Skipping completed run") == 2


def test_run_all_read_only_writes_csv(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(batch, "ENVIRONMENTS", ["test_env"])
    monkeypatch.setattr(batch, "PARAMETER_VALUES", {"option": [1]})
    batch.save_results(
        tmp_path / "all_results" / "all_results.json",
        [
            {
                "environment": "test_env",
                "arguments": {"option": 1},
                "path_length_m": 1.0,
                "search_cost": 2.0,
                "action_set_expansions": 3,
            }
        ],
    )

    results = batch.run_all(
        output_directory=tmp_path,
        name="all_results",
        workers=1,
        read_only=True,
    )

    assert results[0]["environment"] == "test_env"
    assert (tmp_path / "all_results" / "all_results.csv").exists()
    assert "Aggregate results CSV:" in capsys.readouterr().out


def test_parse_args_uses_named_output_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_hybrid_astar_batch.py"])

    arguments = batch.parse_args()

    assert arguments.output_dir == Path("./hybrid_a_star")
    assert arguments.name == "all_results"
    assert arguments.workers == 10


def test_parse_args_accepts_parallel_workers(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hybrid_astar_batch.py",
            "--output-dir",
            "custom-runs",
            "--name",
            "results",
            "--workers",
            "3",
        ],
    )

    arguments = batch.parse_args()

    assert arguments.output_dir == Path("custom-runs")
    assert arguments.name == "results"
    assert arguments.workers == 3
