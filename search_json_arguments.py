#!/usr/bin/env python3
"""Print JSON files whose ``arguments`` object matches every requested filter.

Examples:
    python hybrid_a_star/search_json_arguments.py \
        --root hybrid_a_star/parking2_2 heuristic=default no_animation=false
    python hybrid_a_star/search_json_arguments.py reverse_multiplier=1.1

Unquoted values that are valid JSON scalars are parsed as their JSON type, so
``false``, ``42``, and ``0.15`` match booleans, integers, and floats. Other
unquoted values are strings. To match a string that looks like a JSON scalar,
pass it as a JSON string, for example ``mode='"true"'``.
"""

import argparse
import json
import math
from pathlib import Path
from typing import TypeAlias


ArgumentValue: TypeAlias = str | int | float | bool
Filter: TypeAlias = tuple[str, ArgumentValue]


def parse_filter(text: str) -> Filter:
    """Parse one ``name=value`` command-line filter.

    Args:
        text: A filter supplied on the command line.

    Returns:
        The argument name and its typed expected value.

    Raises:
        argparse.ArgumentTypeError: If the filter is malformed or has a
            non-scalar JSON value.
    """
    if "=" not in text:
        raise argparse.ArgumentTypeError("filters must use name=value syntax")

    name, raw_value = text.split("=", maxsplit=1)
    if not name:
        raise argparse.ArgumentTypeError("filter name cannot be empty")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        return name, raw_value

    if isinstance(value, bool | int | float | str):
        if isinstance(value, float) and not math.isfinite(value):
            raise argparse.ArgumentTypeError("filter values must be finite")
        return name, value
    raise argparse.ArgumentTypeError("filter values must be strings, numbers, or booleans")


def values_match(actual: object, expected: ArgumentValue) -> bool:
    """Return whether two JSON argument values have the same type and value.

    Args:
        actual: The value loaded from a JSON file.
        expected: The typed value from a command-line filter.

    Returns:
        Whether the values match exactly, including their JSON scalar type.
    """
    return type(actual) is type(expected) and actual == expected


def matches_filters(arguments: object, filters: list[Filter]) -> bool:
    """Return whether a JSON ``arguments`` object satisfies every filter.

    Args:
        arguments: The candidate value from a JSON result file.
        filters: Required argument names and values.

    Returns:
        Whether every requested argument is present and matches exactly.
    """
    return isinstance(arguments, dict) and all(
        name in arguments and values_match(arguments[name], expected)
        for name, expected in filters
    )


def json_paths(roots: list[Path]) -> list[Path]:
    """Collect distinct JSON paths from files and directories.

    Args:
        roots: JSON files or directories to search recursively.

    Returns:
        Sorted JSON file paths, with duplicate roots removed.

    Raises:
        ValueError: If a requested root does not exist or is neither a JSON
            file nor a directory.
    """
    paths: dict[Path, Path] = {}
    for root in roots:
        if root.is_dir():
            candidates = root.rglob("*.json")
        elif root.is_file() and root.suffix == ".json":
            candidates = (root,)
        else:
            raise ValueError(f"search root must be an existing directory or JSON file: {root}")
        for path in candidates:
            paths[path.resolve()] = path
    return sorted(paths.values())


def matching_paths(roots: list[Path], filters: list[Filter]) -> list[Path]:
    """Find JSON result files whose ``arguments`` match every filter.

    Invalid JSON files and JSON documents without an ``arguments`` object are
    ignored, allowing a broad directory search without special-case cleanup.

    Args:
        roots: JSON files or directories to search recursively.
        filters: Required argument names and values.

    Returns:
        Sorted paths for matching JSON files.
    """
    matches = []
    for path in json_paths(roots):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(result, dict) and matches_filters(result.get("arguments"), filters):
            matches.append(path)
    return matches


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        The selected roots and argument filters.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="directory or JSON file to search recursively (repeatable; default: current directory)",
    )
    parser.add_argument("filters", nargs="+", type=parse_filter, metavar="name=value")
    arguments = parser.parse_args()
    if not arguments.root:
        arguments.root = [Path(".")]
    return arguments


def main() -> int:
    """Run the JSON argument search command.

    Returns:
        Zero after printing each matching path, one for an invalid search root.
    """
    arguments = parse_arguments()
    try:
        matches = matching_paths(arguments.root, arguments.filters)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for path in matches:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
