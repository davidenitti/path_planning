import json
from pathlib import Path

import pytest

import search_json_arguments as search


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name=default", ("name", "default")),
        ("threshold=0.15", ("threshold", 0.15)),
        ("count=42", ("count", 42)),
        ("enabled=false", ("enabled", False)),
        ('label="false"', ("label", "false")),
    ],
)
def test_parse_filter_preserves_scalar_types(text: str, expected: search.Filter) -> None:
    assert search.parse_filter(text) == expected


@pytest.mark.parametrize("text", ["missing_separator", "=value", "items=[1, 2]", "value=null"])
def test_parse_filter_rejects_invalid_filters(text: str) -> None:
    with pytest.raises(Exception):
        search.parse_filter(text)


def test_matching_paths_matches_all_filters_with_exact_types(tmp_path: Path) -> None:
    matching = tmp_path / "matching.json"
    matching.write_text(
        json.dumps(
            {
                "arguments": {
                    "heuristic": "default",
                    "safety_margin": 0.0,
                    "max_expansions": 1_000_000,
                    "save_video": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wrong_type.json").write_text(
        json.dumps({"arguments": {"max_expansions": True, "save_video": True}}),
        encoding="utf-8",
    )
    (tmp_path / "not_a_result.json").write_text(json.dumps({"other": "value"}), encoding="utf-8")
    (tmp_path / "invalid.json").write_text("not JSON", encoding="utf-8")

    matches = search.matching_paths(
        [tmp_path],
        [
            ("heuristic", "default"),
            ("safety_margin", 0.0),
            ("max_expansions", 1_000_000),
            ("save_video", True),
        ],
    )

    assert matches == [matching]


def test_main_prints_matching_files(tmp_path: Path, monkeypatch, capsys) -> None:
    matching = tmp_path / "result.json"
    matching.write_text(json.dumps({"arguments": {"enabled": True}}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["search_json_arguments.py", "--root", str(tmp_path), "enabled=true"],
    )

    assert search.main() == 0
    assert capsys.readouterr().out == f"{matching}\n"
