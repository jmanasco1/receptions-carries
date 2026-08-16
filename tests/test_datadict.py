"""Tests for the timing classification.

These matter more than they look. Stage 2's leakage test consumes the `timing`
column, so a column misclassified as `before` here becomes a silent leak that
backtests beautifully and loses money live.
"""

from __future__ import annotations

import pytest

from nfl_usage_props.datadict import (
    TIMING_NOTES,
    ColumnDoc,
    classify,
    render,
    summarise_seasons,
)
from nfl_usage_props.ingest.nflverse import TABLES


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [
        # schedules: setup is pre-game, outcome is not
        ("schedules", "spread_line", "before"),
        ("schedules", "total_line", "before"),
        ("schedules", "away_rest", "before"),
        ("schedules", "kickoff_utc", "before"),
        ("schedules", "result", "after"),
        ("schedules", "home_score", "after"),
        ("schedules", "overtime", "after"),
        # weather in nflverse is observed, not forecast
        ("schedules", "temp", "after"),
        ("schedules", "wind", "after"),
        # roof state is a game-time decision at retractable stadiums
        ("schedules", "roof", "after"),
        ("schedules", "stadium_id", "before"),
        # post-game tables
        ("pbp", "receiver_player_id", "after"),
        ("pbp", "pass_oe", "after"),
        ("snap_counts", "offense_pct", "after"),
        ("participation", "offense_players", "after"),
        ("player_stats", "targets", "after"),
        # depth charts: legacy feed is undated, 2025+ feed has a `dt` timestamp
        ("depth_charts", "depth_team", "lagged"),
        ("depth_charts", "club_code", "lagged"),
        ("depth_charts", "week", "lagged"),
        ("depth_charts", "dt", "before"),
        ("depth_charts", "pos_rank", "before"),
        ("depth_charts", "pos_slot", "before"),
        ("rosters_weekly", "status", "lagged"),
        # injuries carry a real timestamp
        ("injuries", "report_status", "before"),
        ("injuries", "date_modified", "before"),
        # players crosswalk
        ("players", "gsis_id", "static"),
        ("players", "birth_date", "static"),
        ("players", "latest_team", "before"),
    ],
)
def test_classify(table, column, expected):
    post_game = TABLES[table].post_game
    assert classify(table, column, post_game_default=post_game) == expected


def test_every_classification_is_a_known_class():
    for name, spec in TABLES.items():
        for column in ("some_col", "gsis_id", "week"):
            assert classify(name, column, post_game_default=spec.post_game) in TIMING_NOTES


def test_observed_roof_is_not_a_pre_kickoff_feature():
    """Five stadiums decide the roof at game time. Classifying `roof` as
    pre-kickoff would leak that decision into forward wind projections."""
    assert classify("schedules", "roof", post_game_default=False) == "after"


def test_score_columns_are_never_classified_before():
    """The single most dangerous leak: outcome fields treated as inputs."""
    for column in ("home_score", "away_score", "result", "total"):
        assert classify("schedules", column, post_game_default=False) == "after"


def test_legacy_and_modern_depth_chart_columns_differ_in_timing():
    """The 2025 feed's `dt` is what makes it exactly as-of filterable. If these
    two ever classify the same, one of them is wrong."""
    legacy = classify("depth_charts", "depth_team", post_game_default=False)
    modern = classify("depth_charts", "dt", post_game_default=False)
    assert legacy == "lagged"
    assert modern == "before"


@pytest.mark.parametrize(
    ("seasons", "all_seasons", "expected"),
    [
        ([2016, 2017, 2018], [2016, 2017, 2018], "all"),
        ([2016, 2017], [2016, 2017, 2018], "2016–2017"),
        ([2025], [2024, 2025], "2025"),
        ([2016, 2018], [2016, 2017, 2018], "2016, 2018"),
        ([], [2016], "—"),
    ],
)
def test_summarise_seasons(seasons, all_seasons, expected):
    assert summarise_seasons(seasons, all_seasons) == expected


def test_render_produces_markdown_with_timing_and_seasons(config):
    docs = [
        ColumnDoc("schedules", "spread_line", "Float64", "before", "closing spread", "all"),
        ColumnDoc("pbp", "pass_oe", "Float64", "after", "", "2016–2025"),
    ]
    (config.raw_dir / "schedules").mkdir(parents=True, exist_ok=True)
    out = render(config, docs)
    assert "# Data dictionary" in out
    assert "`spread_line`" in out
    assert "| column | dtype | timing | seasons | notes |" in out
    assert "closing spread" in out
    assert "2016–2025" in out


def test_build_unions_columns_across_seasons(config):
    """A column present in only some seasons must still be documented, with its
    partial coverage visible — that is exactly the depth_charts 2025 case."""
    import polars as pl

    from nfl_usage_props.datadict import build
    from nfl_usage_props.storage import ParquetStore

    store = ParquetStore(config.raw_dir)
    store.write(pl.DataFrame({"game_id": ["a"], "old_col": [1]}), "pbp", 2023)
    store.write(pl.DataFrame({"game_id": ["b"], "new_col": [2.0]}), "pbp", 2024)

    docs = {d.column: d for d in build(config) if d.table == "pbp"}
    assert set(docs) == {"game_id", "old_col", "new_col"}
    assert docs["game_id"].seasons == "all"
    assert docs["old_col"].seasons == "2023"
    assert docs["new_col"].seasons == "2024"
