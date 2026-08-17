"""Tests for the leakage detector.

Half of these test the feature matrix. The other half test the *detector*,
which matters just as much: a leakage check that cannot detect a leak passes
forever and provides nothing but false confidence. So every leak class the
detector claims to catch is planted deliberately and must be caught.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props import leakage
from nfl_usage_props.features.build import build_features
from nfl_usage_props.leakage import (
    check_future_leakage,
    check_same_game_leakage,
    run_leakage_checks,
)


@pytest.fixture
def clean(synthetic_panel, synthetic_schedules, config):
    return synthetic_panel, synthetic_schedules, config


@pytest.fixture
def planted(monkeypatch):
    """Install a deliberately leaky feature builder."""

    def install(make_leaky):
        def leaky(panel, schedules, cfg):
            return make_leaky(build_features(panel, schedules, cfg))

        monkeypatch.setattr(leakage, "build_features", leaky)

    return install


# --------------------------------------------------- the real matrix is clean


@pytest.mark.parametrize("week", [2, 4, 6])
def test_the_feature_matrix_is_clean(clean, week):
    panel, schedules, config = clean
    assert run_leakage_checks(panel, schedules, config, season=2024, week=week) == []


def test_an_unknown_week_is_an_error_not_a_silent_pass(clean):
    """A typo in the week number must not look like a clean bill of health."""
    panel, schedules, config = clean
    with pytest.raises(ValueError, match="no panel rows"):
        run_leakage_checks(panel, schedules, config, season=2024, week=99)


# ------------------------------------------------------ the detector detects


def test_a_feature_reading_its_own_outcome_is_caught(clean, planted):
    panel, schedules, config = clean
    planted(lambda f: f.with_columns(pl.col("targets").alias("leaky")))
    findings = check_same_game_leakage(panel, schedules, config, season=2024, week=4)
    assert [f.column for f in findings] == ["leaky"]
    assert findings[0].kind == "same-game"


def test_a_rolling_mean_without_a_shift_is_caught(clean, planted):
    """The classic off-by-one: `rolling_mean` includes the current row. It
    leaks same-game while looking exactly like legitimate trailing history."""
    panel, schedules, config = clean
    planted(
        lambda f: f.sort("gsis_id", "kickoff_utc").with_columns(
            pl.col("targets").rolling_mean(3, min_samples=1).over("gsis_id").alias("leaky")
        )
    )
    findings = check_same_game_leakage(panel, schedules, config, season=2024, week=4)
    assert [f.column for f in findings] == ["leaky"]


def test_a_season_wide_aggregate_is_caught_reading_forward(clean, planted):
    """A full-season mean does not read the current game's row alone -- it
    reads every later week too. Only the truncation check sees this."""
    panel, schedules, config = clean
    planted(
        lambda f: f.with_columns(pl.col("targets").mean().over("gsis_id", "season").alias("leaky"))
    )
    findings = check_future_leakage(panel, schedules, config, season=2024, week=3)
    assert [f.column for f in findings] == ["leaky"]
    assert findings[0].kind == "future"


def test_a_future_only_leak_survives_the_same_game_check(clean, planted):
    """Motivates running both checks: a feature built from strictly LATER games
    is invisible to same-game corruption of the current week."""
    panel, schedules, config = clean
    planted(
        lambda f: f.with_columns(
            pl.col("targets").shift(-1).over("gsis_id").fill_null(0).alias("leaky_next_week")
        )
    )
    same_game = check_same_game_leakage(panel, schedules, config, season=2024, week=3)
    future = check_future_leakage(panel, schedules, config, season=2024, week=3)
    assert same_game == []
    assert [f.column for f in future] == ["leaky_next_week"]


def test_a_leak_in_a_string_column_is_caught(clean, planted):
    """The comparison must not quietly skip non-numeric columns."""
    panel, schedules, config = clean
    planted(
        lambda f: f.with_columns(
            pl.when(pl.col("targets") > 5)
            .then(pl.lit("busy"))
            .otherwise(pl.lit("quiet"))
            .alias("leaky_label")
        )
    )
    findings = check_same_game_leakage(panel, schedules, config, season=2024, week=4)
    assert [f.column for f in findings] == ["leaky_label"]


def test_a_leak_that_only_nulls_out_is_caught(clean, planted):
    """Subtraction hides a null-vs-value difference; the detector must not."""
    panel, schedules, config = clean
    planted(
        lambda f: f.with_columns(
            pl.when(pl.col("targets") > 5)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.lit(1.0))
            .alias("leaky_null")
        )
    )
    findings = check_same_game_leakage(panel, schedules, config, season=2024, week=4)
    assert [f.column for f in findings] == ["leaky_null"]


def test_outcome_columns_are_not_themselves_reported_as_leaks(clean, planted):
    """The corruption changes the outcomes by design. Reporting them would
    bury every real finding in noise."""
    panel, schedules, config = clean
    planted(lambda f: f)
    assert check_same_game_leakage(panel, schedules, config, season=2024, week=4) == []


# ----------------------------------------------------------------- reporting


def test_a_finding_says_what_moved_and_by_how_much(clean, planted):
    panel, schedules, config = clean
    planted(lambda f: f.with_columns(pl.col("targets").alias("leaky")))
    finding = check_same_game_leakage(panel, schedules, config, season=2024, week=4)[0]
    assert finding.rows_changed > 0
    assert finding.rows_changed <= finding.total_rows
    assert "->" in finding.example
    assert "leaky" in str(finding)


def test_a_misaligned_comparison_fails_loudly(clean):
    """If the two builds are not row-aligned the comparison is meaningless. It
    must raise rather than report a tidy, wrong answer."""
    panel, schedules, config = clean
    built = build_features(panel, schedules, config)
    with pytest.raises(AssertionError, match="row count"):
        leakage._compare(built, built.head(5), kind="same-game")
