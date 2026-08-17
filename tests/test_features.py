"""Tests for the assembled as-of feature matrix."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.features.build import (
    KEY_COLUMNS,
    OUTCOME_COLUMNS,
    build_features,
    catch_rate_features,
    feature_columns,
    game_context_features,
    role_features,
    team_volume_features,
)
from nfl_usage_props.model.role_tracker import expit
from tests.conftest import SYNTHETIC_ROSTER, make_synthetic_panel


@pytest.fixture
def features(synthetic_panel, synthetic_schedules, config) -> pl.DataFrame:
    return build_features(synthetic_panel, synthetic_schedules, config)


# ------------------------------------------------------------ column policy


def test_outcomes_are_never_offered_as_features(features):
    """The single most consequential line in the module. If an outcome leaks
    into the feature list, the leakage test is the only thing left."""
    offered = set(feature_columns(features))
    assert offered.isdisjoint(OUTCOME_COLUMNS)
    assert offered.isdisjoint(KEY_COLUMNS)


def test_observed_wind_is_not_a_feature(features):
    """It is measured during the game. It exists to fit the weather
    coefficient, and Stage 3 replaces it with a forecast."""
    assert "wind_observed" in features.columns
    assert "wind_observed" not in feature_columns(features)


def test_roof_type_is_a_feature_even_though_wind_is_not(features):
    """A building's roof is knowable on Wednesday; the afternoon's wind is not.
    Collapsing the two is the mistake this separation exists to prevent."""
    assert "roof_type" in feature_columns(features)


def test_new_columns_are_picked_up_automatically(features):
    extended = features.with_columns(pl.lit(1.0).alias("some_new_signal"))
    assert "some_new_signal" in feature_columns(extended)


# ------------------------------------------------------------------- shape


def test_one_feature_row_per_panel_row(features, synthetic_panel):
    assert features.height == synthetic_panel.height
    assert features.select("game_id", "gsis_id").unique().height == features.height


def test_every_feature_row_has_a_role_estimate(features):
    for layer in ("snap_share", "carry_share", "target_share"):
        assert features[f"role_{layer}_mean"].null_count() == 0
        assert features[f"role_{layer}_sd"].null_count() == 0


def test_context_joins_for_every_row(features):
    for column in ("spread", "total_line", "rest", "is_home", "roof_type"):
        assert features[column].null_count() == 0


# -------------------------------------------------------------- role priors


def test_week_one_has_no_history(features):
    week1 = features.filter(pl.col("week") == 1)
    assert week1["games_of_history"].unique().to_list() == [0]
    assert week1["role_target_share_n"].unique().to_list() == [0]


def test_the_tracker_converges_on_a_stable_players_true_share(features):
    """Six weeks of identical usage. By the end the prior should be close to
    what the player actually does, not still anchored on the league prior."""
    truth = {p: ts for p, _, ts, _, _ in SYNTHETIC_ROSTER}
    last = features.filter(pl.col("week") == 6)
    for row in last.to_dicts():
        expected = truth[row["gsis_id"]]
        assert expit(row["role_target_share_mean"]) == pytest.approx(expected, abs=0.06)


def test_uncertainty_falls_as_the_season_goes_on(features):
    early = features.filter(pl.col("week") == 2)["role_target_share_sd"].mean()
    late = features.filter(pl.col("week") == 6)["role_target_share_sd"].mean()
    assert late < early


def test_a_players_role_prior_ignores_his_own_game(synthetic_panel, config):
    """Direct statement of the same-game contract at the role-tracker level."""
    baseline = role_features(synthetic_panel, config)
    tampered = synthetic_panel.with_columns(
        pl.when(pl.col("week") == 4).then(pl.lit(30)).otherwise(pl.col("targets")).alias("targets")
    )
    after = role_features(tampered, config)
    key = ["game_id", "gsis_id"]
    week4 = [g for g in baseline["game_id"].unique().to_list() if "_04_" in g]
    a = baseline.filter(pl.col("game_id").is_in(week4)).sort(key)
    b = after.filter(pl.col("game_id").is_in(week4)).sort(key)
    assert a["role_target_share_mean"].to_list() == pytest.approx(
        b["role_target_share_mean"].to_list()
    )


def test_a_team_change_is_treated_as_a_role_event(synthetic_panel, config):
    """Same numbers, but the player moved teams. The tracker should be less
    certain afterwards than the player who stayed put."""
    moved = synthetic_panel.with_columns(
        pl.when((pl.col("gsis_id") == "00-0000001") & (pl.col("week") >= 4))
        .then(pl.lit("ZZZ"))
        .otherwise(pl.col("team"))
        .alias("team")
    )
    stayed = role_features(synthetic_panel, config).filter(pl.col("gsis_id") == "00-0000001")
    switched = role_features(moved, config).filter(pl.col("gsis_id") == "00-0000001")
    assert (
        switched["role_target_share_sd"].to_list()[3] > stayed["role_target_share_sd"].to_list()[3]
    )


# --------------------------------------------------------------- catch rate


def test_catch_rate_prior_excludes_the_current_game(synthetic_panel, config):
    baseline = catch_rate_features(synthetic_panel, config)
    tampered = synthetic_panel.with_columns(
        pl.when(pl.col("week") == 3)
        .then(pl.lit(0))
        .otherwise(pl.col("receptions"))
        .alias("receptions")
    )
    after = catch_rate_features(tampered, config)
    week3 = [g for g in baseline["game_id"].unique().to_list() if "_03_" in g]
    a = baseline.filter(pl.col("game_id").is_in(week3)).sort("game_id", "gsis_id")
    b = after.filter(pl.col("game_id").is_in(week3)).sort("game_id", "gsis_id")
    assert a["catch_rate_prior"].to_list() == pytest.approx(b["catch_rate_prior"].to_list())


def test_catch_rate_prior_is_a_probability(features):
    assert features["catch_rate_prior"].min() > 0.0
    assert features["catch_rate_prior"].max() < 1.0


def test_catch_rate_prior_starts_at_the_league_mean(features):
    week1 = features.filter(pl.col("week") == 1)
    assert week1["catch_rate_prior"].n_unique() == 1
    assert week1["catch_rate_prior_targets"].unique().to_list() == [0]


# ------------------------------------------------------------- team volume


def test_trailing_team_volume_excludes_the_current_game(synthetic_panel):
    totals = team_game_totals_shim(synthetic_panel)
    volume = team_volume_features(totals)
    first = volume.join(
        totals.filter(pl.col("week") == 1).select("game_id", "team"),
        on=["game_id", "team"],
    )
    assert first["trailing_team_plays"].null_count() == first.height


def team_game_totals_shim(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.select(
        "season",
        "week",
        "season_type",
        "game_id",
        "team",
        "opponent",
        "team_plays",
        "team_dropbacks",
        "team_rush_attempts",
        "team_targets",
        "team_carries",
        "team_completions",
        "team_proe",
    ).unique(subset=["game_id", "team"])


# ----------------------------------------------------------- game context


def test_spread_is_flipped_for_the_away_team(synthetic_schedules):
    context = game_context_features(synthetic_schedules)
    home = context.filter(pl.col("is_home"))["spread"]
    away = context.filter(~pl.col("is_home"))["spread"]
    assert home.unique().to_list() == [-3.0]
    assert away.unique().to_list() == [3.0]


def test_implied_team_total_is_higher_for_the_favourite(synthetic_schedules):
    context = game_context_features(synthetic_schedules)
    favourite = context.filter(pl.col("spread") < 0)["implied_team_total"][0]
    underdog = context.filter(pl.col("spread") > 0)["implied_team_total"][0]
    assert favourite > underdog


def test_a_dome_zeroes_wind_and_open_air_does_not(synthetic_schedules):
    context = game_context_features(synthetic_schedules)
    dome = context.filter(pl.col("roof_type") == "dome")
    outdoor = context.filter(pl.col("roof_type") == "outdoor")
    assert dome["wind_observed"].unique().to_list() == [0.0]
    assert dome["wind_shielded"].unique().to_list() == [True]
    assert outdoor["wind_observed"].max() > 0.0


def test_both_teams_get_a_row_per_game(synthetic_schedules):
    context = game_context_features(synthetic_schedules)
    assert context.height == synthetic_schedules.height * 2


def test_relocated_teams_still_join():
    """nflverse pbp says LV for a 2016 Raiders game while schedules says OAK.
    Left alone this drops 81 team-games and looks like missing history."""
    schedules = pl.DataFrame(
        [
            {
                "season": 2016,
                "week": 1,
                "game_id": "2016_01_OAK_SD",
                "home_team": "SD",
                "away_team": "OAK",
                "kickoff_utc": None,
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": 5,
                "div_game": 1,
                "home_rest": 7,
                "away_rest": 7,
                "spread_line": -1.0,
                "total_line": 45.0,
            }
        ]
    )
    context = game_context_features(schedules)
    assert sorted(context["team"].to_list()) == ["LAC", "LV"]


# ------------------------------------------------------------------ guards


def test_building_without_kickoff_times_is_refused(synthetic_panel, synthetic_schedules, config):
    """Week ordering is not kickoff ordering: a Thursday game follows the
    previous Monday's. Silently sorting by week would misorder the tracker."""
    with pytest.raises(ValueError, match="kickoff_utc"):
        build_features(synthetic_panel.drop("kickoff_utc"), synthetic_schedules, config)


def test_a_second_season_is_carried_not_reset(config):
    """A player does not arrive in September knowing nothing."""
    first = make_synthetic_panel(season=2023)
    second = make_synthetic_panel(season=2024)
    both = pl.concat([first, second], how="vertical")
    roles = role_features(both, config)
    week1_2024 = [g for g in roles["game_id"].unique().to_list() if g.startswith("2024_01")]
    carried = roles.filter(pl.col("game_id").is_in(week1_2024))
    assert carried["role_target_share_n"].min() > 0
