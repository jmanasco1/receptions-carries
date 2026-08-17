"""Tests for the Monte Carlo composition (Stage 5)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_usage_props.model.player_share import PlayerShareModel
from nfl_usage_props.model.projection import MAX_COUNT, Projector, project_week
from nfl_usage_props.model.team_volume import TeamVolumeModel
from tests.test_player_share import make_features
from tests.test_team_volume import synthetic_team_games


@pytest.fixture(scope="module")
def pieces():
    """A fitted stack over data whose team ids line up on both sides."""
    features = make_features(n_team_games=300, seed=21)
    # Enough team-games to cover every game in the share fixture -- 5 seasons
    # of 18 weeks across 4 teams is 360, comfortably above the 300 needed.
    team_games = synthetic_team_games(seasons=(2019, 2020, 2021, 2022, 2023), weeks=18, seed=6)
    # The share fixture uses one team "AAA"; align the volume frame's keys onto
    # the same game ids so the join in `project` has something to match.
    games = features.select("game_id").unique().sort("game_id")["game_id"].to_list()
    team_games = team_games.head(len(games)).with_columns(
        pl.Series("game_id", games), pl.lit("AAA").alias("team")
    )

    volume = TeamVolumeModel(seed=1)
    volume.fit(team_games, fit_wind=False)
    targets = PlayerShareModel("targets", seed=1)
    targets.fit(features)
    carries = PlayerShareModel("carries", seed=1)
    carries.fit(features)
    return features, team_games, volume, targets, carries


@pytest.fixture(scope="module")
def result(pieces):
    features, team_games, volume, targets, carries = pieces
    return project_week(features, team_games, volume, targets, carries, 60.0, draws=800, seed=3)


# -------------------------------------------------------------------- shape


def test_every_player_gets_draws(result):
    assert result.receptions.shape == (result.keys.height, 800)
    assert result.targets.shape == result.receptions.shape
    assert result.carries.shape == result.receptions.shape


def test_receptions_never_exceed_targets(result):
    """A reception requires a target. If this ever fails the composition is
    wired wrong, not merely miscalibrated."""
    assert np.all(result.receptions <= result.targets)


def test_counts_are_non_negative(result):
    assert result.targets.min() >= 0
    assert result.carries.min() >= 0


# ------------------------------------------------------- correlation is kept


def test_teammates_share_one_simulation_of_their_team(pieces):
    """All players on a team must be dividing the SAME sampled game. If they
    were drawn independently the team would project more targets than it threw."""
    features, team_games, volume, targets, carries = pieces
    result = project_week(features, team_games, volume, targets, carries, 60.0, draws=600, seed=3)
    per_game = result.keys.with_row_index("_i").group_by("game_id").agg(pl.col("_i"))
    for row in per_game.head(25).to_dicts():
        rows = np.array(row["_i"])
        team_total = result.targets[rows].sum(axis=0)
        # A whole team's targets in one simulation stay inside a plausible NFL
        # range; independent draws would blow through the top of it.
        assert team_total.max() < 80


def test_the_same_seed_reproduces_the_projection(pieces):
    features, team_games, volume, targets, carries = pieces
    a = project_week(features, team_games, volume, targets, carries, 60.0, draws=200, seed=9)
    b = project_week(features, team_games, volume, targets, carries, 60.0, draws=200, seed=9)
    assert np.array_equal(a.receptions, b.receptions)


# ---------------------------------------------------------------------- PMF


def test_pmf_is_monotone_decreasing(result):
    """P(X >= k) can only fall as k rises. A violation means the table is
    being read off the wrong axis."""
    table = result.pmf_table("receptions")
    columns = [f"p_over_{k - 0.5:g}" for k in range(1, MAX_COUNT + 1)]
    values = np.column_stack([table[c].to_numpy() for c in columns])
    assert np.all(np.diff(values, axis=1) <= 1e-12)


def test_pmf_probabilities_are_probabilities(result):
    table = result.pmf_table("receptions")
    for column in table.columns:
        if column.startswith("p_over_"):
            assert table[column].min() >= 0.0
            assert table[column].max() <= 1.0


def test_pmf_keeps_the_identifying_columns(result):
    table = result.pmf_table("carries")
    for column in ("game_id", "team", "gsis_id", "mean", "p10", "p90"):
        assert column in table.columns


def test_probability_over_matches_the_pmf_table(result):
    table = result.pmf_table("receptions")
    assert result.probability_over("receptions", 2.5) == pytest.approx(
        table["p_over_2.5"].to_numpy()
    )


def test_a_whole_number_line_is_refused(result):
    """Integer lines push rather than resolving. Treating a push as a loss
    would misprice every one of them, and books do post them."""
    with pytest.raises(ValueError, match="push"):
        result.probability_over("receptions", 4.0)


def test_probability_over_falls_as_the_line_rises(result):
    low = result.probability_over("receptions", 1.5).mean()
    high = result.probability_over("receptions", 5.5).mean()
    assert low > high


def test_an_unknown_market_is_rejected(result):
    with pytest.raises(KeyError):
        result.probability_over("passing_yards", 250.5)


# ------------------------------------------------------------------- layers


def test_catch_rate_draws_widen_receptions(pieces):
    """Layer 4 samples the catch rate rather than fixing it at the prior mean.
    Fixing it would make every reception distribution too narrow."""
    features, team_games, volume, targets, carries = pieces
    varied = Projector(volume, targets, carries, 8.0, seed=4).project(
        features, team_games, draws=600
    )
    nearly_fixed = Projector(volume, targets, carries, 5000.0, seed=4).project(
        features, team_games, draws=600
    )
    assert varied.receptions.std(axis=1).mean() > nearly_fixed.receptions.std(axis=1).mean()
