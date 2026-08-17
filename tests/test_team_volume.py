"""Tests for Layers 1 and 2."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_usage_props.model.team_volume import (
    PASS_RATE_FEATURES,
    PLAYS_FEATURES,
    TeamVolumeModel,
    split_by_season,
    team_game_frame,
)


def synthetic_team_games(
    seasons=(2018, 2019, 2020, 2021, 2022, 2023), weeks=18, seed=3
) -> pl.DataFrame:
    """Team-games with a known pace and script relationship built in.

    Six seasons rather than three, and the total-to-pace effect is deliberately
    stronger than reality. Both are for statistical power: with three seasons
    and a realistic effect the relationship is smaller than the Poisson noise
    and the recovery test fails for want of sample, which says nothing about
    whether the model works.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for team in ("AAA", "BBB", "CCC", "DDD"):
                spread = float(rng.normal(0, 6))
                total = float(rng.normal(45, 4))
                pace = 60 + (2.0 if team in ("AAA", "BBB") else -2.0)
                plays = int(rng.poisson(pace + 0.5 * (total - 45)))
                # Favourites run the clock, underdogs throw. `spread` is
                # positive for a favourite -- the same convention the real
                # feature builder emits -- so the relationship is negative.
                # Generating this with the opposite sign would make the test
                # pass while hiding a production sign error.
                rate = 1 / (1 + np.exp(-(0.2 - 0.045 * spread + 0.02 * (total - 45))))
                dropbacks = int(rng.binomial(plays, rng.beta(rate * 40, (1 - rate) * 40)))
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "game_id": f"{season}_{week:02d}_{team}",
                        "team": team,
                        "opponent": "ZZZ",
                        "team_plays": plays,
                        "team_dropbacks": dropbacks,
                        # Layer 3 divides these, not dropbacks: a sack or a
                        # scramble is a dropback that produces neither.
                        "team_targets": max(int(dropbacks * 0.88), 1),
                        "team_carries": max(plays - dropbacks, 1),
                        "trailing_team_plays": float(pace),
                        "def_plays_per_game_allowed": 62.0,
                        "total_line": total,
                        "abs_spread": abs(spread),
                        "spread": spread,
                        "rest": 7.0,
                        "is_home": float(rng.integers(0, 2)),
                        "trailing_team_proe": float(rng.normal(0, 2)),
                        "def_pass_rate_allowed": 0.58,
                        "trailing_pass_rate": 0.58,
                        "wind_observed": float(rng.uniform(0, 15)),
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def team_games() -> pl.DataFrame:
    return synthetic_team_games()


@pytest.fixture(scope="module")
def fitted(team_games):
    model = TeamVolumeModel()
    model.fit(team_games)
    return model


# ------------------------------------------------------------------- shaping


def test_team_game_frame_collapses_the_player_matrix(synthetic_panel, synthetic_schedules, config):
    from nfl_usage_props.features.build import build_features

    features = build_features(synthetic_panel, synthetic_schedules, config)
    collapsed = team_game_frame(features)
    assert collapsed.height == features.select("game_id", "team").unique().height
    assert "abs_spread" in collapsed.columns
    assert "trailing_pass_rate" in collapsed.columns


def test_split_is_by_whole_seasons(team_games):
    """Rows within a season share team history, so a random split would let
    training inform the test set's features and report a score the model
    cannot reproduce live."""
    all_seasons = set(team_games["season"].unique().to_list())
    train, test = split_by_season(team_games, [2023])
    assert set(train["season"].unique().to_list()) == all_seasons - {2023}
    assert set(test["season"].unique().to_list()) == {2023}
    assert train.height + test.height == team_games.height


# ----------------------------------------------------------------------- fit


def test_both_layers_converge(fitted):
    assert fitted.fit_result.plays.converged
    assert fitted.fit_result.pass_rate.converged


def test_the_pass_rate_responds_to_the_spread(fitted):
    """The core Layer 2 claim: game script drives the split. `spread` is
    positive for a favourite, and favourites run the clock, so the coefficient
    must be negative -- as it is on real data."""
    names = fitted.fit_result.pass_rate.feature_names
    coefficient = fitted.fit_result.pass_rate.coefficients[names.index("spread")]
    assert coefficient < 0


def test_plays_respond_to_the_game_total(fitted):
    names = fitted.fit_result.plays.feature_names
    assert fitted.fit_result.plays.coefficients[names.index("total_line")] > 0


def test_the_wind_model_is_fitted_separately_and_not_used(fitted):
    """Wind is measured during the game. It may inform, never predict."""
    assert fitted.fit_result.wind_pass_rate is not None
    assert fitted.fit_result.wind_coefficient is not None
    assert "wind_observed" not in PASS_RATE_FEATURES
    assert "wind_observed" not in PLAYS_FEATURES


def test_wind_fitting_can_be_skipped(team_games):
    model = TeamVolumeModel()
    fit = model.fit(team_games, fit_wind=False)
    assert fit.wind_pass_rate is None
    assert fit.wind_coefficient is None


def test_rows_without_history_are_dropped_not_imputed(team_games):
    holed = team_games.with_columns(
        pl.when(pl.col("week") == 1)
        .then(None)
        .otherwise(pl.col("trailing_team_plays"))
        .alias("trailing_team_plays")
    )
    model = TeamVolumeModel()
    fit = model.fit(holed, fit_wind=False)
    assert fit.plays.n_observations < team_games.height


# ------------------------------------------------------------------ predict


def test_predictions_are_plausible_nfl_numbers(fitted, team_games):
    predicted = fitted.predict(team_games)
    assert 50 < predicted["expected_plays"].min()
    assert predicted["expected_plays"].max() < 80
    assert 0.3 < predicted["expected_pass_rate"].min()
    assert predicted["expected_pass_rate"].max() < 0.8


def test_dropbacks_and_rush_attempts_sum_to_plays(fitted, team_games):
    predicted = fitted.predict(team_games)
    total = predicted["expected_dropbacks"] + predicted["expected_rush_attempts"]
    assert total.to_numpy() == pytest.approx(predicted["expected_plays"].to_numpy())


def test_prediction_uses_the_training_scaling(fitted, team_games):
    """The regression that broke this once: rebuilding an estimator from its
    coefficients alone drops the standardiser and predicts on raw inputs."""
    predicted = fitted.predict(team_games)
    assert predicted["expected_plays"].mean() == pytest.approx(
        team_games["team_plays"].mean(), rel=0.05
    )


# ------------------------------------------------------------------- sample


def test_samples_have_the_right_shape(fitted, team_games):
    draws = fitted.sample(team_games.head(20), draws=500)
    assert draws["plays"].shape == (20, 500)
    assert draws["dropbacks"].shape == (20, 500)


def test_dropbacks_never_exceed_plays(fitted, team_games):
    draws = fitted.sample(team_games.head(50), draws=500)
    assert np.all(draws["dropbacks"] <= draws["plays"])
    assert np.all(draws["rush_attempts"] >= 0)


def test_samples_centre_on_the_predictions(fitted, team_games):
    subset = team_games.head(100)
    draws = fitted.sample(subset, draws=3000)
    predicted = fitted.predict(subset)
    assert draws["plays"].mean() == pytest.approx(predicted["expected_plays"].mean(), rel=0.03)


def test_dropback_variance_exceeds_binomial(fitted, team_games):
    """Plays and the rate are both random, so the dropback count must inherit
    both. If it only inherited binomial noise, every downstream tail would be
    too thin."""
    subset = team_games.head(100)
    draws = fitted.sample(subset, draws=4000)
    predicted = fitted.predict(subset)
    plays = predicted["expected_plays"].to_numpy()
    rate = predicted["expected_pass_rate"].to_numpy()
    assert draws["dropbacks"].var(axis=1).mean() > (plays * rate * (1 - rate)).mean()


def test_sampling_is_reproducible(team_games):
    a = TeamVolumeModel(seed=5)
    a.fit(team_games, fit_wind=False)
    b = TeamVolumeModel(seed=5)
    b.fit(team_games, fit_wind=False)
    assert np.array_equal(
        a.sample(team_games.head(10), draws=100)["plays"],
        b.sample(team_games.head(10), draws=100)["plays"],
    )


def test_using_the_model_before_fitting_is_an_error(team_games):
    with pytest.raises(RuntimeError, match="not fitted"):
        TeamVolumeModel().predict(team_games)
    with pytest.raises(RuntimeError, match="not fitted"):
        TeamVolumeModel().sample(team_games)
