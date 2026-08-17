"""Stage 3 against the real corpus.

The synthetic tests prove the fitters recover parameters they were given. Only
real data can say whether the resulting distributions are calibrated on games
nobody fitted them to, which is the property the Monte Carlo depends on.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.features.build import build_features
from nfl_usage_props.features.dataset import load_panel, load_schedules
from nfl_usage_props.model.calibration import assess
from nfl_usage_props.model.team_volume import (
    PLAYS_FEATURES,
    TeamVolumeModel,
    split_by_season,
    team_game_frame,
)
from nfl_usage_props.storage import ParquetStore

pytestmark = pytest.mark.network

TRAIN_SEASONS = list(range(2016, 2024))
HOLDOUT = [2024]


@pytest.fixture(scope="module")
def team_games():
    config = load_config(load_env=False)
    store = ParquetStore(config.raw_dir)
    seasons = TRAIN_SEASONS + HOLDOUT
    for table in ("pbp", "participation", "schedules"):
        if any(not store.exists(table, s) for s in seasons):
            pytest.skip(f"{table} not fully ingested; run `nfl-props ingest run`")

    panel = load_panel(config, seasons)
    features = build_features(panel, load_schedules(config, seasons), config)
    return team_game_frame(features).filter(pl.col("season_type") == "REG")


@pytest.fixture(scope="module")
def fitted(team_games):
    train, _ = split_by_season(team_games, HOLDOUT)
    model = TeamVolumeModel()
    model.fit(train)
    return model


def test_both_layers_converge_on_real_data(fitted):
    assert fitted.fit_result.plays.converged, fitted.fit_result.plays.message
    assert fitted.fit_result.pass_rate.converged, fitted.fit_result.pass_rate.message


def test_play_counts_are_mildly_overdispersed(fitted):
    """Not the double-digit dispersion the term usually implies. Guards both
    directions: a tiny phi would mean the fit found structure that is not
    there, and a huge one means it collapsed to Poisson -- which is exactly the
    failure the joint optimiser produced before dispersion was profiled out."""
    assert 200 < fitted.fit_result.plays.dispersion < 3000


def test_the_pass_split_is_clearly_overdispersed(fitted):
    """Game script is real: dropbacks within a game are not independent flips."""
    assert 10 < fitted.fit_result.pass_rate.dispersion < 200


def test_favourites_pass_less(fitted):
    """The sign that was wrong in `implied_team_total` and would have been
    wrong here too. Positive spread means favoured; favourites run the clock."""
    names = fitted.fit_result.pass_rate.feature_names
    assert fitted.fit_result.pass_rate.coefficients[names.index("spread")] < 0


def test_wind_suppresses_passing(fitted):
    """Fitted but not used. Direction is the check -- if wind made teams throw
    more, something upstream is inverted."""
    assert fitted.fit_result.wind_coefficient < 0


@pytest.mark.parametrize("quantity", ["plays", "dropbacks"])
def test_held_out_distributions_are_calibrated(fitted, team_games, quantity):
    """The property the Monte Carlo actually depends on. Accuracy here is
    poor by design -- team volume is close to unpredictable -- but the
    intervals have to mean what they say."""
    _, test = split_by_season(team_games, HOLDOUT)
    usable = fitted._usable(test, PLAYS_FEATURES)
    sampled = fitted.sample(test, draws=3000)
    actual = usable["team_plays" if quantity == "plays" else "team_dropbacks"].to_numpy()
    report = assess(sampled[quantity], actual)
    assert report.well_calibrated, report.summary()


def test_the_model_beats_a_constant_but_only_just(fitted, team_games):
    """Documents the honest size of the effect so nobody mistakes these layers
    for where the edge is. If this ever improves a lot, the docstring claiming
    a ~2% gain needs updating -- and if it degrades, something broke."""
    train, test = split_by_season(team_games, HOLDOUT)
    usable = fitted._usable(test, PLAYS_FEATURES)
    predicted = fitted.predict(test).join(
        usable.select("game_id", "team", "team_plays"), on=["game_id", "team"]
    )
    model_error = (predicted["expected_plays"] - predicted["team_plays"]).abs().mean()
    constant_error = (train["team_plays"].mean() - predicted["team_plays"]).abs().mean()
    assert model_error < constant_error
    assert model_error > constant_error * 0.85
