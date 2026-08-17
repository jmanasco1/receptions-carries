"""The full stack against the real corpus, held out on unseen seasons.

Every other test proves a component does what it claims on data built to make
it true. This one asks the only question that matters: projected onto seasons
nobody fitted, are the distributions honest?
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.features.build import build_features
from nfl_usage_props.features.dataset import load_panel, load_schedules
from nfl_usage_props.model.calibration import assess
from nfl_usage_props.model.player_share import PlayerShareModel, fit_catch_rate_dispersion
from nfl_usage_props.model.projection import project_week
from nfl_usage_props.model.team_volume import TeamVolumeModel, split_by_season, team_game_frame
from nfl_usage_props.storage import ParquetStore

pytestmark = pytest.mark.network

SEASONS = list(range(2018, 2025))
HOLDOUT = [2024]


@pytest.fixture(scope="module")
def stack():
    config = load_config(load_env=False)
    store = ParquetStore(config.raw_dir)
    for table in ("pbp", "participation", "schedules", "players"):
        seasons = SEASONS if table != "players" else []
        if any(not store.exists(table, s) for s in seasons):
            pytest.skip(f"{table} not fully ingested; run `nfl-props ingest run`")

    panel = load_panel(config, SEASONS)
    features = build_features(panel, load_schedules(config, SEASONS), config).filter(
        pl.col("season_type") == "REG"
    )
    team_games = team_game_frame(features).filter(pl.col("season_type") == "REG")

    train_features, test_features = split_by_season(features, HOLDOUT)
    train_teams, test_teams = split_by_season(team_games, HOLDOUT)

    volume = TeamVolumeModel(seed=7)
    volume.fit(train_teams, fit_wind=False)
    targets = PlayerShareModel("targets", seed=7)
    targets.fit(train_features)
    carries = PlayerShareModel("carries", seed=7)
    carries.fit(train_features)
    catch = fit_catch_rate_dispersion(train_features)

    result = project_week(
        test_features,
        test_teams,
        volume,
        targets,
        carries,
        catch.concentration,
        draws=2000,
        seed=7,
    )
    joined = result.keys.with_row_index("_i").join(
        test_features.select("game_id", "team", "gsis_id", "receptions", "carries", "targets"),
        on=["game_id", "team", "gsis_id"],
        how="inner",
    )
    return result, joined, targets, carries, catch


def test_the_share_layers_fit_cleanly(stack):
    _, _, targets, carries, catch = stack
    assert targets.fit_result.converged
    assert carries.fit_result.converged
    assert catch.converged


def test_the_catch_rate_prior_strength_matches_the_configured_guess(stack):
    """config.toml sets `prior_strength_targets = 50` as a prior. The fitted
    dispersion landing nearby is independent support for it, not a coincidence
    worth ignoring."""
    _, _, _, _, catch = stack
    assert 30 < catch.concentration < 120


@pytest.mark.parametrize("market", ["receptions", "carries"])
def test_held_out_projections_are_calibrated(stack, market):
    """The whole project in one assertion. A model that is accurate on average
    and wrong about its own uncertainty cannot price a half-point."""
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    draws = {"receptions": result.receptions, "carries": result.carries}[market]
    report = assess(draws[index], joined[market].to_numpy())
    assert report.pit_uniformity < 0.12, report.summary()
    assert report.well_calibrated, report.summary()


def test_carry_mass_goes_to_running_backs(stack):
    """The bug that a flat per-layer prior caused: every zero-carry receiver
    was floored near a 1.9% carry share by the continuity correction, and
    because the simplex normalises, backs lost exactly that mass. RBs took 83%
    of real carries while the model gave them 68%."""
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    projected = result.carries[index].mean(axis=1)
    frame = joined.with_columns(pl.Series("projected", projected))

    by_position = frame.group_by("position").agg(
        pl.col("projected").sum().alias("model"),
        pl.col("carries").sum().alias("actual"),
    )
    model_total = by_position["model"].sum()
    actual_total = by_position["actual"].sum()
    for row in by_position.to_dicts():
        model_share = row["model"] / model_total
        actual_share = row["actual"] / actual_total
        assert abs(model_share - actual_share) < 0.07, f"{row['position']}: {row}"


def test_projections_track_realised_usage(stack):
    """Calibration without accuracy is a well-behaved constant."""
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    for market, draws in (("receptions", result.receptions), ("carries", result.carries)):
        predicted = draws[index].mean(axis=1)
        actual = joined[market].to_numpy()
        assert np.corrcoef(predicted, actual)[0, 1] > 0.6


def test_receptions_are_calibrated_where_a_line_would_exist(stack):
    """`output.scope = posted_line_only`, so the population that matters is
    players a book would price -- not every body on the field."""
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    line_worthy = result.targets[index].mean(axis=1) >= 2.0
    report = assess(
        result.receptions[index][line_worthy], joined["receptions"].to_numpy()[line_worthy]
    )
    assert report.well_calibrated, report.summary()
    assert report.pit_uniformity < 0.10, report.summary()
