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
        test_features.select(
            "game_id",
            "team",
            "gsis_id",
            "receptions",
            "carries",
            "targets",
            "games_of_history",
        ),
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


# ------------------------------------------------- when output is trustworthy


def test_weeks_two_and_three_are_calibrated_enough_to_report(stack):
    """The evidence behind `suppress_output_before_week = 2`.

    Suppressing through week 3 threw away a quarter of the season on the
    assumption that early role estimates are worthless. They are not: the
    tracker carries prior-season history across the offseason, most players
    have plenty of it, and MAE in weeks 2-3 matches mid-season. If this ever
    fails, raise the config value -- but raise it with a number attached.
    """
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    line_worthy = result.targets[index].mean(axis=1) >= 2.0
    early = ((joined["week"] >= 2) & (joined["week"] <= 3)).to_numpy() & line_worthy

    report = assess(result.receptions[index][early], joined["receptions"].to_numpy()[early])
    assert report.well_calibrated, report.summary()


def test_week_one_is_the_one_that_is_not(stack):
    """Complement of the above, and the reason week 1 stays suppressed.

    Offseason moves have not been observed even once, so a carried-over role
    level can be flatly wrong for anyone whose situation changed. The failure
    is in the distribution's shape, not its accuracy -- week 1 MAE is fine,
    which is exactly why an accuracy-only check would have missed this.
    """
    result, joined, *_ = stack
    index = joined["_i"].to_numpy()
    line_worthy = result.targets[index].mean(axis=1) >= 2.0
    week_one = (joined["week"] == 1).to_numpy() & line_worthy

    week_one_pit = assess(
        result.receptions[index][week_one], joined["receptions"].to_numpy()[week_one]
    ).pit_uniformity
    settled = (joined["week"] >= 7).to_numpy() & line_worthy
    settled_pit = assess(
        result.receptions[index][settled], joined["receptions"].to_numpy()[settled]
    ).pit_uniformity
    assert week_one_pit > settled_pit * 2


def test_most_week_one_players_are_not_starting_from_nothing(stack):
    """The premise the suppression window rests on. If prior-season history
    stopped carrying across the offseason, week 1 really would be a blank
    slate and a longer window would be justified."""
    _, joined, *_ = stack
    week_one = joined.filter(pl.col("week") == 1)
    assert (week_one["games_of_history"] > 0).mean() > 0.75
    assert week_one["games_of_history"].median() > 10


# --------------------------------------------- projecting a slate not yet played


@pytest.fixture(scope="module")
def forward_vs_actual():
    """Run the FORWARD path on a week that was played, then compare to truth.

    The forward path builds its roster from depth charts rather than from
    play-by-play, because an unplayed game has no plays. That makes it a
    genuinely different input to everything the backtest exercised, and the way
    it fails is silent: too many names on Layer 3's simplex deflates every real
    player's projection and the only symptom is that every flagged market says
    Under.
    """
    from nfl_usage_props.features.upcoming import panel_with_upcoming
    from nfl_usage_props.model.player_share import eligible

    # 2025, not 2024: the timestamped depth chart feed starts in 2025, and the
    # legacy one is undated and cannot be filtered to a kickoff.
    season, week = 2025, 10
    seasons = [*SEASONS, season]
    config = load_config(load_env=False)
    store = ParquetStore(config.raw_dir)
    for table in ("pbp", "participation", "schedules", "depth_charts"):
        if not store.exists(table, season):
            pytest.skip(f"{table} not ingested for {season}")

    panel = load_panel(config, seasons)
    schedules = load_schedules(config, seasons)
    depth = store.read("depth_charts", season)
    if "dt" not in depth.columns:
        pytest.skip(f"{season} depth charts use the undated legacy feed")

    forward = panel_with_upcoming(panel, schedules, depth, season=season, week=week)
    features = build_features(forward, schedules, config).filter(pl.col("season_type") == "REG")
    history = features.filter(pl.col("team_plays").is_not_null())

    volume = TeamVolumeModel(seed=11)
    volume.fit(team_game_frame(history), fit_wind=False)
    targets = PlayerShareModel("targets", seed=11)
    targets.fit(history)
    carries = PlayerShareModel("carries", seed=11)
    carries.fit(history)
    catch = fit_catch_rate_dispersion(history)

    slate = features.filter((pl.col("season") == season) & (pl.col("week") == week))
    slate_teams = team_game_frame(features).filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    )
    projection = project_week(
        slate, slate_teams, volume, targets, carries, catch.concentration, draws=2000, seed=11
    )
    projected = projection.keys.with_columns(
        pl.Series("projected", projection.receptions.mean(axis=1))
    ).select("game_id", "team", "gsis_id", "projected")

    truth = eligible(panel.filter((pl.col("season") == season) & (pl.col("week") == week))).select(
        "game_id", "team", "gsis_id", "receptions"
    )
    return projected.join(truth, on=["game_id", "team", "gsis_id"], how="inner")


def test_forward_projections_are_unbiased_against_what_happened(forward_vs_actual):
    """The check that would have caught the depth-cap bug immediately. Caps
    summing to 16 instead of 13 put this ratio at roughly 0.8 while every other
    test stayed green."""
    priced = forward_vs_actual.filter(pl.col("projected") >= 2.0)
    assert priced.height > 50
    ratio = priced["projected"].mean() / priced["receptions"].mean()
    assert 0.88 < ratio < 1.12, f"forward projections biased: ratio {ratio:.3f}"


def test_forward_projections_still_track_usage(forward_vs_actual):
    """Unbiased on the mean is not enough -- a constant would manage that."""
    correlation = np.corrcoef(forward_vs_actual["projected"], forward_vs_actual["receptions"])[0, 1]
    assert correlation > 0.55


def test_the_forward_roster_is_the_right_size(forward_vs_actual):
    """Layer 3 normalises across this set, so its size scales every
    projection directly."""
    per_team = forward_vs_actual.group_by("game_id", "team").len()["len"]
    assert 8 <= per_team.median() <= 15
