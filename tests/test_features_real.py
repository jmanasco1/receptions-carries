"""Stage 2 against the real corpus.

The synthetic fixtures prove the code does what it says. They cannot prove the
panel agrees with nflverse, that the leakage checks survive contact with ten
seasons of schema drift, or that the role tracker predicts anything. These do.

Skipped when the raw store is empty, so a fresh clone still runs a green suite.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.features.build import build_features, feature_columns
from nfl_usage_props.features.dataset import load_panel, load_schedules
from nfl_usage_props.leakage import run_leakage_checks
from nfl_usage_props.model.role_tracker import expit
from nfl_usage_props.storage import ParquetStore

pytestmark = pytest.mark.network

SEASONS = [2023, 2024]


@pytest.fixture(scope="module")
def real_config():
    return load_config(load_env=False)


@pytest.fixture(scope="module")
def store(real_config):
    store = ParquetStore(real_config.raw_dir)
    for table in ("pbp", "participation", "schedules", "player_stats"):
        missing = [s for s in SEASONS if not store.exists(table, s)]
        if missing:
            pytest.skip(f"{table} not ingested for {missing}; run `nfl-props ingest run`")
    return store


@pytest.fixture(scope="module")
def real_panel(real_config, store):
    return load_panel(real_config, SEASONS)


@pytest.fixture(scope="module")
def real_features(real_config, real_panel):
    return build_features(real_panel, load_schedules(real_config, SEASONS), real_config)


# ------------------------------------------------- the panel agrees with nflverse


@pytest.fixture(scope="module")
def against_player_stats(real_panel, store):
    """Our per-player season totals joined to nflverse's own."""
    ours = (
        real_panel.filter(pl.col("season_type") == "REG")
        .group_by("gsis_id", "season")
        .agg(
            pl.col("targets").sum().alias("targets_ours"),
            pl.col("receptions").sum().alias("receptions_ours"),
            pl.col("carries").sum().alias("carries_ours"),
        )
    )
    theirs = (
        pl.concat([store.read("player_stats", s) for s in SEASONS], how="diagonal_relaxed")
        .filter(pl.col("season_type") == "REG")
        .group_by(pl.col("player_id").alias("gsis_id"), "season")
        .agg(
            pl.col("targets").sum().alias("targets_theirs"),
            pl.col("receptions").sum().alias("receptions_theirs"),
            pl.col("carries").sum().alias("carries_theirs"),
            pl.col("position").first(),
        )
    )
    return ours.join(theirs, on=["gsis_id", "season"], how="inner")


@pytest.mark.parametrize("quantity", ["targets", "receptions"])
def test_panel_matches_nflverse_exactly(against_player_stats, quantity):
    """Not "close to" -- exactly. These are counts of the same events, and a
    discrepancy means our play universe drifted from theirs."""
    diff = against_player_stats[f"{quantity}_ours"] - against_player_stats[f"{quantity}_theirs"]
    assert diff.abs().max() == 0


def test_carry_totals_differ_from_nflverse_essentially_only_by_qb_kneels(against_player_stats):
    """A deliberate, documented divergence, plus one upstream oddity.

    nflverse counts kneels as carries; we do not, because a kneel is not an
    opportunity anyone competes for. That accounts for every QB here.

    Non-QBs appear only via a rarer case: a penalty enforced *after* a run that
    stood is tagged `no_play` by nflverse even though the carry counted (a 2024
    Texans fake punt is the example in this corpus). We exclude `no_play`
    because the overwhelming majority of them really were wiped, and telling
    the two apart needs description parsing. The cost is bounded, and this test
    is what bounds it: a single carry, not a systematic shortfall.
    """
    mismatched = against_player_stats.filter(pl.col("carries_ours") != pl.col("carries_theirs"))
    assert mismatched.height > 0, "no mismatches at all means kneels are being counted"

    # Always fewer, never more: we drop plays, we never invent carries.
    assert (mismatched["carries_ours"] < mismatched["carries_theirs"]).all()

    quarterbacks = mismatched.filter(pl.col("position") == "QB")
    assert quarterbacks.height / mismatched.height > 0.9

    others = mismatched.filter(pl.col("position") != "QB")
    gap = others["carries_theirs"] - others["carries_ours"]
    assert gap.max() <= 1 if others.height else True


def test_shares_sum_to_one_for_every_real_team_game(real_panel):
    sums = real_panel.group_by("game_id", "team").agg(
        pl.col("target_share").sum().alias("targets"),
        pl.col("carry_share").sum().alias("carries"),
    )
    for column in ("targets", "carries"):
        assert sums[column].min() == pytest.approx(1.0)
        assert sums[column].max() == pytest.approx(1.0)


def test_snaps_broadly_agree_with_the_independent_snap_count_feed(real_panel, store):
    """`participation` and PFR's `snap_counts` are separate feeds counting the
    same thing. They will not match exactly -- different play universes -- but a
    large divergence means one of them is being read wrong."""
    snap_counts = pl.concat([store.read("snap_counts", s) for s in SEASONS], how="diagonal_relaxed")
    theirs = snap_counts.group_by("pfr_player_id").agg(
        pl.col("offense_snaps").sum().alias("theirs")
    )
    players = store.read("players", 0).select(
        pl.col("gsis_id"), pl.col("pfr_id").alias("pfr_player_id")
    )
    ours = real_panel.group_by("gsis_id").agg(pl.col("snaps").sum().alias("ours"))
    joined = (
        ours.join(players, on="gsis_id", how="inner")
        .join(theirs, on="pfr_player_id", how="inner")
        .filter(pl.col("theirs") > 200)
    )
    assert joined.height > 300, "join collapsed; the crosswalk is not working"
    ratio = joined["ours"] / joined["theirs"]
    assert ratio.median() == pytest.approx(1.0, abs=0.06)


# -------------------------------------------------------------- leakage, for real


@pytest.mark.parametrize("week", [1, 6, 12, 18])
def test_no_leakage_anywhere_in_a_real_season(real_panel, real_config, week):
    schedules = load_schedules(real_config, SEASONS)
    findings = run_leakage_checks(real_panel, schedules, real_config, season=2024, week=week)
    assert findings == [], "\n".join(str(f) for f in findings)


# ------------------------------------------------------ the features are usable


def test_every_feature_row_gets_context(real_features):
    """The relocated-team join failure showed up as exactly this: a handful of
    rows quietly missing their spread and total."""
    for column in ("spread", "total_line", "rest", "roof_type", "def_pass_rate_allowed"):
        assert real_features[column].null_count() == 0


def test_the_role_prior_predicts_the_realised_share(real_features):
    """The point of the whole exercise. A tracker that does not beat noise on
    held-out games is an elaborate way to produce a constant."""
    settled = real_features.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("games_of_history") >= 4)
        & (pl.col("team_targets") > 0)
    ).with_columns(
        pl.col("role_target_share_mean")
        .map_elements(expit, return_dtype=pl.Float64)
        .alias("predicted")
    )
    correlation = settled.select(pl.corr("predicted", "target_share")).item()
    error = settled.select((pl.col("predicted") - pl.col("target_share")).abs().mean()).item()
    assert correlation > 0.7
    assert error < 0.06


def test_uncertainty_is_wider_for_players_with_less_history(real_features):
    settled = real_features.filter(pl.col("season_type") == "REG")
    early = settled.filter(pl.col("games_of_history") <= 2)["role_target_share_sd"].mean()
    late = settled.filter(pl.col("games_of_history") >= 12)["role_target_share_sd"].mean()
    assert early > late


def test_the_feature_matrix_has_no_constant_columns(real_features):
    """A column with one value carries no information and usually means a join
    silently produced a default."""
    for column in feature_columns(real_features):
        if real_features[column].dtype.is_numeric():
            assert real_features[column].n_unique() > 1, f"{column} is constant"
