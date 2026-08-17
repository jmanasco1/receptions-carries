"""Tests for as-of opponent adjustments.

The property under test is not "the number looks right" -- it is "the number
could have been computed on Saturday". Everything else is secondary.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.features.opponent import DEFENSIVE_RATES, defensive_rates


def team_games(rows: list[dict]) -> pl.DataFrame:
    """Build a team_game_totals-shaped frame from sparse overrides."""
    defaults = {
        "season": 2024,
        "season_type": "REG",
        "team_plays": 60,
        "team_dropbacks": 35,
        "team_rush_attempts": 25,
        "team_targets": 33,
        "team_carries": 25,
        "team_completions": 22,
        "team_proe": 0.0,
        "team_xpass": 0.58,
        "team_pass_rate": 35 / 60,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


@pytest.fixture
def two_seasons() -> pl.DataFrame:
    """Six weeks, four teams, with DDD's defence facing pass-heavy offences."""
    rows = []
    for season in (2023, 2024):
        for week in range(1, 7):
            for offense, defense in (("AAA", "BBB"), ("CCC", "DDD")):
                heavy = defense == "DDD"
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"{season}_{week:02d}_{offense}_{defense}",
                        "team": offense,
                        "opponent": defense,
                        "team_dropbacks": 48 if heavy else 28,
                        "team_rush_attempts": 12 if heavy else 32,
                    }
                )
    return team_games(rows)


# --------------------------------------------------------------------- as-of


def test_a_defence_never_sees_its_own_game(two_seasons):
    """The load-bearing property. Change one game's result and the rate for
    that same game must not move."""
    baseline = defensive_rates(two_seasons)
    tampered = two_seasons.with_columns(
        pl.when(pl.col("game_id") == "2024_03_CCC_DDD")
        .then(pl.lit(60))
        .otherwise(pl.col("team_dropbacks"))
        .alias("team_dropbacks")
    )
    after = defensive_rates(tampered)

    key = ["game_id", "defense"]
    before_row = baseline.filter(pl.col("game_id") == "2024_03_CCC_DDD").sort(key)
    after_row = after.filter(pl.col("game_id") == "2024_03_CCC_DDD").sort(key)
    assert before_row["def_pass_rate_allowed"][0] == pytest.approx(
        after_row["def_pass_rate_allowed"][0]
    )


def test_a_later_game_does_move_the_rate(two_seasons):
    """The complement: if nothing ever moved the rate, the as-of test above
    would pass trivially on a constant."""
    baseline = defensive_rates(two_seasons)
    tampered = two_seasons.with_columns(
        pl.when(pl.col("game_id") == "2024_03_CCC_DDD")
        .then(pl.lit(60))
        .otherwise(pl.col("team_dropbacks"))
        .alias("team_dropbacks")
    )
    after = defensive_rates(tampered)
    later = "2024_05_CCC_DDD"
    assert baseline.filter(pl.col("game_id") == later)["def_pass_rate_allowed"][0] != pytest.approx(
        after.filter(pl.col("game_id") == later)["def_pass_rate_allowed"][0]
    )


def test_week_one_is_informed_by_the_prior_season(two_seasons):
    """A defence is not a fresh draw each September. Week 1 2024 should already
    know DDD faced pass-heavy offences all through 2023."""
    rates = defensive_rates(two_seasons)
    week1 = rates.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
    ddd = week1.filter(pl.col("defense") == "DDD")["def_pass_rate_allowed"][0]
    bbb = week1.filter(pl.col("defense") == "BBB")["def_pass_rate_allowed"][0]
    assert ddd > bbb


def test_the_first_week_of_the_first_season_still_produces_a_number(two_seasons):
    """No history at all, in either direction. It must shrink to something,
    not emit a null that poisons every downstream join."""
    rates = defensive_rates(two_seasons)
    first = rates.filter((pl.col("season") == 2023) & (pl.col("week") == 1))
    assert first.height > 0
    for name in DEFENSIVE_RATES:
        assert first[name].null_count() == 0


# ----------------------------------------------------------------- shrinkage


def test_early_season_rates_are_pulled_toward_the_league(two_seasons):
    """After one game, 'this defence allows X' is a statement about one game."""
    loose = defensive_rates(two_seasons, prior_strength=1.0, carryover_weight=0.0)
    tight = defensive_rates(two_seasons, prior_strength=5000.0, carryover_weight=0.0)
    week2 = (pl.col("season") == 2023) & (pl.col("week") == 2)

    spread_loose = loose.filter(week2)["def_pass_rate_allowed"]
    spread_tight = tight.filter(week2)["def_pass_rate_allowed"]
    assert spread_tight.max() - spread_tight.min() < spread_loose.max() - spread_loose.min()


def test_more_history_means_less_shrinkage(two_seasons):
    """By week 6 a defence should look more like itself than it did in week 2."""
    rates = defensive_rates(two_seasons, carryover_weight=0.0)
    early = rates.filter((pl.col("season") == 2023) & (pl.col("week") == 2))
    late = rates.filter((pl.col("season") == 2023) & (pl.col("week") == 6))
    early_gap = early["def_pass_rate_allowed"].max() - early["def_pass_rate_allowed"].min()
    late_gap = late["def_pass_rate_allowed"].max() - late["def_pass_rate_allowed"].min()
    assert late_gap > early_gap


def test_carryover_weight_of_zero_forgets_the_prior_season(two_seasons):
    rates = defensive_rates(two_seasons, carryover_weight=0.0)
    week1 = rates.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
    assert week1["def_pass_rate_allowed"].n_unique() == 1


def test_history_counter_excludes_the_current_game(two_seasons):
    rates = defensive_rates(two_seasons, carryover_weight=0.0).sort("season", "week")
    first = rates.filter((pl.col("season") == 2023) & (pl.col("week") == 1))
    assert first["def_games_of_history"].unique().to_list() == [0]


def test_every_rate_stays_inside_its_natural_bounds(two_seasons):
    rates = defensive_rates(two_seasons)
    for name in ("def_pass_rate_allowed", "def_rush_rate_allowed", "def_catch_rate_allowed"):
        assert rates[name].min() > 0.0
        assert rates[name].max() < 1.0
    assert 40 < rates["def_plays_per_game_allowed"].min()
    assert rates["def_plays_per_game_allowed"].max() < 90


def test_one_row_per_game_and_defence(two_seasons):
    rates = defensive_rates(two_seasons)
    assert rates.height == two_seasons.height
    assert rates.select("game_id", "defense").unique().height == rates.height
