"""As-of opponent adjustments.

What a defence has allowed, computed from games it had already played. Two
properties are load-bearing:

**As-of by construction, not by discipline.** Every quantity is a cumulative
sum over games sorted by kickoff, shifted by one. There is no window in which
a game can contribute to its own features, because the shift happens before
the division. Getting this wrong is the classic backtest-beautifully,
lose-money-live bug, so it is structural rather than a rule to remember.

**Shrunk, always.** After one week, "this defence allows a 78% catch rate" is
a statement about one game. Every rate is pulled toward the league mean with a
prior strength measured in the same units as the denominator, so a
small-sample defence is quietly ignored rather than loudly wrong. The league
mean is itself as-of.

Prior-season carryover exists because week 1 has no current-season history at
all, and a defence is not a fresh draw each September. It is discounted --
personnel and coordinators move -- and decays to nothing as real games arrive.
"""

from __future__ import annotations

import polars as pl

# Rates a defence allows, as (numerator, denominator) over the plays it faced.
# Deliberately short: these feed the volume layers, and each extra rate is
# another chance to fit noise. Slot-vs-wide splits and similar refinements are
# Stage 3 candidates, to be added only if they earn it out of sample.
DEFENSIVE_RATES: dict[str, tuple[str, str]] = {
    "def_pass_rate_allowed": ("team_dropbacks", "team_plays"),
    "def_rush_rate_allowed": ("team_rush_attempts", "team_plays"),
    "def_catch_rate_allowed": ("team_completions", "team_targets"),
    "def_plays_per_game_allowed": ("team_plays", "games"),
}


def defensive_rates(
    team_games: pl.DataFrame,
    *,
    prior_strength: float = 250.0,
    carryover_weight: float = 0.35,
) -> pl.DataFrame:
    """Per (game, defence) the rates that defence had allowed beforehand.

    `team_games` is `panel.team_game_totals` output: one row per offence-game,
    carrying the defence it faced in `opponent`.

    `prior_strength` is in denominator units -- plays for the rate stats -- so
    250 is roughly four games of shrinkage, enough that a week-2 defence is
    mostly league average and a week-12 defence is mostly itself.
    """
    faced = team_games.rename({"team": "offense", "opponent": "defense"}).sort(
        "season", "week", "game_id"
    )

    numerators = sorted({num for num, _ in DEFENSIVE_RATES.values()})
    denominators = sorted({den for _, den in DEFENSIVE_RATES.values() if den != "games"})
    quantities = sorted(set(numerators + denominators))

    faced = faced.with_columns(pl.lit(1).cast(pl.Int32).alias("games"))
    quantities_with_games = [*quantities, "games"]

    # Cumulative within (defence, season), shifted so the current game is
    # excluded. Sorting by (week, game_id) rather than kickoff is safe here
    # because a defence plays at most once a week.
    running = faced.sort("season", "week", "game_id").with_columns(
        [
            pl.col(q)
            .fill_null(0)
            .cum_sum()
            .shift(1, fill_value=0)
            .over("defense", "season")
            .alias(f"_cum_{q}")
            for q in quantities_with_games
        ]
    )

    # Prior-season carryover, discounted. Seeds the cumulative sums so week 1
    # is informed rather than blank, and is swamped by real games within a
    # month -- which is the intent, not a limitation.
    season_totals = faced.group_by("defense", "season").agg(
        [pl.col(q).fill_null(0).sum().alias(f"_prev_{q}") for q in quantities_with_games]
    )
    carryover = season_totals.with_columns(
        pl.col("season") + 1,
        *[(pl.col(f"_prev_{q}") * carryover_weight) for q in quantities_with_games],
    )
    running = running.join(carryover, on=["defense", "season"], how="left").with_columns(
        [
            (pl.col(f"_cum_{q}") + pl.col(f"_prev_{q}").fill_null(0.0)).alias(f"_tot_{q}")
            for q in quantities_with_games
        ]
    )

    # The league mean is as-of too. Using the full-season league average would
    # leak the rest of the season into week 3 through the shrinkage target --
    # a small leak, but the leakage test does not grade on size.
    league = _as_of_league_rates(faced, quantities_with_games)
    running = running.join(league, on=["season", "week"], how="left")

    rates = []
    for name, (num, den) in DEFENSIVE_RATES.items():
        league_rate = pl.col(f"_league_{name}")
        strength = prior_strength if den != "games" else prior_strength / 60.0
        rates.append(
            (
                (pl.col(f"_tot_{num}") + league_rate * strength)
                / (pl.col(f"_tot_{den}") + strength)
            ).alias(name)
        )

    return (
        running.with_columns(rates)
        .with_columns(pl.col("_tot_games").alias("def_games_of_history"))
        .select(
            "season",
            "week",
            "game_id",
            pl.col("defense"),
            "def_games_of_history",
            *DEFENSIVE_RATES,
        )
    )


def _as_of_league_rates(faced: pl.DataFrame, quantities: list[str]) -> pl.DataFrame:
    """League-wide rates from every game played before each (season, week)."""
    weekly = (
        faced.group_by("season", "week")
        .agg([pl.col(q).fill_null(0).sum().alias(q) for q in quantities])
        .sort("season", "week")
    )
    cumulative = weekly.with_columns(
        [
            pl.col(q).cum_sum().shift(1, fill_value=0).over("season").alias(f"_c_{q}")
            for q in quantities
        ]
    )
    # Week 1 has no within-season history; fall back to the season-wide league
    # rate of the PRIOR season, which is knowable in advance.
    prior = (
        weekly.group_by("season")
        .agg([pl.col(q).sum().alias(f"_p_{q}") for q in quantities])
        .with_columns(pl.col("season") + 1)
    )
    cumulative = cumulative.join(prior, on="season", how="left").with_columns(
        [(pl.col(f"_c_{q}") + pl.col(f"_p_{q}").fill_null(0)).alias(f"_t_{q}") for q in quantities]
    )

    exprs = []
    for name, (num, den) in DEFENSIVE_RATES.items():
        exprs.append(
            (pl.col(f"_t_{num}") / pl.col(f"_t_{den}").replace(0, None)).alias(f"_league_{name}")
        )
    # A league rate is a constant across defences, and the very first week of
    # the very first season has nothing at all behind it. Forward-fill from the
    # next available week rather than emitting nulls that would poison every
    # shrinkage target downstream.
    return (
        cumulative.with_columns(exprs)
        .select("season", "week", *[f"_league_{n}" for n in DEFENSIVE_RATES])
        .with_columns(
            [pl.col(f"_league_{n}").fill_null(strategy="backward") for n in DEFENSIVE_RATES]
        )
    )
