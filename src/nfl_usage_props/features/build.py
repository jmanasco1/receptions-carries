"""Assemble the as-of feature matrix.

Every column this module emits is knowable before kickoff. That is not a
convention to be careful about -- `leakage.py` re-derives the whole matrix from
a truncated corpus and fails the build if any column moves.

Three sources of pre-kickoff information, and they are different in kind:

* **History**: what the player and his team did in *earlier* games. Fed through
  the role tracker, which returns a distribution rather than a point.
* **The matchup**: what the opponent has allowed, as-of.
* **The setup**: spread, total, rest, wind. Genuinely pre-kickoff, no history
  needed -- and the place where it is easiest to leak by accident, because
  nflverse stores the observed weather and the observed roof state in the same
  row as the closing line.

Outcome columns ride along under `OUTCOME_COLUMNS` so training has its targets,
and are excluded from the feature set by name rather than by hoping nobody
selects them.
"""

from __future__ import annotations

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.features.opponent import DEFENSIVE_RATES, defensive_rates
from nfl_usage_props.model.role_tracker import RoleObservation, filter_series
from nfl_usage_props.reference import is_wind_shielded, load_stadiums, normalize_team

# The layers the role tracker runs, as (feature prefix, numerator, denominator).
ROLE_LAYERS: dict[str, tuple[str, str]] = {
    "snap_share": ("snaps", "team_plays"),
    "carry_share": ("carries", "team_carries"),
    "target_share": ("targets", "team_targets"),
}

# What the model predicts. Never features.
OUTCOME_COLUMNS = (
    "receptions",
    "carries",
    "targets",
    "snaps",
    "pass_snaps",
    "target_share",
    "carry_share",
    "snap_share",
    "route_share",
    "catch_rate",
    "team_plays",
    "team_dropbacks",
    "team_rush_attempts",
    "team_targets",
    "team_carries",
    "team_completions",
    "team_proe",
    "team_xpass",
    "team_pass_rate",
)

# Known only after kickoff, but kept because the model has to be FITTED on
# something. `schedules.wind` is measured during the game, so it can teach the
# weather coefficient on historical data and can never be read on a Sunday
# morning. Stage 3 pulls a forecast before kickoff and emits `wind_forecast`,
# which is a real feature; until then this column exists to fit against and is
# excluded from the feature set by name.
#
# The stadium's roof TYPE is a different thing entirely and is a legitimate
# feature -- it is a property of the building, not of the afternoon.
FITTING_ONLY_COLUMNS = ("wind_observed",)

# Identifiers: neither features nor outcomes.
KEY_COLUMNS = (
    "season",
    "week",
    "season_type",
    "game_id",
    "gsis_id",
    "team",
    "opponent",
    "position",
    "kickoff_utc",
)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """The columns a model may train on at prediction time.

    Everything that is not a key, an outcome, or fitting-only. Selecting
    features by exclusion rather than by an allow-list is deliberate: a new
    column added to the builder shows up in training automatically, whereas an
    allow-list would let it be silently forgotten. The leakage test is what
    makes that safe.
    """
    excluded = set(KEY_COLUMNS) | set(OUTCOME_COLUMNS) | set(FITTING_ONLY_COLUMNS)
    return [c for c in frame.columns if c not in excluded]


# --------------------------------------------------------------------- roles


def role_features(panel: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Run the role tracker over every player, returning pre-kickoff priors.

    One pass per player covering all three layers, in kickoff order. The
    tracker returns the prior for each game -- the state before that game's
    observation is folded in -- so the output is same-game safe by
    construction.
    """
    tracker = config.model.role_tracker
    early = config.model.early_season

    needed = [
        "gsis_id",
        "game_id",
        "kickoff_utc",
        "season",
        "week",
        "team",
        *{c for pair in ROLE_LAYERS.values() for c in pair},
    ]
    ordered = panel.select(sorted(set(needed))).sort("gsis_id", "kickoff_utc", "game_id")

    columns = {name: ordered[name].to_list() for name in ordered.columns}
    n = ordered.height

    # Group boundaries over the sorted frame -- cheaper than partitioning into
    # thousands of small DataFrames.
    player_ids = columns["gsis_id"]
    starts: list[int] = [0] if n else []
    for i in range(1, n):
        if player_ids[i] != player_ids[i - 1]:
            starts.append(i)
    bounds = list(zip(starts, starts[1:] + [n], strict=True))

    out: dict[str, list] = {
        f"role_{layer}_{stat}": []
        for layer in ROLE_LAYERS
        for stat in ("mean", "sd", "n", "changepoint")
    }
    out["games_of_history"] = []
    out["_row"] = []

    for start, stop in bounds:
        idx = range(start, stop)
        seasons = [columns["season"][i] for i in idx]
        teams = [columns["team"][i] for i in idx]
        weeks = [columns["week"][i] for i in idx]

        # A player's role changes for reasons we can see coming. Two of them are
        # visible in the panel itself and need no extra feed: he changed teams,
        # or he is back after missing time. Both get the event multiplier.
        events = [False] * len(idx)
        breaks = [False] * len(idx)
        for j in range(1, len(idx)):
            breaks[j] = seasons[j] != seasons[j - 1]
            missed = (not breaks[j]) and (weeks[j] - weeks[j - 1] > 1)
            events[j] = teams[j] != teams[j - 1] or missed

        exposure = _exposure(
            [columns["snaps"][i] for i in idx],
            [columns["team_plays"][i] for i in idx],
            threshold=tracker.partial_game_exposure_ratio,
        )

        for layer, (num, den) in ROLE_LAYERS.items():
            observations = [
                RoleObservation(
                    successes=columns[num][i] or 0,
                    trials=columns[den][i] or 0,
                    # Snap share IS the exposure measure, so downweighting it by
                    # itself would double-count a partial game.
                    exposure=1.0 if layer == "snap_share" else exposure[j],
                    event=events[j],
                    season_break=breaks[j],
                )
                for j, i in enumerate(idx)
            ]
            estimates = filter_series(
                observations,
                process_noise=tracker.process_noise(layer),
                prior_mean_logit=_layer_prior_logit(layer),
                prior_var_logit=_PRIOR_VAR_LOGIT,
                event_multiplier=tracker.process_noise_event_multiplier,
                season_break_weight=early.prior_season_weight_week1,
                changepoint_lookback=tracker.changepoint_lookback_games,
                changepoint_threshold_sd=tracker.changepoint_threshold_sd,
            )
            out[f"role_{layer}_mean"].extend(e.mean_logit for e in estimates)
            out[f"role_{layer}_sd"].extend(e.sd_logit for e in estimates)
            out[f"role_{layer}_n"].extend(e.n_observations for e in estimates)
            out[f"role_{layer}_changepoint"].extend(e.changepoint for e in estimates)

        out["games_of_history"].extend(range(len(idx)))
        out["_row"].extend(idx)

    frame = pl.DataFrame(out).sort("_row").drop("_row")
    return pl.concat([ordered.select("game_id", "gsis_id"), frame], how="horizontal")


_PRIOR_VAR_LOGIT = 4.0
# Diffuse but not absurd starting points, on the logit scale. A player with no
# history is not assumed to be a starter or a ghost; the first real game moves
# him a long way, which is correct.
_LAYER_PRIOR_SHARE = {"snap_share": 0.35, "carry_share": 0.12, "target_share": 0.09}


def _layer_prior_logit(layer: str) -> float:
    from nfl_usage_props.model.role_tracker import logit

    return logit(_LAYER_PRIOR_SHARE[layer])


def _exposure(snaps: list, team_plays: list, *, threshold: float) -> list[float]:
    """How much of each game the player was actually available for.

    Measured against the player's own trailing median snap share, not a league
    constant: a rotational back playing his usual 40% is at full exposure, while
    a starter at 40% left early. Only the second should have his target share
    downweighted, and only a player-relative baseline can tell them apart.
    """
    exposures: list[float] = []
    history: list[float] = []
    for snap, plays in zip(snaps, team_plays, strict=True):
        share = (snap or 0) / plays if plays else 0.0
        if not history:
            exposures.append(1.0)
        else:
            baseline = sorted(history)[len(history) // 2]
            ratio = share / baseline if baseline > 0 else 1.0
            exposures.append(min(1.0, ratio) if ratio < threshold else 1.0)
        if share > 0:
            history.append(share)
    return exposures


# ---------------------------------------------------------------- catch rate


def catch_rate_features(panel: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Beta-Binomial catch rate from earlier games, shrunk to the league mean.

    Not run through the logit tracker: catch rate is close to a fixed player
    trait plus target-quality noise, so it wants shrinkage toward a mean rather
    than a random walk that would chase week-to-week variance.
    """
    strength = config.model.catch_rate.prior_strength_targets
    league = _as_of_league_catch_rate(panel)

    return (
        panel.sort("gsis_id", "kickoff_utc", "game_id")
        .with_columns(
            pl.col("receptions")
            .fill_null(0)
            .cum_sum()
            .shift(1, fill_value=0)
            .over("gsis_id")
            .alias("_r"),
            pl.col("targets")
            .fill_null(0)
            .cum_sum()
            .shift(1, fill_value=0)
            .over("gsis_id")
            .alias("_t"),
        )
        .join(league, on=["season", "week"], how="left")
        .with_columns(
            (
                (pl.col("_r") + pl.col("_league_catch_rate") * strength) / (pl.col("_t") + strength)
            ).alias("catch_rate_prior"),
            pl.col("_t").alias("catch_rate_prior_targets"),
        )
        .select("game_id", "gsis_id", "catch_rate_prior", "catch_rate_prior_targets")
    )


def _as_of_league_catch_rate(panel: pl.DataFrame) -> pl.DataFrame:
    weekly = (
        panel.group_by("season", "week")
        .agg(
            pl.col("receptions").fill_null(0).sum().alias("r"),
            pl.col("targets").fill_null(0).sum().alias("t"),
        )
        .sort("season", "week")
    )
    prior = (
        weekly.group_by("season")
        .agg(pl.col("r").sum().alias("pr"), pl.col("t").sum().alias("pt"))
        .with_columns(pl.col("season") + 1)
    )
    return (
        weekly.with_columns(
            pl.col("r").cum_sum().shift(1, fill_value=0).over("season").alias("cr"),
            pl.col("t").cum_sum().shift(1, fill_value=0).over("season").alias("ct"),
        )
        .join(prior, on="season", how="left")
        .with_columns(
            (
                (pl.col("cr") + pl.col("pr").fill_null(0))
                / (pl.col("ct") + pl.col("pt").fill_null(0)).replace(0, None)
            ).alias("_league_catch_rate")
        )
        .with_columns(pl.col("_league_catch_rate").fill_null(strategy="backward"))
        .select("season", "week", "_league_catch_rate")
    )


# -------------------------------------------------------------- team volume


def team_volume_features(team_games: pl.DataFrame, *, window: int = 5) -> pl.DataFrame:
    """Trailing team volume and tendency, excluding the current game."""
    quantities = ("team_plays", "team_dropbacks", "team_rush_attempts", "team_proe")
    return (
        team_games.sort("team", "season", "week", "game_id")
        .with_columns(
            [
                pl.col(q)
                .shift(1)
                .rolling_mean(window, min_samples=1)
                .over("team")
                .alias(f"trailing_{q}")
                for q in quantities
            ]
        )
        .select("game_id", "team", *[f"trailing_{q}" for q in quantities])
    )


# ------------------------------------------------------------- game context


def game_context_features(schedules: pl.DataFrame) -> pl.DataFrame:
    """Pre-kickoff setup, one row per (game, team).

    The wind handling is the delicate part. `schedules.wind` is *observed*, so
    it is post-kickoff and cannot be a feature as-is -- but it is the only wind
    series available for fitting, and Stage 3 will replace it with a forecast
    pulled before kickoff. It is emitted here under a name that says so, and
    zeroed wherever the stadium shields it, which is a fixed characteristic
    from the reference table rather than the observed roof state.
    """
    stadiums = load_stadiums().select("stadium_id", "roof_type")
    schedules = schedules.with_columns(
        normalize_team("home_team").alias("home_team"),
        normalize_team("away_team").alias("away_team"),
    )

    long = pl.concat(
        [
            schedules.select(
                "game_id",
                "season",
                "week",
                "kickoff_utc",
                "stadium_id",
                "roof",
                "wind",
                "div_game",
                pl.col("home_team").alias("team"),
                pl.col("away_team").alias("opponent"),
                pl.col("home_rest").alias("rest"),
                pl.col("spread_line").alias("_spread_home"),
                "total_line",
                pl.lit(True).alias("is_home"),
            ),
            schedules.select(
                "game_id",
                "season",
                "week",
                "kickoff_utc",
                "stadium_id",
                "roof",
                "wind",
                "div_game",
                pl.col("away_team").alias("team"),
                pl.col("home_team").alias("opponent"),
                pl.col("away_rest").alias("rest"),
                pl.col("spread_line").alias("_spread_home"),
                "total_line",
                pl.lit(False).alias("is_home"),
            ),
        ],
        how="vertical",
    ).join(stadiums, on="stadium_id", how="left")

    shielded = pl.Series(
        "wind_shielded",
        [is_wind_shielded(roof_type, None) for roof_type in long["roof_type"].to_list()],
    )

    return (
        long.with_columns(shielded)
        .with_columns(
            # Spread from THIS team's perspective: negative means favoured.
            pl.when(pl.col("is_home"))
            .then(pl.col("_spread_home"))
            .otherwise(-pl.col("_spread_home"))
            .alias("spread"),
            pl.when(pl.col("wind_shielded"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("wind").cast(pl.Float64))
            .alias("wind_observed"),
        )
        .with_columns(
            (pl.col("total_line") / 2.0 - pl.col("spread") / 2.0).alias("implied_team_total"),
        )
        .select(
            "game_id",
            "team",
            # kickoff_utc is deliberately not re-emitted: the panel already
            # carries it, and a second copy would collide on the join.
            "rest",
            "is_home",
            "div_game",
            "spread",
            "total_line",
            "implied_team_total",
            "wind_observed",
            "wind_shielded",
            "roof_type",
        )
    )


# ------------------------------------------------------------------ assembly


def build_features(
    panel: pl.DataFrame,
    schedules: pl.DataFrame,
    config: Config,
) -> pl.DataFrame:
    """The full as-of feature matrix, one row per player-game.

    `panel` must already carry `kickoff_utc` -- ordering by week alone is wrong
    for a Thursday game following a Monday one.
    """
    if "kickoff_utc" not in panel.columns:
        raise ValueError(
            "panel must carry kickoff_utc before features are built; "
            "join it from schedules first (week ordering is not kickoff ordering)"
        )

    team_games = team_game_totals_from_panel(panel)
    roles = role_features(panel, config)
    catch = catch_rate_features(panel, config)
    volume = team_volume_features(team_games)
    context = game_context_features(schedules)
    defence = defensive_rates(team_games).select(
        "game_id", pl.col("defense").alias("opponent"), "def_games_of_history", *DEFENSIVE_RATES
    )

    return (
        panel.join(roles, on=["game_id", "gsis_id"], how="left")
        .join(catch, on=["game_id", "gsis_id"], how="left")
        .join(volume, on=["game_id", "team"], how="left")
        .join(context, on=["game_id", "team"], how="left")
        .join(defence, on=["game_id", "opponent"], how="left")
        .sort("season", "week", "game_id", "team", "gsis_id")
    )


def team_game_totals_from_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Recover the per-team totals already carried on every panel row."""
    return (
        panel.select(
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
        )
        .unique(subset=["game_id", "team"])
        .sort("season", "week", "game_id", "team")
    )
