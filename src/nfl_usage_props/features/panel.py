"""The player-game usage panel: one row per (game, player) of realised usage.

Everything in this module is post-game truth. It is the history the role
tracker filters and the outcome the model is scored against. It is never a
same-game feature -- see `features.build` for that side of the wall.

Why shares are computed against play-level denominators
-------------------------------------------------------
Layer 3 is a Dirichlet-Multinomial over the roster, so the shares it consumes
must sum to one across the team by construction. That rules out the convenient
denominators:

* `player_stats.target_share` is already published by nflverse, but it is
  computed against a slightly different play universe and does not sum to
  exactly one within a game.
* Dividing by dropbacks rather than targets would make the shares sum to the
  team's completion-adjusted target rate, which is not one.

So targets are divided by *team targets* and carries by *team carries*, both
counted from the same filtered play universe on the same rows. The invariant
is asserted in the tests, not hoped for.

Snaps come from `participation`, not `snap_counts`
--------------------------------------------------
`snap_counts` is keyed on `pfr_player_id` and needs the crosswalk to join,
loses players the crosswalk misses, and gives one number per game.
`participation` lists the eleven gsis_ids on the field for every play, which
gives the same snap count *and* lets it be split by play type -- so the
routes-run proxy (snaps on dropbacks) falls out of the same pass. The two
agree closely; `tests/test_panel.py` pins the agreement so a silent
divergence in either feed shows up as a failure.
"""

from __future__ import annotations

import polars as pl

# A play is "offensive" for our purposes when it is a real scrimmage snap.
# `play_type` already separates kneels and spikes into their own categories,
# and `no_play` (a penalty that wiped the snap) is excluded because a wiped
# play produces no target and no carry, so counting it would deflate shares.
#
# Two known, measured divergences from nflverse's own `player_stats` follow
# from this, both deliberate:
#
#   * Kneels. nflverse counts them as carries. We do not -- a kneel is a clock
#     play, not an opportunity anyone competes for. This is the whole of the QB
#     difference and it is large (~400 carries a season).
#   * Penalty-after-the-fact. A penalty enforced *after* a play that stood is
#     still tagged `no_play`, so we drop a carry that counted. Vanishingly rare
#     (one non-QB instance in 2023-24) and not worth description parsing to
#     recover. `tests/test_features_real.py` bounds it rather than assuming it.
OFFENSIVE_PLAY_TYPES = ("pass", "run")

# Columns the panel needs from pbp. Reading a 400-column frame ten times over
# is the difference between a 3-second build and a 90-second one.
PBP_COLUMNS = (
    "season",
    "week",
    "season_type",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "play_type",
    "two_point_attempt",
    "qb_dropback",
    "pass_attempt",
    "rush_attempt",
    "complete_pass",
    "receiver_player_id",
    "rusher_player_id",
    "pass_oe",
    "xpass",
)


def _offensive_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Filter to real scrimmage snaps with a known offense.

    The player-id columns are cast explicitly because polars infers a Null
    dtype for a column that happens to be entirely null in the slice being
    read -- which is ordinary for a single game with no designed QB runs, and
    turns a later join into a dtype error rather than an empty result.
    """
    return pbp.filter(
        pl.col("play_type").is_in(OFFENSIVE_PLAY_TYPES)
        & (pl.col("two_point_attempt").fill_null(0) == 0)
        & pl.col("posteam").is_not_null()
    ).with_columns(
        pl.col("receiver_player_id").cast(pl.String),
        pl.col("rusher_player_id").cast(pl.String),
    )


def team_game_totals(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (game, team) offensive volume and tendency.

    `team_targets` and `team_carries` are the share denominators. `plays`,
    `dropbacks` and `rush_attempts` are the volume Layers 1 and 2 predict.
    """
    plays = _offensive_plays(pbp)
    return (
        plays.group_by("season", "week", "season_type", "game_id", "posteam", "defteam")
        .agg(
            pl.len().alias("team_plays"),
            pl.col("qb_dropback").fill_null(0).sum().cast(pl.Int32).alias("team_dropbacks"),
            pl.col("rush_attempt").fill_null(0).sum().cast(pl.Int32).alias("team_rush_attempts"),
            pl.col("receiver_player_id").is_not_null().sum().cast(pl.Int32).alias("team_targets"),
            pl.col("rusher_player_id").is_not_null().sum().cast(pl.Int32).alias("team_carries"),
            pl.col("complete_pass").fill_null(0).sum().cast(pl.Int32).alias("team_completions"),
            pl.col("pass_oe").mean().alias("team_proe"),
            pl.col("xpass").mean().alias("team_xpass"),
        )
        .rename({"posteam": "team", "defteam": "opponent"})
        .with_columns(
            (pl.col("team_dropbacks") / pl.col("team_plays")).alias("team_pass_rate"),
        )
        .sort("season", "week", "game_id", "team")
    )


def player_game_usage(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per (game, player) targets, receptions and carries.

    Built by unpivoting the two player-id columns rather than joining two
    aggregations, so a player who both caught and carried in the same game
    lands on one row instead of being counted twice or dropped.
    """
    plays = _offensive_plays(pbp)

    receiving = (
        plays.filter(pl.col("receiver_player_id").is_not_null())
        .group_by("game_id", "posteam", pl.col("receiver_player_id").alias("gsis_id"))
        .agg(
            pl.len().cast(pl.Int32).alias("targets"),
            pl.col("complete_pass").fill_null(0).sum().cast(pl.Int32).alias("receptions"),
        )
    )
    rushing = (
        plays.filter(pl.col("rusher_player_id").is_not_null())
        .group_by("game_id", "posteam", pl.col("rusher_player_id").alias("gsis_id"))
        .agg(pl.len().cast(pl.Int32).alias("carries"))
    )

    return (
        receiving.join(rushing, on=["game_id", "posteam", "gsis_id"], how="full", coalesce=True)
        .rename({"posteam": "team"})
        .with_columns(
            pl.col("targets").fill_null(0),
            pl.col("receptions").fill_null(0),
            pl.col("carries").fill_null(0),
        )
    )


def player_game_snaps(pbp: pl.DataFrame, participation: pl.DataFrame) -> pl.DataFrame:
    """Per (game, player) offensive snaps and snaps on dropbacks.

    `pass_snaps` is the routes-run proxy: on the field when the quarterback
    dropped back. It over-counts (a back who stayed in to block ran no route)
    and the error is largest exactly where it matters least -- for backs, whose
    receptions we model off carries anyway. `participation.route` cannot fix
    this: it records the route of the *targeted* receiver only, one string per
    play, not a per-player route list.
    """
    if participation.is_empty():
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "gsis_id": pl.String,
                "team": pl.String,
                "snaps": pl.Int32,
                "pass_snaps": pl.Int32,
            }
        )

    plays = _offensive_plays(pbp).select(
        "game_id", "play_id", "posteam", pl.col("qb_dropback").fill_null(0).alias("qb_dropback")
    )
    on_field = (
        participation.select(
            pl.col("nflverse_game_id").alias("game_id"),
            "play_id",
            "offense_players",
        )
        .filter(pl.col("offense_players").is_not_null() & (pl.col("offense_players") != ""))
        .with_columns(pl.col("offense_players").str.split(";").alias("gsis_id"))
        .explode("gsis_id")
        .filter(pl.col("gsis_id") != "")
        .join(plays, on=["game_id", "play_id"], how="inner")
    )
    return (
        on_field.group_by("game_id", "posteam", "gsis_id")
        .agg(
            pl.len().cast(pl.Int32).alias("snaps"),
            pl.col("qb_dropback").sum().cast(pl.Int32).alias("pass_snaps"),
        )
        .rename({"posteam": "team"})
    )


def build_panel(
    pbp: pl.DataFrame,
    participation: pl.DataFrame,
    *,
    positions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Assemble the player-game panel with shares.

    `positions` is an optional (gsis_id, position) frame -- normally
    `player_stats` reduced to its position column, which is more reliable than
    the `players` crosswalk for what someone actually played in a season.
    """
    teams = team_game_totals(pbp)
    usage = player_game_usage(pbp)
    snaps = player_game_snaps(pbp, participation)

    panel = (
        usage.join(snaps, on=["game_id", "team", "gsis_id"], how="full", coalesce=True)
        .with_columns(
            pl.col("targets").fill_null(0),
            pl.col("receptions").fill_null(0),
            pl.col("carries").fill_null(0),
        )
        .join(teams, on=["game_id", "team"], how="inner")
    )

    if positions is not None and not positions.is_empty():
        panel = panel.join(positions, on="gsis_id", how="left")

    return panel.with_columns(
        _share("targets", "team_targets").alias("target_share"),
        _share("carries", "team_carries").alias("carry_share"),
        _share("snaps", "team_plays").alias("snap_share"),
        _share("pass_snaps", "team_dropbacks").alias("route_share"),
        pl.when(pl.col("targets") > 0)
        .then(pl.col("receptions") / pl.col("targets"))
        .otherwise(None)
        .alias("catch_rate"),
    ).sort("season", "week", "game_id", "team", "gsis_id")


def _share(numerator: str, denominator: str) -> pl.Expr:
    """A share, null rather than infinite when the denominator is zero.

    A team with zero carries in a game is rare but real (and zero dropbacks
    happens in weather games). Null propagates into the tracker as "no
    observation", which is correct; 0/0 as a float would be silently poisonous.
    """
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )
