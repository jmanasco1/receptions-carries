"""Feature rows for games that have not been played.

The panel is built from play-by-play, which means it only contains games that
already happened. That is exactly right for fitting and for backtesting, and
completely useless on a Sunday morning: an unplayed game has no plays, so it
has no panel rows, so it has no features and cannot be projected. Everything
upstream of here works forward; this module is what lets the pipeline point at
next week instead of last week.

How it works
------------
Rather than a separate prediction path -- which would drift out of sync with
the fitting path and quietly start computing features differently -- this emits
*pseudo panel rows* for the upcoming slate: one row per (game, expected
player), with every outcome column null.

Null outcomes are not a placeholder to be filled in. They are the correct
input. The role tracker already treats a game with no opportunities as "no
information this week": the state drifts, uncertainty widens, the level is
carried. That is precisely what should happen to a player's role estimate
between his last game and his next one, and it is the same code path a bye
week takes. Every other feature -- trailing team volume, catch rate prior,
opponent rates -- is a cumulative sum shifted by one, so they carry forward
onto these rows without knowing they are special.

Who is expected to play
-----------------------
From `depth_charts`, which for 2025+ carries a real UTC timestamp and is
therefore filterable as-of kickoff rather than needing a week lag. Capped by
depth rank per position: the raw feed lists about 29 eligible players a team,
which is practice-squad deep, against roughly 12 who take a meaningful snap.
Leaving the tail in would put two dozen near-zero priors on Layer 3's simplex
and dilute every real share.

This is a roster guess refined by the injury report, not an inactives list.
True inactives post 90 minutes before kickoff and are not published in any feed
this project can reach. What `injuries` does carry is the weekly participation
report, and a player designated Out or Doubtful is dropped from the simplex --
he is not going to take a target, and leaving him on it hands him share that
belongs to the players who will.

Two limits worth knowing before trusting it. The report fills in across the
week: a Wednesday opening snapshot sees a nearly empty report, and the Friday
designations are what actually matter, so this helps closing pulls far more
than opening ones. And it is small. Measured over 2026 weeks 1-5 it removes
0.2-0.6 players per team, against a roster of 12.4 where only about 9.7 record
a touch. Most of the players on the simplex who never touch the ball are
healthy -- a WR5, a blocking TE -- and no injury feed will ever remove them.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nfl_usage_props.reference import normalize_team

# How deep each position's rotation actually goes, MEASURED from 2023-25
# played games rather than guessed. Mean distinct players per team-game:
#
#   WR 4.87   TE 2.98   RB 2.74   QB 1.23   FB 1.00     total 12.8
#
# These caps sum to 13, which matches. An earlier set summing to 16 looked
# harmlessly generous and was not: Layer 3 normalises across whoever is on the
# simplex, so three extra depth players who will never see the field take a
# share of the team's targets from the players who will. The effect was a 22%
# systematic under-projection on exactly the featured players books post lines
# for -- every single flagged market came back "Under", which is the signature
# of a biased projection rather than a real edge.
DEPTH_LIMITS: dict[str, int] = {"QB": 1, "RB": 3, "WR": 5, "TE": 3, "FB": 1}

# What the caps above should produce, per team-game. Checked at build time --
# a roster far from this means the depth feed changed shape, and the failure
# mode is a quiet bias rather than an error.
EXPECTED_ROSTER_SIZE = (10, 15)

# Columns the panel carries that an unplayed game cannot have. Emitted as null
# so the tracker reads them as "no observation" rather than as a real zero.
OUTCOME_COLUMNS = (
    "targets",
    "receptions",
    "carries",
    "snaps",
    "pass_snaps",
    "team_plays",
    "team_dropbacks",
    "team_rush_attempts",
    "team_targets",
    "team_carries",
    "team_completions",
    "team_proe",
    "team_xpass",
    "team_pass_rate",
    "target_share",
    "carry_share",
    "snap_share",
    "route_share",
    "catch_rate",
)


# Participation-report designations that mean the player will not take a snap.
# Questionable is deliberately absent: those players mostly play, and dropping
# them would hand their share to a teammate who is not going to get it either.
RULED_OUT = ("Out", "Doubtful")


def expected_rosters(
    depth_charts: pl.DataFrame,
    as_of,
    limits: dict[str, int] | None = None,
    injuries: pl.DataFrame | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
) -> pl.DataFrame:
    """Eligible players per team from the latest depth chart before `as_of`.

    Requires the 2025+ feed, which carries a `dt` timestamp. The legacy feed is
    an undated weekly snapshot and cannot be filtered to a kickoff, so this
    refuses it rather than silently using a chart published after the game.

    When `injuries` is supplied along with `season` and `week`, players carrying
    an Out or Doubtful designation for that week are removed.
    """
    if "dt" not in depth_charts.columns:
        raise ValueError(
            "depth_charts has no `dt` column, so it is the pre-2025 undated feed "
            "and cannot be filtered as-of a kickoff. Forward projection needs the "
            "timestamped feed."
        )
    limits = limits or DEPTH_LIMITS

    parsed = depth_charts.with_columns(
        pl.col("dt").cast(pl.String).str.to_datetime(strict=False, time_zone="UTC").alias("_dt")
    ).filter(pl.col("_dt").is_not_null() & (pl.col("_dt") <= as_of))
    if parsed.is_empty():
        # Built from an explicit schema, not by selecting literals off the
        # empty frame: `pl.lit` broadcasts, so that route returns one row of
        # nulls and an "empty" roster silently becomes a phantom player.
        return pl.DataFrame(
            schema={
                "team": pl.String,
                "gsis_id": pl.String,
                "position": pl.String,
                "depth_rank": pl.Int32,
            }
        )

    latest = parsed.filter(pl.col("_dt") == parsed["_dt"].max())
    caps = pl.DataFrame({"position": list(limits), "_limit": [limits[p] for p in limits]})
    roster = (
        latest.select(
            normalize_team("team").alias("team"),
            "gsis_id",
            pl.col("pos_abb").alias("position"),
            pl.col("pos_rank").alias("depth_rank"),
        )
        .drop_nulls(["gsis_id", "team", "position"])
        .join(caps, on="position", how="inner")
        .filter(pl.col("depth_rank") <= pl.col("_limit"))
        .drop("_limit")
        .unique(subset=["team", "gsis_id"])
    )
    return _drop_ruled_out(roster, injuries, season=season, week=week)


def _drop_ruled_out(
    roster: pl.DataFrame,
    injuries: pl.DataFrame | None,
    *,
    season: int | None,
    week: int | None,
) -> pl.DataFrame:
    """Remove players the participation report rules out for this week.

    Scoped to the exact (season, week). An unscoped anti-join would delete a
    player for every week of the season because he missed one, which is a far
    worse error than leaving him in.
    """
    if injuries is None or injuries.is_empty() or season is None or week is None:
        return roster
    required = {"season", "week", "gsis_id", "report_status"}
    if not required <= set(injuries.columns):
        return roster
    ruled_out = (
        injuries.filter(
            (pl.col("season") == season)
            & (pl.col("week") == week)
            & pl.col("report_status").is_in(RULED_OUT)
        )
        .select("gsis_id")
        .drop_nulls()
        .unique()
    )
    if ruled_out.is_empty():
        return roster
    return roster.join(ruled_out, on="gsis_id", how="anti")


def upcoming_panel_rows(
    schedules: pl.DataFrame,
    depth_charts: pl.DataFrame,
    *,
    season: int,
    week: int,
    limits: dict[str, int] | None = None,
    injuries: pl.DataFrame | None = None,
    skip_game_ids: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Panel-shaped rows for one unplayed slate, with null outcomes.

    `skip_game_ids` drops games that already have real panel rows. An NFL week
    is not played all at once -- Thursday night kicks off four days before the
    Sunday slate -- so by Friday the week is neither unplayed nor complete.
    Treating it as all-or-nothing is what produced a week 6 report covering a
    single game.
    """
    slate = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))
    if slate.is_empty():
        raise ValueError(f"no scheduled games for {season} week {week}")
    if skip_game_ids:
        slate = slate.filter(~pl.col("game_id").is_in(list(skip_game_ids)))
        if slate.is_empty():
            raise ValueError(
                f"every game in {season} week {week} already has panel rows; "
                "there is nothing left to project"
            )

    kickoff = slate["kickoff_utc"].max()
    rosters = expected_rosters(
        depth_charts, kickoff, limits, injuries, season=season, week=week
    )
    if rosters.is_empty():
        raise ValueError(f"no depth chart published before {kickoff}; cannot guess who plays")

    sides = pl.concat(
        [
            slate.select(
                "season",
                "week",
                "game_id",
                "kickoff_utc",
                pl.col("home_team").alias("team"),
                pl.col("away_team").alias("opponent"),
            ),
            slate.select(
                "season",
                "week",
                "game_id",
                "kickoff_utc",
                pl.col("away_team").alias("team"),
                pl.col("home_team").alias("opponent"),
            ),
        ]
    ).with_columns(
        normalize_team("team").alias("team"), normalize_team("opponent").alias("opponent")
    )

    rows = sides.join(rosters, on="team", how="inner").with_columns(
        pl.lit("REG").alias("season_type"),
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in OUTCOME_COLUMNS],
    )

    # Guard the share denominator. Too many names on the simplex silently
    # deflates every real player's projection, and nothing about the output
    # looks wrong -- it just reads Under on everything.
    per_team = rows.group_by("game_id", "team").len()["len"]
    if not per_team.is_empty():
        median = float(per_team.median())
        low, high = EXPECTED_ROSTER_SIZE
        if not low <= median <= high:
            raise ValueError(
                f"expected roster is {median:.0f} players per team, outside the "
                f"{low}-{high} a real offence uses. Layer 3 normalises across "
                "this set, so the count directly scales every projection. "
                "Check DEPTH_LIMITS against the depth chart feed."
            )
    return rows.drop("depth_rank").sort("game_id", "team", "gsis_id")


def panel_with_upcoming(
    panel: pl.DataFrame,
    schedules: pl.DataFrame,
    depth_charts: pl.DataFrame,
    *,
    season: int,
    week: int,
    limits: dict[str, int] | None = None,
    injuries: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """History plus the unplayed part of the slate, ready for `build_features`.

    Concatenated rather than handled separately so the upcoming rows go through
    exactly the same feature code as every fitted row. A separate prediction
    path is how training and serving drift apart.

    Games in the target week that already have panel rows are left alone: they
    have been played, their outcomes are real, and synthesising null-outcome
    rows over the top of them would project a result that is already known.
    """
    played = (
        panel.filter((pl.col("season") == season) & (pl.col("week") == week))["game_id"]
        .unique()
        .to_list()
    )
    upcoming = upcoming_panel_rows(
        schedules,
        depth_charts,
        season=season,
        week=week,
        limits=limits,
        injuries=injuries,
        skip_game_ids=played,
    )
    history = panel.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    )
    return pl.concat([history, upcoming], how="diagonal_relaxed").sort(
        "season", "week", "game_id", "team", "gsis_id"
    )
