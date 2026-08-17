"""Committed reference tables that nflverse does not provide.

These live in `reference/` at the repo root, not in `data/`, because they are
hand-maintained inputs rather than downloaded artifacts.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_usage_props.config import REPO_ROOT

REFERENCE_DIR = REPO_ROOT / "reference"
STADIUMS_CSV = REFERENCE_DIR / "stadiums.csv"
NAME_OVERRIDES_CSV = REFERENCE_DIR / "player_name_overrides.csv"

# Roof types where the field is shielded from wind. `retractable` is
# deliberately absent: whether it is open is a game-time decision, so it must
# be read from the observed `roof` column and never assumed.
WIND_SHIELDED_ROOF_TYPES = frozenset({"dome"})

# nflverse is not internally consistent about relocated franchises: `pbp`
# rewrites history to the current abbreviation (a 2016 Raiders game says LV),
# while `schedules` preserves what the team was called at the time (OAK). Left
# alone this silently drops the join for 81 team-games, which does not look
# like a bug -- it looks like those teams having slightly less history.
#
# Everything is normalised TO the current code, matching pbp, because pbp is
# the larger and more widely joined table.
TEAM_RELOCATIONS = {
    "OAK": "LV",  # Oakland -> Las Vegas, 2020
    "SD": "LAC",  # San Diego -> Los Angeles, 2017
    "STL": "LA",  # St. Louis -> Los Angeles, 2016
}


def normalize_team(column: str = "team") -> pl.Expr:
    """Map historical team abbreviations onto their current codes."""
    return pl.col(column).replace(TEAM_RELOCATIONS)


def load_stadiums(path: Path | None = None) -> pl.DataFrame:
    """Stadium coordinates and roof type, keyed on nflverse `stadium_id`.

    Keyed on id rather than name because stadiums get renamed constantly
    (KAN00 is both "Arrowhead Stadium" and "GEHA Field at Arrowhead Stadium";
    PIT00 is both "Heinz Field" and "Acrisure Stadium") while the id is stable.

    `roof_type` is the stadium's FIXED characteristic and is safe to use as a
    pre-kickoff feature. It is not the same as `schedules.roof`, which records
    the observed state on the day and is only known afterwards for retractable
    roofs. See `is_wind_shielded`.

    Coordinates are stadium centroids to ~4 decimal places. That is far more
    precision than wind needs -- wind fields vary on kilometre scales -- so the
    approximation is immaterial for the only weather variable we use.
    """
    return pl.read_csv(path or STADIUMS_CSV, comment_prefix="#")


def load_name_overrides(path: Path | None = None) -> pl.DataFrame:
    """Manual book-name -> gsis_id overrides. May legitimately be empty."""
    frame = pl.read_csv(
        path or NAME_OVERRIDES_CSV,
        comment_prefix="#",
        schema_overrides={"book_name": pl.Utf8, "team": pl.Utf8, "gsis_id": pl.Utf8},
    )
    for column in ("book_name", "team", "gsis_id", "note"):
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column))
    return frame


def is_wind_shielded(roof_type: str | None, observed_roof: str | None) -> bool:
    """Whether wind should be zeroed for this game.

    Two distinct inputs, deliberately:

    * `roof_type` is the stadium's fixed characteristic from `stadiums.csv` and
      is known before kickoff.
    * `observed_roof` is `schedules.roof`, which for a retractable stadium
      records what actually happened and is therefore NOT known before kickoff.

    A fixed dome is shielded regardless. A retractable stadium is shielded only
    when the observed state says `closed` -- and because that is post-kickoff
    information, a forward projection must treat a retractable roof as unknown
    rather than guessing. Returning False there means the wind forecast is
    applied, which is the conservative direction: it is better to model wind
    that turned out to be absent than to zero out wind that was really blowing.
    """
    if roof_type in WIND_SHIELDED_ROOF_TYPES:
        return True
    return observed_roof == "closed"
