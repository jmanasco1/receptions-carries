"""Generate the data dictionary from the schemas actually on disk.

Hand-maintaining ~800 column descriptions across nine nflverse tables is a
losing game, and a stale dictionary is worse than none -- Stage 2's as-of joins
are built off the timing column here. So the doc is generated from the parquet
that ingestion actually wrote, and the interesting part (the timing
classification) lives in code where it can be tested.

Timing values
-------------
before   Known before kickoff. Safe as a feature for that same game.
after    Known only once the game is played. Usable as history for LATER
         games only; using it for its own game is leakage.
static   Effectively time-invariant (IDs, birth dates, physical attributes).
lagged   Published before kickoff, but nflverse stores one undated snapshot
         per week, so we cannot prove which side of kickoff a given value
         came from. Treat as `after` for its own game unless a lag is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.ingest.nflverse import TABLES
from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore

TIMING_NOTES = {
    "before": "Known before kickoff — safe as a same-game feature.",
    "after": "Known only after the game — history only, never same-game.",
    "static": "Time-invariant identity/biographical field.",
    "lagged": "Pre-game in principle, but undated — needs an explicit week lag.",
}

# Columns in `schedules` that describe the RESULT rather than the setup.
SCHEDULE_POST_GAME = {
    "away_score",
    "home_score",
    "result",
    "total",
    "overtime",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
}
# Weather is measured at/after kickoff in nflverse, not forecast.
SCHEDULE_POST_GAME |= {"temp", "wind"}
# `roof` records the OBSERVED state on the day. For a fixed dome or an open-air
# stadium that is a constant and knowable in advance -- but for the five
# retractable stadiums it is a game-time decision, so the column as a whole
# cannot be treated as pre-kickoff. Use `reference/stadiums.csv` roof_type for
# the fixed characteristic, and see `reference.is_wind_shielded`.
SCHEDULE_POST_GAME |= {"roof"}

STATIC_ID_COLUMNS = {
    "gsis_id",
    "pfr_id",
    "pff_id",
    "otc_id",
    "espn_id",
    "esb_id",
    "nfl_id",
    "smart_id",
    "birth_date",
    "college_name",
    "college_conference",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
    "rookie_season",
    "height",
    "weight",
}

# nflverse replaced the depth chart feed for 2025. The two schemas share almost
# no columns and, critically, differ in whether they can be filtered as-of:
#
#   <= 2024: season/week/club_code/depth_team, ONE undated snapshot per week.
#            Cannot prove which side of kickoff a value came from -> `lagged`.
#   >= 2025: dt/team/pos_rank/pos_slot, a real ISO-8601 UTC timestamp per
#            snapshot (221 distinct timestamps in 2025) -> exactly as-of
#            filterable, so `before`.
#
# Verified against the ingested 2024 and 2025 parquet.
DEPTH_CHART_LEGACY_COLUMNS = {
    "season",
    "week",
    "club_code",
    "game_type",
    "depth_team",
    "depth_position",
    "formation",
    "elias_id",
    "first_name",
    "last_name",
    "football_name",
    "full_name",
    "jersey_number",
    "position",
}

CURATED_NOTES: dict[tuple[str, str], str] = {
    ("schedules", "kickoff_utc"): (
        "DERIVED by this repo: gameday + gametime parsed as US/Eastern, converted "
        "to UTC. The as-of cutoff for every Stage 2 feature."
    ),
    ("schedules", "spread_line"): (
        "Closing spread from the home team's perspective. Drives Layer 2 "
        "(pass/rush split). Note this is the CLOSING line — using it as a feature "
        "is fine for modelling but it is not what you would have had on Wednesday."
    ),
    ("schedules", "total_line"): "Closing game total. Drives Layer 1 (team plays).",
    ("schedules", "away_rest"): "Days of rest — known pre-kickoff.",
    ("schedules", "home_rest"): "Days of rest — known pre-kickoff.",
    ("schedules", "roof"): (
        "OBSERVED roof state on the day (dome/outdoors/closed/open). For the five "
        "retractable stadiums this is a game-time decision and is NOT known before "
        "kickoff — using it as a forward feature leaks. Use roof_type from "
        "reference/stadiums.csv for the fixed characteristic instead."
    ),
    ("schedules", "wind"): (
        "OBSERVED wind. The only weather variable that reliably affects volume "
        "(above ~15mph pass attempts fall, rush attempts rise). Null for domes, "
        "which is structurally zero rather than missing. Forward projections need a "
        "forecast, not this column."
    ),
    ("schedules", "temp"): (
        "OBSERVED temperature. Deliberately UNUSED: apparent effects are confounded "
        "with team quality and game total."
    ),
    ("schedules", "stadium_id"): (
        "Stable stadium key — survives renames (KAN00 is both Arrowhead and GEHA "
        "Field). Join key for reference/stadiums.csv."
    ),
    ("pbp", "xpass"): "Model-expected pass probability. `pass_oe` = pass − xpass gives PROE.",
    ("pbp", "pass_oe"): "Pass rate over expected for the play. Aggregate for team PROE (Layer 2).",
    ("pbp", "receiver_player_id"): "Targeted receiver (gsis). Numerator of target share (Layer 3).",
    ("pbp", "rusher_player_id"): "Ball carrier (gsis). Numerator of carry share (Layer 3).",
    ("pbp", "complete_pass"): "1 on a completed pass. Numerator of catch rate (Layer 4).",
    ("participation", "offense_players"): (
        "Semicolon-delimited gsis_ids of the 11 offensive players on the play. "
        "Filtered to dropbacks this is the routes-run PROXY (Layer 3 exposure). "
        "Verified to join pbp pass plays 1:1 for 2024."
    ),
    ("participation", "route"): (
        "ONE route label per play, not per player — cannot be used to count "
        "routes run by individual receivers."
    ),
    (
        "participation",
        "offense_personnel",
    ): "e.g. '1 RB, 1 TE, 3 WR'. Personnel context for shares.",
    ("snap_counts", "offense_pct"): (
        "Share of team offensive snaps. The Stage 1 exposure term, superseded by "
        "pass-play participation for receptions."
    ),
    ("snap_counts", "pfr_player_id"): "PFR id — join via players.pfr_id to reach gsis_id.",
    ("injuries", "date_modified"): (
        "Real UTC timestamp. The ONLY nflverse field that supports exact as-of "
        "filtering without a lag heuristic."
    ),
    ("injuries", "report_status"): "Out / Doubtful / Questionable. Drives Layer 3 redistribution.",
    ("depth_charts", "depth_team"): (
        "LEGACY (<=2024) depth rank, 1 = starter. Prior for shrinkage (Layer 3/4). "
        "Undated weekly snapshot — needs a week lag."
    ),
    ("depth_charts", "dt"): (
        "MODERN (>=2025) ISO-8601 UTC snapshot timestamp. Makes depth charts exactly "
        "as-of filterable against schedules.kickoff_utc — no lag heuristic needed. "
        "Note there is no `week` column in this schema; join on time, not week."
    ),
    ("depth_charts", "pos_rank"): "MODERN (>=2025) depth rank within position, 1 = starter.",
    ("depth_charts", "pos_slot"): "MODERN (>=2025) ordering slot within the unit.",
}


@dataclass(frozen=True)
class ColumnDoc:
    table: str
    column: str
    dtype: str
    timing: str
    note: str
    # Which ingested seasons actually contain this column. nflverse adds,
    # removes and replaces columns between years (see depth_charts 2025), and a
    # dictionary that only described the latest season would quietly hide that
    # from Stage 2.
    seasons: str = ""


def classify(table: str, column: str, *, post_game_default: bool) -> str:
    """Timing class for one column. Tested in tests/test_datadict.py."""
    if table == "players":
        return "static" if column in STATIC_ID_COLUMNS else "before"

    if table == "schedules":
        if column == "kickoff_utc":
            return "before"
        return "after" if column in SCHEDULE_POST_GAME else "before"

    if table == "depth_charts":
        # The 2025+ feed carries `dt`, so it is exactly as-of filterable. The
        # legacy feed is an undated weekly snapshot.
        return "lagged" if column in DEPTH_CHART_LEGACY_COLUMNS else "before"

    if table == "injuries":
        # date_modified lets us filter exactly, so these are genuinely pre-game.
        return "before"

    if table == "rosters_weekly":
        return "lagged"

    return "after" if post_game_default else "before"


def summarise_seasons(seasons: list[int], all_seasons: list[int]) -> str:
    """Render a column's season coverage, highlighting partial coverage."""
    if not seasons:
        return "—"
    if set(seasons) == set(all_seasons):
        return "all"
    if seasons == list(range(min(seasons), max(seasons) + 1)):
        return f"{min(seasons)}–{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
    return ", ".join(str(s) for s in seasons)


def build(config: Config) -> list[ColumnDoc]:
    """Describe every column across every ingested season.

    Reads parquet *schemas* rather than data, so this stays fast even over the
    full 2016–present pbp corpus.
    """
    store = ParquetStore(config.raw_dir, compression=config.storage.compression)
    docs: list[ColumnDoc] = []

    for name in sorted(TABLES):
        spec = TABLES[name]
        partitions = [SEASONLESS_SENTINEL] if not spec.seasonal else store.available_seasons(name)
        partitions = [p for p in partitions if store.exists(name, p)]
        if not partitions:
            continue

        # column -> (dtype from the newest season, seasons it appears in)
        columns: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        for partition in partitions:
            schema = pl.read_parquet_schema(store.path_for(name, partition))
            for column, dtype in schema.items():
                columns.setdefault(column, []).append(partition)
                dtypes[column] = str(dtype)  # newest wins, partitions are sorted

        for column, present_in in columns.items():
            docs.append(
                ColumnDoc(
                    table=name,
                    column=column,
                    dtype=dtypes[column],
                    timing=classify(name, column, post_game_default=spec.post_game),
                    note=CURATED_NOTES.get((name, column), ""),
                    seasons=summarise_seasons(sorted(present_in), partitions),
                )
            )
    return docs


def render(config: Config, docs: list[ColumnDoc]) -> str:
    store = ParquetStore(config.raw_dir)
    lines: list[str] = [
        "# Data dictionary",
        "",
        "**Generated** by `nfl-props docs data-dictionary` from the parquet files on",
        "disk. Do not edit by hand — edit `src/nfl_usage_props/datadict.py` and",
        "regenerate.",
        "",
        "## Timing classes",
        "",
        "This column is the whole point of the document. Stage 2's leakage test",
        "asserts that no feature for a week-N game is built from an `after` column",
        "of that same game.",
        "",
        "| class | meaning |",
        "| --- | --- |",
    ]
    for key, note in TIMING_NOTES.items():
        lines.append(f"| `{key}` | {note} |")

    lines += [
        "",
        "### The `lagged` trap",
        "",
        "`rosters_weekly` and the pre-2025 `depth_charts` feed carry a `week` but no",
        "timestamp. nflverse stores one snapshot per week and overwrites it, so a row",
        "labelled week 5 may reflect a chart published *after* the week 5 game. Using",
        "week-N data for the week-N game is therefore leakage that no schema check",
        "will catch. Use week N-1, or accept the leak knowingly and document it.",
        "",
        "Two tables escape this:",
        "",
        "* `injuries` has `date_modified`, a real UTC timestamp.",
        "* **`depth_charts` from 2025 onward** switched to an entirely different feed",
        "  with a `dt` ISO-8601 UTC timestamp (221 distinct snapshots in 2025). Those",
        "  columns are `before`, not `lagged`, and join on time rather than `week` —",
        "  the modern schema has no `week` column at all.",
        "",
        "Both are filtered exactly against `schedules.kickoff_utc`.",
        "",
        "### Schema drift",
        "",
        "The `seasons` column on each table below records which ingested seasons",
        "actually contain that column. Anything other than `all` means the column",
        "appeared, disappeared, or was replaced partway through the corpus — read it",
        "before building a feature on that column.",
        "",
        "## Tables",
        "",
        "| table | rows (latest season) | seasons | columns | default timing |",
        "| --- | --- | --- | --- | --- |",
    ]

    by_table: dict[str, list[ColumnDoc]] = {}
    for doc in docs:
        by_table.setdefault(doc.table, []).append(doc)

    for name, cols in by_table.items():
        spec = TABLES[name]
        seasons = store.available_seasons(name)
        partition = SEASONLESS_SENTINEL if not spec.seasonal else (seasons or [0])[-1]
        rows = store.read(name, partition).height if store.exists(name, partition) else 0
        if not spec.seasonal:
            span = "n/a (snapshot)"
        elif seasons:
            span = f"{min(seasons)}–{max(seasons)}"
        else:
            span = "none on disk"
        lines.append(
            f"| [`{name}`](#{name}) | {rows:,} | {span} | {len(cols)} | "
            f"`{'after' if spec.post_game else 'before'}` |"
        )

    for name, cols in by_table.items():
        spec = TABLES[name]
        lines += [
            "",
            f"### `{name}`",
            "",
            spec.description,
            "",
            "| column | dtype | timing | seasons | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        for col in cols:
            lines.append(
                f"| `{col.column}` | `{col.dtype}` | `{col.timing}` | {col.seasons} | {col.note} |"
            )

    lines.append("")
    return "\n".join(lines)
