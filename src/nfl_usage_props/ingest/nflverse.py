"""Pull nflverse tables and persist them as season-partitioned parquet.

Idempotency contract
--------------------
A (table, season) pair is downloaded again only if BOTH of these are false:

  1. the parquet file already exists on disk, and
  2. the season is marked complete in the metadata DB.

A season is "complete" when every game in its schedule has a result and
`storage.completion_lag_days` have passed since the last game -- the lag exists
because nflverse back-fills corrections (snap counts, charting) for weeks after
a season ends. Once complete, the season is immutable and never re-fetched.

Running `ingest_all` twice in a row therefore does real work the first time and
nothing the second time, apart from the seasons still in progress.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import nflreadpy as nfl
import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.metadata import MetadataStore
from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore

# Kickoff times in nflverse `schedules` are US Eastern wall-clock.
SCHEDULE_TZ = "America/New_York"


@dataclass(frozen=True)
class TableSpec:
    """How to fetch one nflverse table, and what it means for leakage."""

    name: str
    loader: Callable[..., pl.DataFrame]
    seasonal: bool = True
    # True when the table's rows only exist after a game is played. These
    # tables can be used as *history* for a future game, but a row's own game
    # must never contribute features to itself. Stage 2's as-of joins depend
    # on this flag.
    post_game: bool = True
    description: str = ""


def _load_seasonal(fn: Callable[..., pl.DataFrame], **kwargs):
    def loader(season: int) -> pl.DataFrame:
        return fn(seasons=[season], **kwargs)

    return loader


TABLES: dict[str, TableSpec] = {
    "schedules": TableSpec(
        name="schedules",
        loader=_load_seasonal(nfl.load_schedules),
        post_game=False,
        description=(
            "Game-level schedule with kickoff date/time, closing spread and total, "
            "roof/surface, rest days. Pre-game fields are known before kickoff; "
            "score/result columns are not."
        ),
    ),
    "players": TableSpec(
        name="players",
        loader=lambda season: nfl.load_players(),
        seasonal=False,
        post_game=False,
        description="Player ID crosswalk (gsis/pfr/pff/espn/ngs) plus biographical fields.",
    ),
    "pbp": TableSpec(
        name="pbp",
        loader=_load_seasonal(nfl.load_pbp),
        description=(
            "Play-by-play. Source of team plays, pass/rush split, PROE (xpass/pass_oe), "
            "targets and carries by player."
        ),
    ),
    "player_stats": TableSpec(
        name="player_stats",
        loader=_load_seasonal(nfl.load_player_stats, summary_level="week"),
        description="Weekly per-player box score: targets, receptions, carries, yards.",
    ),
    "snap_counts": TableSpec(
        name="snap_counts",
        loader=_load_seasonal(nfl.load_snap_counts),
        description=(
            "PFR snap counts per player-game (offense/defense/ST, count and pct). "
            "Keyed on pfr_player_id -- needs the players crosswalk to join to gsis_id."
        ),
    ),
    "depth_charts": TableSpec(
        name="depth_charts",
        loader=_load_seasonal(nfl.load_depth_charts),
        post_game=False,
        description=(
            "Depth chart position and rank. **Two incompatible schemas**: through 2024 "
            "an undated weekly snapshot (season/week/club_code/depth_team); from 2025 "
            "a different feed keyed on a `dt` UTC timestamp (dt/team/pos_rank/pos_slot) "
            "with no `week` column. The 2025+ feed is exactly as-of filterable; the "
            "legacy one needs a week lag. Any code touching this table must handle both."
        ),
    ),
    "injuries": TableSpec(
        name="injuries",
        loader=_load_seasonal(nfl.load_injuries),
        post_game=False,
        description=(
            "Weekly injury report with practice and game status. Carries a real "
            "`date_modified` UTC timestamp -- the only table that supports exact "
            "as-of filtering without a lag heuristic."
        ),
    ),
    "rosters_weekly": TableSpec(
        name="rosters_weekly",
        loader=_load_seasonal(nfl.load_rosters_weekly),
        post_game=False,
        description="Weekly roster status (active/inactive/IR) per player.",
    ),
    "participation": TableSpec(
        name="participation",
        loader=_load_seasonal(nfl.load_participation),
        description=(
            "Per-play personnel: `offense_players` lists the 11 gsis_ids on the field. "
            "Joined to pbp dropbacks this yields pass-play participation, our routes-run "
            "proxy. See README limitations."
        ),
    ),
}


@dataclass
class IngestResult:
    table: str
    season: int
    status: str  # fetched | skipped_complete | unavailable | error
    rows: int | None = None
    bytes_written: int | None = None
    duration_s: float | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


def is_unavailable_upstream(exc: Exception, *, season_started: bool) -> bool:
    """True when a fetch failed because the season simply has not happened yet.

    A future season is not an error. nflverse publishes `schedules` and
    `depth_charts` for an upcoming season months before week 1, while `pbp`,
    `snap_counts` and friends do not exist until games are played, so a full
    ingest run in e.g. August legitimately cannot fetch every table for the
    upcoming season.

    Two signals, and the second is deliberately narrow:

    * nflreadpy raises `ValueError: Season must be between X and Y` when the
      requested season is outside the range it knows about. That is upstream
      telling us the data does not exist -- unambiguous.
    * A download failure (404) for a season that has not kicked off yet. This
      is only treated as "unavailable" when we can see from the schedule that
      no game has been played, so a genuine outage during the season still
      surfaces as an error rather than being silently swallowed.
    """
    message = str(exc)
    if isinstance(exc, ValueError) and "Season must be between" in message:
        return True
    if not season_started and "Failed to download" in message:
        return True
    return False


def season_has_started(schedules: pl.DataFrame, season: int) -> bool:
    """True once at least one game of `season` has been played."""
    if schedules.is_empty() or "result" not in schedules.columns:
        return False
    games = schedules.filter(pl.col("season") == season)
    if games.is_empty():
        return False
    return games.filter(pl.col("result").is_not_null()).height > 0


# --------------------------------------------------------------------- helpers


def add_kickoff_utc(schedules: pl.DataFrame) -> pl.DataFrame:
    """Attach a real UTC kickoff timestamp to the schedule.

    `gameday` is a date string and `gametime` is Eastern wall-clock "HH:MM".
    Every as-of feature computation in Stage 2 keys off this column, so it is
    derived once, here, rather than re-parsed at each call site.

    Rows with a missing `gametime` (occasionally true for future games not yet
    scheduled to a slot) get a null kickoff, which downstream as-of joins must
    treat as "unknown" rather than "midnight".
    """
    if schedules.is_empty():
        return schedules.with_columns(
            pl.lit(None, dtype=pl.Datetime(time_unit="us", time_zone="UTC")).alias("kickoff_utc")
        )

    naive = (
        pl.concat_str(
            [pl.col("gameday").cast(pl.Utf8), pl.lit(" "), pl.col("gametime").cast(pl.Utf8)]
        )
        .str.strptime(pl.Datetime(time_unit="us"), "%Y-%m-%d %H:%M", strict=False)
        .alias("_naive")
    )
    return (
        schedules.with_columns(naive)
        .with_columns(
            pl.when(pl.col("_naive").is_null())
            .then(None)
            .otherwise(
                pl.col("_naive")
                .dt.replace_time_zone(SCHEDULE_TZ, ambiguous="earliest", non_existent="null")
                .dt.convert_time_zone("UTC")
            )
            .alias("kickoff_utc")
        )
        .drop("_naive")
    )


def season_is_complete(
    schedules: pl.DataFrame,
    season: int,
    *,
    lag_days: int,
    now: datetime | None = None,
) -> bool:
    """True when `season` is finished and safe to treat as immutable."""
    now = now or datetime.now(UTC)
    if schedules.is_empty():
        return False

    games = schedules.filter(pl.col("season") == season)
    if games.is_empty():
        return False

    # Every game must have been played.
    if "result" not in games.columns:
        return False
    if games.filter(pl.col("result").is_null()).height > 0:
        return False

    # And the lag must have elapsed since the final game.
    last_gameday = games.select(pl.col("gameday").max()).item()
    if last_gameday is None:
        return False
    try:
        last_dt = datetime.strptime(str(last_gameday), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return False
    return now >= last_dt + timedelta(days=lag_days)


def _fetch_schedules_for(seasons: list[int]) -> pl.DataFrame:
    """Schedules for completeness checks, fetched straight from nflverse."""
    return nfl.load_schedules(seasons=seasons)


# ------------------------------------------------------------------- ingestion


def ingest_table_season(
    spec: TableSpec,
    season: int,
    *,
    store: ParquetStore,
    meta: MetadataStore,
    complete: bool,
    force: bool = False,
    started_upstream: bool = True,
) -> IngestResult:
    """Fetch and persist one (table, season). Skips completed seasons."""
    partition = season if spec.seasonal else SEASONLESS_SENTINEL

    if not force and complete and store.exists(spec.name, partition):
        result = IngestResult(
            table=spec.name,
            season=partition,
            status="skipped_complete",
            message="season complete and already on disk",
        )
        meta.record_ingest(
            table_name=result.table,
            season=result.season,
            status=result.status,
            message=result.message,
        )
        return result

    started = time.perf_counter()
    try:
        df = spec.loader(season)
    except Exception as exc:  # noqa: BLE001 - we want the message recorded, not raised
        duration = time.perf_counter() - started
        unavailable = is_unavailable_upstream(exc, season_started=started_upstream)
        result = IngestResult(
            table=spec.name,
            season=partition,
            status="unavailable" if unavailable else "error",
            duration_s=duration,
            message=f"{type(exc).__name__}: {exc}",
        )
        meta.record_ingest(
            table_name=result.table,
            season=result.season,
            status=result.status,
            duration_s=duration,
            message=result.message,
        )
        return result

    if spec.name == "schedules":
        df = add_kickoff_utc(df)

    store.write(df, spec.name, partition)
    duration = time.perf_counter() - started
    size = store.size_bytes(spec.name, partition)

    meta.set_season_state(
        table_name=spec.name,
        season=partition,
        complete=complete and spec.seasonal,
        rows=df.height,
    )
    meta.record_ingest(
        table_name=spec.name,
        season=partition,
        status="fetched",
        rows=df.height,
        bytes_written=size,
        duration_s=duration,
    )
    return IngestResult(
        table=spec.name,
        season=partition,
        status="fetched",
        rows=df.height,
        bytes_written=size,
        duration_s=duration,
    )


def ingest_all(
    config: Config,
    *,
    tables: list[str] | None = None,
    seasons: list[int] | None = None,
    force: bool = False,
    now: datetime | None = None,
    progress: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Ingest every configured table for every configured season.

    Idempotent: a second run re-fetches only seasons that are still in
    progress, unless `force=True`.
    """
    store = ParquetStore(config.raw_dir, compression=config.storage.compression)
    meta = MetadataStore(config.metadata_db)

    table_names = tables or list(config.ingest.tables)
    unknown = [t for t in table_names if t not in TABLES]
    if unknown:
        raise KeyError(f"unknown table(s): {unknown}. known: {sorted(TABLES)}")

    season_list = seasons if seasons is not None else config.ingest.seasons(now)

    # One schedules pull drives every completeness decision below.
    schedules = _fetch_schedules_for(season_list)
    completeness = {
        s: season_is_complete(schedules, s, lag_days=config.storage.completion_lag_days, now=now)
        for s in season_list
    }
    # Distinguishes "this season has not happened yet" from "the download broke".
    started = {s: season_has_started(schedules, s) for s in season_list}

    results: list[IngestResult] = []
    seen_seasonless: set[str] = set()

    for name in table_names:
        spec = TABLES[name]
        if not spec.seasonal:
            if name in seen_seasonless:
                continue
            seen_seasonless.add(name)
            # Seasonless tables are current-state snapshots: always refresh.
            result = ingest_table_season(
                spec,
                SEASONLESS_SENTINEL,
                store=store,
                meta=meta,
                complete=False,
                force=True,
            )
            results.append(result)
            if progress:
                progress(result)
            continue

        for season in season_list:
            result = ingest_table_season(
                spec,
                season,
                store=store,
                meta=meta,
                complete=completeness.get(season, False),
                force=force,
                started_upstream=started.get(season, True),
            )
            results.append(result)
            if progress:
                progress(result)

    return results
