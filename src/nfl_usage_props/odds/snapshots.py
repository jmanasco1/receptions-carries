"""Prop snapshot logging — the CLV record.

Why this runs before any model exists
-------------------------------------
Historical prop lines are paid-tier at 10x credits, so ROI backtesting is off
the table on the free tier. Closing line value is the only validation signal
available, and unlike almost everything else in this project it CANNOT be
reconstructed later. A week that goes unlogged is gone permanently. So the
logger runs on its own, from Stage 1, with no dependency on projections.

Two ordering rules follow from that, and both are load-bearing:

1. **Raw bytes hit disk before anything is parsed.** A schema surprise in one
   book's payload must not cost the whole slate.
2. **Player-name resolution never gates logging.** Raw book strings are stored
   verbatim; `gsis_id` is nullable and filled by a later pass that can be re-run
   over history at any time. Hard-failing on an unmappable name here would
   destroy the very data the hard-fail rule exists to protect.

"Closing" is per game day
-------------------------
A Thursday game closes Thursday. One Sunday-morning pull would capture the TNF
line after the game had already been played, which is not a closing line at all.
`events_to_snapshot` selects only events inside the horizon that have not yet
passed the cutoff, so running the closer on Thursday, Sunday and Monday picks up
each game day's slate and costs about the same in total as one blanket pull.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.odds.parse import add_hold, parse_event_odds, parse_iso, rows_to_frame

SNAPSHOT_KINDS = ("open", "close", "repoll")


def make_snapshot_id(kind: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{kind}"


@dataclass
class SnapshotResult:
    snapshot_id: str
    kind: str
    events_requested: int
    events_fetched: int
    rows: int
    credits_spent: int
    credits_remaining: int | None
    path: Path | None
    raw_paths: list[Path]
    errors: list[str]


def events_to_snapshot(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    horizon_hours: int,
    close_cutoff_minutes: int,
) -> list[dict[str, Any]]:
    """Events worth pulling right now.

    Includes an event when it kicks off within `horizon_hours` and is still
    more than `close_cutoff_minutes` away. The lower bound is what makes the
    closing pull safe to run repeatedly: it will never spend credits on a game
    that has already started.
    """
    horizon = now + timedelta(hours=horizon_hours)
    cutoff = now + timedelta(minutes=close_cutoff_minutes)

    selected = []
    for event in events:
        commence = parse_iso(event.get("commence_time"))
        if commence is None:
            continue
        if cutoff <= commence <= horizon:
            selected.append(event)
    return sorted(selected, key=lambda e: parse_iso(e.get("commence_time")) or datetime.max)


class SnapshotStore:
    """Immutable, append-only snapshot storage.

    One parquet file per snapshot under `season=YYYY/`. Files are never
    overwritten or merged: prop history is unrecoverable, so an accidental
    clobber is not a recoverable mistake.
    """

    def __init__(self, root: str | Path, *, compression: str = "zstd"):
        self.root = Path(root)
        self.compression = compression

    def path_for(self, season: int, snapshot_id: str) -> Path:
        return self.root / f"season={season}" / f"{snapshot_id}.parquet"

    def write(self, frame: pl.DataFrame, season: int, snapshot_id: str) -> Path:
        path = self.path_for(season, snapshot_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"snapshot {snapshot_id} already exists at {path}; refusing to overwrite "
                "prop history"
            )
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp, compression=self.compression)
        tmp.replace(path)
        return path

    def read_all(self, seasons: list[int] | None = None) -> pl.DataFrame:
        if not self.root.exists():
            return pl.DataFrame()
        pattern = "season=*/*.parquet" if seasons is None else None
        if pattern:
            files = sorted(self.root.glob(pattern))
        else:
            files = [
                f
                for s in (seasons or [])
                for f in sorted((self.root / f"season={s}").glob("*.parquet"))
            ]
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")

    def snapshot_ids(self, season: int) -> list[str]:
        directory = self.root / f"season={season}"
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))


def season_for(commence: datetime | None, fallback: int) -> int:
    """NFL season label for a kickoff timestamp."""
    if commence is None:
        return fallback
    return commence.year - 1 if commence.month < 3 else commence.year


def take_snapshot(
    client: Any,
    config: Config,
    *,
    kind: str = "open",
    now: datetime | None = None,
    event_ids: list[str] | None = None,
    save_raw: bool = True,
) -> SnapshotResult:
    """Pull, persist raw, then parse and persist rows.

    Errors on individual events are collected rather than raised: a slate that
    is 15/16 logged beats a slate that raised on event 3. Credit-floor aborts
    stop the loop but keep everything already fetched.
    """
    if kind not in SNAPSHOT_KINDS:
        raise ValueError(f"kind must be one of {SNAPSHOT_KINDS}, got {kind!r}")

    now = now or datetime.now(UTC)
    snapshot_id = make_snapshot_id(kind, now)
    snap_cfg = config.odds.snapshots
    errors: list[str] = []

    events_resp = client.get_events()
    all_events = events_resp.data or []

    if event_ids is not None:
        selected = [e for e in all_events if e.get("id") in set(event_ids)]
    else:
        selected = events_to_snapshot(
            all_events,
            now=now,
            horizon_hours=snap_cfg.horizon_hours,
            close_cutoff_minutes=snap_cfg.close_cutoff_minutes,
        )

    raw_paths: list[Path] = []
    all_rows = []
    fetched = 0

    for event in selected:
        event_id = event.get("id")
        if not event_id:
            continue
        try:
            resp = client.get_event_odds(event_id)
        except Exception as exc:  # noqa: BLE001 - partial slate beats no slate
            errors.append(f"{event_id}: {type(exc).__name__}: {exc}")
            # A credit-floor abort will keep raising for every remaining event,
            # so stop rather than burning the loop.
            if type(exc).__name__ == "CreditFloorError":
                break
            continue

        fetched += 1
        if save_raw:
            raw_paths.append(client.save_snapshot(resp, f"props/{snapshot_id}/{event_id}"))

        all_rows.extend(
            parse_event_odds(
                resp.data or {},
                snapshot_id=snapshot_id,
                snapshot_kind=kind,
                fetched_at=now,
                markets=list(config.odds.prop_markets),
            )
        )

    frame = add_hold(rows_to_frame(all_rows))

    path = None
    if frame.height:
        season = season_for(frame["commence_time"].min(), fallback=now.year)
        store = SnapshotStore(config.props_dir, compression=config.storage.compression)
        path = store.write(frame, season, snapshot_id)

    return SnapshotResult(
        snapshot_id=snapshot_id,
        kind=kind,
        events_requested=len(selected),
        events_fetched=fetched,
        rows=frame.height,
        credits_spent=getattr(client, "spent_this_run", 0),
        credits_remaining=getattr(client, "requests_remaining", None),
        path=path,
        raw_paths=raw_paths,
        errors=errors,
    )


def resolve_snapshot(
    frame: pl.DataFrame,
    resolver: Any,
    *,
    team_lookup: dict[str, str] | None = None,
    strict: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Attach gsis_id to a snapshot frame.

    Runs as a SEPARATE pass over already-persisted snapshots, which is what
    makes the hard-fail rule safe: raising here costs nothing, because the
    snapshot is already on disk and the pass can be re-run after adding
    overrides.

    With `strict=True` an unresolved name raises. With `strict=False` the
    unresolved names are returned and their `gsis_id` left null.
    """
    if frame.is_empty():
        return frame, []

    team_lookup = team_lookup or {}
    unresolved: list[str] = []
    mapping: dict[tuple[str, str], str] = {}

    pairs = frame.select("player_name_raw", "home_team", "away_team").unique()
    for name, home, away in pairs.iter_rows():
        teams = [team_lookup.get(t, t) for t in (home, away) if t]
        try:
            mapping[(name, home or "")] = resolver.resolve(name, teams=teams).gsis_id
        except LookupError as exc:
            unresolved.append(str(exc))

    if unresolved and strict:
        raise LookupError(
            f"{len(unresolved)} prop name(s) could not be resolved. "
            "Add rows to reference/player_name_overrides.csv:\n  " + "\n  ".join(unresolved)
        )

    resolved = frame.with_columns(
        pl.struct(["player_name_raw", "home_team"])
        .map_elements(
            lambda s: mapping.get((s["player_name_raw"], s["home_team"] or "")),
            return_dtype=pl.Utf8,
        )
        .alias("gsis_id")
    )
    return resolved, unresolved


def load_resolved_snapshots(config: Any) -> tuple[pl.DataFrame, list[str]]:
    """Every logged snapshot, with `gsis_id` filled in at read time.

    Resolution happens on read rather than being written back into the
    parquet, for two reasons. The price data is the CLV record and is
    deliberately immutable -- the store refuses to overwrite a snapshot at
    all. And resolution is a function of the resolver plus
    `reference/player_name_overrides.csv`, both of which change: adding an
    override should fix every snapshot ever logged, retroactively, without
    rewriting history.

    It is cheap. There are a few hundred distinct names in a season.

    (`nfl-props odds resolve` computes exactly this and prints a count, which
    is all it was ever meant to do. It used to be the only resolution step,
    which meant nothing downstream ever saw a gsis_id -- every consumer read
    the raw parquet, where the column is null by construction.)
    """
    from nfl_usage_props.identity import PlayerResolver
    from nfl_usage_props.reference import load_name_overrides
    from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore

    snapshots = SnapshotStore(config.props_dir).read_all()
    if snapshots.is_empty():
        return snapshots, []

    store = ParquetStore(config.raw_dir)
    if not store.exists("players", SEASONLESS_SENTINEL):
        raise FileNotFoundError("no players table; run `nfl-props ingest run` first")

    players = store.read("players", SEASONLESS_SENTINEL)
    if "latest_team" in players.columns:
        players = players.with_columns(pl.col("latest_team").alias("team"))

    team_lookup: dict[str, str] = {}
    if store.exists("teams", SEASONLESS_SENTINEL):
        teams = store.read("teams", SEASONLESS_SENTINEL)
        team_lookup = dict(zip(teams["team_name"], teams["team_abbr"], strict=False))

    resolver = PlayerResolver(players, overrides=load_name_overrides())
    return resolve_snapshot(snapshots, resolver, team_lookup=team_lookup, strict=False)
