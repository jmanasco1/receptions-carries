"""Load the raw store into a panel and a feature matrix.

The seam between "parquet on disk" and "frames the model understands". Kept
separate from `panel` and `build` so those two stay pure functions of their
inputs and remain testable without a populated data directory.
"""

from __future__ import annotations

import polars as pl

from nfl_usage_props.config import Config
from nfl_usage_props.features.build import build_features
from nfl_usage_props.features.panel import PBP_COLUMNS, build_panel
from nfl_usage_props.reference import normalize_team
from nfl_usage_props.storage import ParquetStore

PANEL_TABLE = "panel"
FEATURES_TABLE = "features"


def load_panel(config: Config, seasons: list[int] | None = None) -> pl.DataFrame:
    """Build the player-game panel from the ingested raw tables."""
    store = ParquetStore(config.raw_dir, compression=config.storage.compression)
    seasons = seasons or store.available_seasons("pbp")
    if not seasons:
        raise FileNotFoundError("no ingested pbp seasons found; run `nfl-props ingest run` first")

    pbp = _read(store, "pbp", seasons, columns=PBP_COLUMNS)
    participation = _read(store, "participation", seasons)
    schedules = load_schedules(config, seasons)

    positions = None
    if store.available_seasons("player_stats"):
        positions = (
            _read(store, "player_stats", seasons, columns=("player_id", "position", "season"))
            .sort("season", descending=True)
            .unique(subset="player_id", keep="first")
            .select(pl.col("player_id").alias("gsis_id"), "position")
        )

    panel = build_panel(pbp, participation, positions=positions)
    # Kickoff time is what orders the tracker. Week numbers do not: a Thursday
    # game in week 5 precedes the Monday game of week 4.
    return panel.join(schedules.select("game_id", "kickoff_utc"), on="game_id", how="left")


def load_schedules(config: Config, seasons: list[int] | None = None) -> pl.DataFrame:
    """Schedules with team abbreviations normalised to match pbp."""
    store = ParquetStore(config.raw_dir, compression=config.storage.compression)
    seasons = seasons or store.available_seasons("schedules")
    return _read(store, "schedules", seasons).with_columns(
        normalize_team("home_team").alias("home_team"),
        normalize_team("away_team").alias("away_team"),
    )


def build_dataset(
    config: Config, seasons: list[int] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Panel and feature matrix together, since features need the panel anyway."""
    panel = load_panel(config, seasons)
    schedules = load_schedules(config, seasons)
    return panel, build_features(panel, schedules, config)


def write_dataset(config: Config, panel: pl.DataFrame, features: pl.DataFrame) -> list[str]:
    """Persist both frames to the derived store, partitioned by season."""
    store = ParquetStore(config.derived_dir, compression=config.storage.compression)
    written = []
    for name, frame in ((PANEL_TABLE, panel), (FEATURES_TABLE, features)):
        for season in sorted(frame["season"].unique().to_list()):
            store.write(frame.filter(pl.col("season") == season), name, season)
            written.append(f"{name}/season={season}")
    return written


def _read(
    store: ParquetStore, table: str, seasons: list[int], columns: tuple[str, ...] | None = None
) -> pl.DataFrame:
    frames = []
    for season in seasons:
        if not store.exists(table, season):
            continue
        frame = store.read(table, season)
        if columns:
            # A column can be absent in some seasons (nflverse schemas drift),
            # and silently dropping the season would be worse than a short frame.
            frame = frame.select([c for c in columns if c in frame.columns])
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
