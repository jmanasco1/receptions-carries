"""Parquet storage, partitioned by season.

Layout:

    data/raw/<table>/season=<YYYY>/part.parquet
    data/derived/<table>/season=<YYYY>/part.parquet
    data/odds/<endpoint>/<timestamp>.json

One file per season rather than a directory of fragments: nflverse ships one
file per season anyway, and a single file makes the "is this season already
on disk and immutable?" question trivial to answer.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Tables that are not season-partitioned (a single snapshot of current truth).
SEASONLESS_SENTINEL = 0


class ParquetStore:
    def __init__(self, root: str | Path, *, compression: str = "zstd"):
        self.root = Path(root)
        self.compression = compression

    def table_dir(self, table: str) -> Path:
        return self.root / table

    def path_for(self, table: str, season: int) -> Path:
        return self.table_dir(table) / f"season={season}" / "part.parquet"

    def exists(self, table: str, season: int) -> bool:
        path = self.path_for(table, season)
        return path.exists() and path.stat().st_size > 0

    def write(self, df: pl.DataFrame, table: str, season: int) -> Path:
        """Write atomically: temp file then rename, so an interrupted run never
        leaves a half-written parquet that later looks like a complete one."""
        path = self.path_for(table, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression=self.compression)
        tmp.replace(path)
        return path

    def read(self, table: str, season: int) -> pl.DataFrame:
        return pl.read_parquet(self.path_for(table, season))

    def read_seasons(self, table: str, seasons: list[int] | None = None) -> pl.DataFrame:
        """Read and concatenate several seasons.

        Uses `how="diagonal_relaxed"` because nflverse schemas drift across
        years (columns appear, dtypes widen). Failing loudly on that would make
        the store unusable for exactly the multi-year reads we need.
        """
        seasons = seasons if seasons is not None else self.available_seasons(table)
        frames = [self.read(table, s) for s in seasons if self.exists(table, s)]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def available_seasons(self, table: str) -> list[int]:
        table_dir = self.table_dir(table)
        if not table_dir.exists():
            return []
        seasons = []
        for child in table_dir.iterdir():
            if child.is_dir() and child.name.startswith("season="):
                try:
                    season = int(child.name.split("=", 1)[1])
                except ValueError:
                    continue
                if self.exists(table, season):
                    seasons.append(season)
        return sorted(seasons)

    def size_bytes(self, table: str, season: int) -> int:
        path = self.path_for(table, season)
        return path.stat().st_size if path.exists() else 0

    def tables(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
