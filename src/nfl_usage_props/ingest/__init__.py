"""nflverse ingestion."""

from nfl_usage_props.ingest.nflverse import (
    TABLES,
    IngestResult,
    TableSpec,
    ingest_all,
    ingest_table_season,
    season_is_complete,
)

__all__ = [
    "TABLES",
    "IngestResult",
    "TableSpec",
    "ingest_all",
    "ingest_table_season",
    "season_is_complete",
]
