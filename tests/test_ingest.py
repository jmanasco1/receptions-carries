"""Ingestion tests.

These use a stub loader rather than hitting nflverse, so they run offline and
in CI. The one test that touches the network is marked `network`/`slow` and is
deselected by default.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_usage_props.ingest.nflverse import (
    TABLES,
    TableSpec,
    add_kickoff_utc,
    ingest_all,
    ingest_table_season,
    is_unavailable_upstream,
    season_has_started,
    season_is_complete,
)
from nfl_usage_props.metadata import MetadataStore
from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore


def make_schedule(season: int, *, results: list[int | None], last_gameday: str) -> pl.DataFrame:
    n = len(results)
    return pl.DataFrame(
        {
            "season": [season] * n,
            "game_id": [f"{season}_{i:02d}" for i in range(n)],
            "gameday": [last_gameday] * n,
            "gametime": ["13:00"] * n,
            "result": results,
        },
        schema_overrides={"result": pl.Int32},
    )


# ----------------------------------------------------------- season completion


def test_season_complete_when_all_played_and_lag_elapsed():
    sched = make_schedule(2023, results=[3, -7, 10], last_gameday="2024-02-11")
    now = datetime(2024, 4, 1, tzinfo=UTC)
    assert season_is_complete(sched, 2023, lag_days=30, now=now)


def test_season_incomplete_while_a_game_is_unplayed():
    sched = make_schedule(2024, results=[3, None, 10], last_gameday="2025-02-09")
    now = datetime(2025, 6, 1, tzinfo=UTC)
    assert not season_is_complete(sched, 2024, lag_days=30, now=now)


def test_season_incomplete_inside_the_correction_lag():
    """All games played, but nflverse may still back-fill corrections."""
    sched = make_schedule(2023, results=[3, -7, 10], last_gameday="2024-02-11")
    now = datetime(2024, 2, 20, tzinfo=UTC)  # 9 days later, lag is 30
    assert not season_is_complete(sched, 2023, lag_days=30, now=now)


def test_season_incomplete_when_absent_from_schedule():
    sched = make_schedule(2023, results=[3], last_gameday="2024-02-11")
    assert not season_is_complete(sched, 2019, lag_days=30, now=datetime(2025, 1, 1, tzinfo=UTC))


def test_season_incomplete_for_empty_schedule():
    assert not season_is_complete(pl.DataFrame(), 2023, lag_days=30)


# ------------------------------------------------------------------- kickoff


def test_kickoff_utc_converts_eastern_to_utc():
    sched = pl.DataFrame({"gameday": ["2024-09-08"], "gametime": ["13:00"]})
    out = add_kickoff_utc(sched)
    kickoff = out["kickoff_utc"].item()
    # 13:00 EDT (UTC-4) -> 17:00 UTC
    assert kickoff.hour == 17
    assert str(kickoff.tzinfo) in {"UTC", "Etc/UTC"}


def test_kickoff_utc_handles_standard_time_offset():
    """February games are EST (UTC-5), not EDT — the offset must not be hardcoded."""
    sched = pl.DataFrame({"gameday": ["2024-02-11"], "gametime": ["18:30"]})
    kickoff = add_kickoff_utc(sched)["kickoff_utc"].item()
    assert kickoff.hour == 23  # 18:30 EST -> 23:30 UTC
    assert kickoff.minute == 30


def test_kickoff_utc_is_null_when_gametime_missing():
    """Unscheduled future games must be null, never midnight."""
    sched = pl.DataFrame({"gameday": ["2024-09-08"], "gametime": [None]})
    assert add_kickoff_utc(sched)["kickoff_utc"].item() is None


def test_kickoff_utc_column_exists_on_empty_frame():
    assert "kickoff_utc" in add_kickoff_utc(pl.DataFrame()).columns


# ------------------------------------------------------------------ idempotency


@pytest.fixture
def stub_spec():
    calls = {"n": 0}

    def loader(season: int) -> pl.DataFrame:
        calls["n"] += 1
        return pl.DataFrame({"season": [season], "call": [calls["n"]]})

    spec = TableSpec(name="stub", loader=loader, description="test double")
    return spec, calls


def test_completed_season_is_fetched_once_then_skipped(config, stub_spec):
    spec, calls = stub_spec
    store = ParquetStore(config.raw_dir)
    meta = MetadataStore(config.metadata_db)

    first = ingest_table_season(spec, 2022, store=store, meta=meta, complete=True)
    assert first.status == "fetched"
    assert calls["n"] == 1

    second = ingest_table_season(spec, 2022, store=store, meta=meta, complete=True)
    assert second.status == "skipped_complete"
    assert calls["n"] == 1, "a completed season must never be re-downloaded"


def test_incomplete_season_is_refetched_every_run(config, stub_spec):
    spec, calls = stub_spec
    store = ParquetStore(config.raw_dir)
    meta = MetadataStore(config.metadata_db)

    ingest_table_season(spec, 2024, store=store, meta=meta, complete=False)
    ingest_table_season(spec, 2024, store=store, meta=meta, complete=False)
    assert calls["n"] == 2, "an in-progress season must refresh"


def test_force_overrides_the_skip(config, stub_spec):
    spec, calls = stub_spec
    store = ParquetStore(config.raw_dir)
    meta = MetadataStore(config.metadata_db)

    ingest_table_season(spec, 2022, store=store, meta=meta, complete=True)
    result = ingest_table_season(spec, 2022, store=store, meta=meta, complete=True, force=True)
    assert result.status == "fetched"
    assert calls["n"] == 2


def test_missing_file_forces_refetch_even_when_marked_complete(config, stub_spec):
    """Metadata says complete but the parquet was deleted — must re-download."""
    spec, calls = stub_spec
    store = ParquetStore(config.raw_dir)
    meta = MetadataStore(config.metadata_db)

    ingest_table_season(spec, 2022, store=store, meta=meta, complete=True)
    store.path_for("stub", 2022).unlink()

    result = ingest_table_season(spec, 2022, store=store, meta=meta, complete=True)
    assert result.status == "fetched"
    assert calls["n"] == 2


def test_loader_failure_is_recorded_not_raised(config):
    def boom(season: int):
        raise ConnectionError("nflverse unreachable")

    spec = TableSpec(name="boom", loader=boom)
    store = ParquetStore(config.raw_dir)
    meta = MetadataStore(config.metadata_db)

    result = ingest_table_season(spec, 2022, store=store, meta=meta, complete=False)
    assert result.status == "error"
    assert "nflverse unreachable" in result.message
    assert not result.ok
    assert not store.exists("boom", 2022), "a failed fetch must not leave a file"


def test_error_does_not_mark_season_complete(config):
    def boom(season: int):
        raise ConnectionError("down")

    meta = MetadataStore(config.metadata_db)
    ingest_table_season(
        TableSpec(name="boom", loader=boom),
        2022,
        store=ParquetStore(config.raw_dir),
        meta=meta,
        complete=True,
    )
    assert not meta.is_season_complete("boom", 2022)


def test_seasonless_table_written_once_under_sentinel(config, monkeypatch):
    calls = {"n": 0}

    def loader(season: int) -> pl.DataFrame:
        calls["n"] += 1
        return pl.DataFrame({"gsis_id": ["00-0011111"]})

    monkeypatch.setitem(
        TABLES,
        "players",
        TableSpec(name="players", loader=loader, seasonal=False, post_game=False),
    )
    monkeypatch.setattr(
        "nfl_usage_props.ingest.nflverse._fetch_schedules_for",
        lambda seasons: make_schedule(2022, results=[1], last_gameday="2023-02-12"),
    )

    ingest_all(config, tables=["players"], seasons=[2022, 2023, 2024])
    assert calls["n"] == 1, "a seasonless table must not be fetched once per season"

    store = ParquetStore(config.raw_dir)
    assert store.exists("players", SEASONLESS_SENTINEL)


def test_ingest_all_rejects_unknown_table(config):
    with pytest.raises(KeyError, match="unknown table"):
        ingest_all(config, tables=["not_a_table"], seasons=[2022])


def test_ingest_all_second_run_skips_completed_seasons(config, monkeypatch):
    calls = {"n": 0}

    def loader(season: int) -> pl.DataFrame:
        calls["n"] += 1
        return pl.DataFrame({"season": [season]})

    monkeypatch.setitem(TABLES, "pbp", TableSpec(name="pbp", loader=loader))
    monkeypatch.setattr(
        "nfl_usage_props.ingest.nflverse._fetch_schedules_for",
        lambda seasons: make_schedule(2022, results=[1, 2], last_gameday="2023-02-12"),
    )
    now = datetime(2024, 1, 1, tzinfo=UTC)

    first = ingest_all(config, tables=["pbp"], seasons=[2022], now=now)
    assert [r.status for r in first] == ["fetched"]

    second = ingest_all(config, tables=["pbp"], seasons=[2022], now=now)
    assert [r.status for r in second] == ["skipped_complete"]
    assert calls["n"] == 1


def test_metadata_records_rows_and_completeness(config, stub_spec):
    spec, _ = stub_spec
    meta = MetadataStore(config.metadata_db)
    ingest_table_season(spec, 2022, store=ParquetStore(config.raw_dir), meta=meta, complete=True)
    state = meta.get_season_state("stub", 2022)
    assert state["rows"] == 1
    assert state["complete"] == 1


# ------------------------------------------------- future / unavailable seasons


def test_upstream_season_range_error_is_unavailable_not_error():
    exc = ValueError("Season must be between 1999 and 2025")
    assert is_unavailable_upstream(exc, season_started=False)
    # Even mid-season: upstream is authoritative about what it has.
    assert is_unavailable_upstream(exc, season_started=True)


def test_download_failure_before_kickoff_is_unavailable():
    exc = ConnectionError("Failed to download https://github.com/nflverse/...")
    assert is_unavailable_upstream(exc, season_started=False)


def test_download_failure_during_a_live_season_is_a_real_error():
    """An outage mid-season must not be silently reclassified as 'pending'."""
    exc = ConnectionError("Failed to download https://github.com/nflverse/...")
    assert not is_unavailable_upstream(exc, season_started=True)


def test_unrelated_exception_is_always_an_error():
    assert not is_unavailable_upstream(KeyError("bad column"), season_started=False)
    assert not is_unavailable_upstream(ValueError("something else"), season_started=False)


def test_season_has_started():
    played = make_schedule(2024, results=[3, None], last_gameday="2024-09-08")
    assert season_has_started(played, 2024)

    unplayed = make_schedule(2026, results=[None, None], last_gameday="2026-09-13")
    assert not season_has_started(unplayed, 2026)

    assert not season_has_started(played, 2019)
    assert not season_has_started(pl.DataFrame(), 2024)


def test_future_season_reports_unavailable_and_does_not_fail_the_run(config):
    """August 2026: the 2026 season exists on the schedule but has no pbp yet."""

    def not_yet(season: int):
        raise ValueError("Season must be between 1999 and 2025")

    spec = TableSpec(name="pbp", loader=not_yet)
    result = ingest_table_season(
        spec,
        2026,
        store=ParquetStore(config.raw_dir),
        meta=MetadataStore(config.metadata_db),
        complete=False,
        started_upstream=False,
    )
    assert result.status == "unavailable"
    assert result.ok, "a season that has not happened yet is not a failure"


def test_live_season_download_failure_still_reports_error(config):
    def broken(season: int):
        raise ConnectionError("Failed to download https://github.com/nflverse/...")

    result = ingest_table_season(
        TableSpec(name="pbp", loader=broken),
        2024,
        store=ParquetStore(config.raw_dir),
        meta=MetadataStore(config.metadata_db),
        complete=False,
        started_upstream=True,
    )
    assert result.status == "error"
    assert not result.ok


# -------------------------------------------------------------- table registry


def test_every_configured_table_is_registered(config):
    from nfl_usage_props.config import load_config

    real = load_config(load_env=False)
    unknown = [t for t in real.ingest.tables if t not in TABLES]
    assert unknown == [], f"config.toml lists tables with no TableSpec: {unknown}"


def test_post_game_flags_are_set_correctly():
    """These flags drive Stage 2's leakage test — a wrong one is a silent bug."""
    assert TABLES["pbp"].post_game is True
    assert TABLES["snap_counts"].post_game is True
    assert TABLES["participation"].post_game is True
    assert TABLES["player_stats"].post_game is True
    assert TABLES["injuries"].post_game is False
    assert TABLES["depth_charts"].post_game is False
    assert TABLES["schedules"].post_game is False


# ------------------------------------------------------------------- network


@pytest.mark.network
@pytest.mark.slow
def test_real_nflverse_schedules_have_expected_columns(config):
    """Guards against nflverse renaming the columns we build on."""
    import nflreadpy as nfl

    sched = add_kickoff_utc(nfl.load_schedules(seasons=[2023]))
    for col in ("game_id", "season", "week", "gameday", "gametime", "spread_line", "total_line"):
        assert col in sched.columns
    assert sched.filter(pl.col("kickoff_utc").is_null()).height == 0
    assert sched.height > 250
