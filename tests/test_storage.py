from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.storage import ParquetStore


@pytest.fixture
def store(tmp_path):
    return ParquetStore(tmp_path / "raw")


def _df(n: int = 3) -> pl.DataFrame:
    return pl.DataFrame({"game_id": [f"g{i}" for i in range(n)], "week": list(range(n))})


def test_round_trip_preserves_data(store):
    original = _df()
    store.write(original, "pbp", 2023)
    assert store.read("pbp", 2023).equals(original)


def test_exists_is_false_before_write_and_true_after(store):
    assert not store.exists("pbp", 2023)
    store.write(_df(), "pbp", 2023)
    assert store.exists("pbp", 2023)


def test_exists_is_false_for_zero_byte_file(store):
    """A truncated file must not be mistaken for a complete season."""
    path = store.path_for("pbp", 2023)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert path.exists()
    assert not store.exists("pbp", 2023)


def test_write_is_atomic_leaving_no_temp_files(store):
    store.write(_df(), "pbp", 2023)
    leftovers = list(store.path_for("pbp", 2023).parent.glob("*.tmp"))
    assert leftovers == []


def test_overwrite_replaces_rather_than_appends(store):
    store.write(_df(3), "pbp", 2023)
    store.write(_df(5), "pbp", 2023)
    assert store.read("pbp", 2023).height == 5


def test_available_seasons_sorted_and_only_written_ones(store):
    for season in (2024, 2022, 2023):
        store.write(_df(), "pbp", season)
    assert store.available_seasons("pbp") == [2022, 2023, 2024]
    assert store.available_seasons("snap_counts") == []


def test_available_seasons_ignores_non_season_directories(store):
    store.write(_df(), "pbp", 2023)
    (store.table_dir("pbp") / "scratch").mkdir()
    (store.table_dir("pbp") / "season=notanint").mkdir()
    assert store.available_seasons("pbp") == [2023]


def test_read_seasons_concatenates_in_order(store):
    store.write(pl.DataFrame({"game_id": ["a"], "week": [1]}), "pbp", 2022)
    store.write(pl.DataFrame({"game_id": ["b"], "week": [2]}), "pbp", 2023)
    combined = store.read_seasons("pbp")
    assert combined.height == 2
    assert combined["game_id"].to_list() == ["a", "b"]


def test_read_seasons_tolerates_schema_drift_across_years(store):
    """nflverse adds columns between seasons; the store must not choke."""
    store.write(pl.DataFrame({"game_id": ["a"]}), "pbp", 2022)
    store.write(pl.DataFrame({"game_id": ["b"], "new_col": [1.5]}), "pbp", 2023)
    combined = store.read_seasons("pbp")
    assert combined.height == 2
    assert "new_col" in combined.columns
    assert combined.filter(pl.col("game_id") == "a")["new_col"].item() is None


def test_read_seasons_returns_empty_frame_when_nothing_on_disk(store):
    assert store.read_seasons("pbp").is_empty()


def test_size_bytes_zero_for_missing_and_positive_after_write(store):
    assert store.size_bytes("pbp", 2023) == 0
    store.write(_df(), "pbp", 2023)
    assert store.size_bytes("pbp", 2023) > 0
