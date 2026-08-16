"""End-to-end: odds payload -> parse -> store -> resolve to gsis_id.

This is the "100% resolution against the fixture" check the spec asks for, with
an honest caveat: the fixture's player names are real, so this genuinely
exercises the resolver against the real nflverse universe, but the fixture's
*shape* is hand-built from documentation. It proves the chain works end to end.
It does not prove the book writes names the way the fixture does.

Replace tests/fixtures/event_odds.json with a real capture
(`nfl-props odds verify-markets --save-fixture`) and this test becomes a real
guarantee rather than a structural one.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.identity import PlayerResolver
from nfl_usage_props.odds.parse import add_hold, parse_event_odds, rows_to_frame
from nfl_usage_props.odds.snapshots import SnapshotStore, resolve_snapshot
from nfl_usage_props.reference import load_name_overrides
from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore
from tests.conftest import load_fixture

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def real_resolver() -> PlayerResolver:
    cfg = load_config(load_env=False)
    store = ParquetStore(cfg.raw_dir)
    if not store.exists("players", SEASONLESS_SENTINEL):
        pytest.skip("players table not ingested; run `nfl-props ingest run`")
    players = store.read("players", SEASONLESS_SENTINEL).with_columns(
        pl.col("latest_team").alias("team")
    )
    return PlayerResolver(players, overrides=load_name_overrides())


@pytest.fixture
def parsed_frame() -> pl.DataFrame:
    rows = parse_event_odds(
        load_fixture("event_odds.json"), snapshot_id="itest", snapshot_kind="open"
    )
    return add_hold(rows_to_frame(rows))


def test_every_fixture_name_resolves(parsed_frame, real_resolver):
    """No silent drops. A pipeline that quietly skips 8% of props looks like it
    works and does not."""
    resolved, unresolved = resolve_snapshot(parsed_frame, real_resolver, strict=False)
    assert unresolved == [], f"unresolved names: {unresolved}"
    assert resolved["gsis_id"].null_count() == 0


def test_resolved_ids_are_distinct_per_player(parsed_frame, real_resolver):
    """Two different players must not collapse onto one id."""
    resolved, _ = resolve_snapshot(parsed_frame, real_resolver, strict=True)
    pairs = resolved.select("player_name_raw", "gsis_id").unique()
    assert pairs["player_name_raw"].n_unique() == pairs["gsis_id"].n_unique()


def test_full_round_trip_through_the_store(parsed_frame, real_resolver, tmp_path):
    resolved, _ = resolve_snapshot(parsed_frame, real_resolver, strict=True)
    store = SnapshotStore(tmp_path / "props")
    store.write(resolved, 2024, "20240908T120000Z_open")

    reloaded = store.read_all()
    assert reloaded.height == parsed_frame.height
    assert reloaded["gsis_id"].null_count() == 0
    assert reloaded["hold"].drop_nulls().len() > 0
    # Both sides survive, which is what devig will need.
    assert set(reloaded["side"].unique()) == {"Over", "Under"}


def test_both_markets_survive_the_round_trip(parsed_frame, real_resolver):
    resolved, _ = resolve_snapshot(parsed_frame, real_resolver, strict=True)
    assert set(resolved["market"].unique()) == {
        "player_receptions",
        "player_rush_attempts",
    }
