from __future__ import annotations

from nfl_usage_props.metadata import MetadataStore, utcnow_iso


def test_schema_created_on_construction(tmp_path):
    store = MetadataStore(tmp_path / "nested" / "meta.sqlite")
    assert store.path.exists()


def test_construction_is_idempotent(tmp_path):
    path = tmp_path / "meta.sqlite"
    MetadataStore(path).record_ingest(table_name="pbp", season=2023, status="fetched", rows=5)
    reopened = MetadataStore(path)  # must not wipe or error
    assert reopened.get_season_state("pbp", 2023) is None
    assert len(reopened.recent_odds_calls()) == 0


def test_season_state_upserts(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    store.set_season_state(table_name="pbp", season=2023, complete=False, rows=10)
    store.set_season_state(table_name="pbp", season=2023, complete=True, rows=50)
    state = store.get_season_state("pbp", 2023)
    assert state["complete"] == 1
    assert state["rows"] == 50


def test_is_season_complete(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    assert not store.is_season_complete("pbp", 2023)
    store.set_season_state(table_name="pbp", season=2023, complete=True)
    assert store.is_season_complete("pbp", 2023)


def test_latest_credit_balance_is_the_most_recent(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    assert store.latest_credit_balance() is None
    store.record_odds_call(endpoint="/a", requests_remaining=400)
    store.record_odds_call(endpoint="/b", requests_remaining=398)
    assert store.latest_credit_balance() == 398


def test_latest_credit_balance_skips_null_rows(tmp_path):
    """A failed call with no headers must not erase the known balance."""
    store = MetadataStore(tmp_path / "m.sqlite")
    store.record_odds_call(endpoint="/a", requests_remaining=400)
    store.record_odds_call(endpoint="/b", error="timeout")
    assert store.latest_credit_balance() == 400


def test_credits_spent_since(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    cutoff = utcnow_iso()
    store.record_odds_call(endpoint="/a", cost=2)
    store.record_odds_call(endpoint="/b", cost=32)
    assert store.credits_spent_since(cutoff) == 34
    assert store.credits_spent_since("2999-01-01T00:00:00+00:00") == 0


def test_recent_odds_calls_newest_first(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    for endpoint in ("/first", "/second", "/third"):
        store.record_odds_call(endpoint=endpoint)
    calls = store.recent_odds_calls(limit=2)
    assert [c["endpoint"] for c in calls] == ["/third", "/second"]


def test_ingest_summary_sorted(tmp_path):
    store = MetadataStore(tmp_path / "m.sqlite")
    store.set_season_state(table_name="pbp", season=2023, complete=True, rows=1)
    store.set_season_state(table_name="pbp", season=2022, complete=True, rows=1)
    store.set_season_state(table_name="injuries", season=2023, complete=False, rows=1)
    summary = store.ingest_summary()
    assert [(s["table_name"], s["season"]) for s in summary] == [
        ("injuries", 2023),
        ("pbp", 2022),
        ("pbp", 2023),
    ]
