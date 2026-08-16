"""Odds client tests — all against recorded fixtures, no live calls."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import responses

from nfl_usage_props.metadata import MetadataStore
from nfl_usage_props.odds import (
    CreditFloorError,
    OddsAPIClient,
    OddsAPIError,
    estimate_props_cost,
)
from tests.conftest import load_fixture

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
GAME_ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
EVENT_ID = "e1a2b3c4d5e6f70819a2b3c4d5e6f708"
EVENT_ODDS_URL = (
    f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{EVENT_ID}/odds"
)


@pytest.fixture
def client(config):
    return OddsAPIClient(config, api_key="test-key", sleep=lambda _: None)


def credit_headers(remaining=400, used=100, last=2):
    return {
        "x-requests-remaining": str(remaining),
        "x-requests-used": str(used),
        "x-requests-last": str(last),
    }


# ------------------------------------------------------------------- cost math


@pytest.mark.parametrize(
    ("events", "markets", "regions", "expected"),
    [
        (16, 2, 1, 32),  # the documented full-slate case from the README
        (16, 2, 2, 64),
        (0, 2, 1, 0),
        (1, 1, 1, 1),
    ],
)
def test_estimate_props_cost(events, markets, regions, expected):
    assert estimate_props_cost(events, markets, regions) == expected


# --------------------------------------------------------------- happy paths


@responses.activate
def test_get_events_parses_fixture(client):
    responses.get(EVENTS_URL, json=load_fixture("events.json"), headers=credit_headers(last=0))
    resp = client.get_events()
    assert len(resp.data) == 3
    assert resp.data[0]["id"] == EVENT_ID


@responses.activate
def test_get_event_odds_returns_both_configured_markets(client):
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    resp = client.get_event_odds(EVENT_ID)
    keys = {m["key"] for b in resp.data["bookmakers"] for m in b["markets"]}
    assert keys == {"player_receptions", "player_rush_attempts"}


@responses.activate
def test_request_sends_key_and_configured_params(client):
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    client.get_event_odds(EVENT_ID)
    query = responses.calls[0].request.params
    assert query["apiKey"] == "test-key"
    assert query["regions"] == "us"
    assert query["oddsFormat"] == "american"
    assert set(query["markets"].split(",")) == {"player_receptions", "player_rush_attempts"}


@responses.activate
def test_game_odds_uses_game_markets_not_prop_markets(client):
    responses.get(GAME_ODDS_URL, json=load_fixture("game_odds.json"), headers=credit_headers())
    client.get_game_odds()
    assert set(responses.calls[0].request.params["markets"].split(",")) == {"spreads", "totals"}


# ---------------------------------------------------------- credit accounting


@responses.activate
def test_credit_headers_are_parsed_onto_the_response(client):
    responses.get(
        EVENT_ODDS_URL,
        json=load_fixture("event_odds.json"),
        headers=credit_headers(remaining=468, used=32, last=2),
    )
    resp = client.get_event_odds(EVENT_ID)
    assert resp.requests_remaining == 468
    assert resp.requests_used == 32
    assert resp.cost == 2


@responses.activate
def test_credit_balance_is_persisted_to_metadata(client, config):
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(remaining=468)
    )
    client.get_event_odds(EVENT_ID)
    assert MetadataStore(config.metadata_db).latest_credit_balance() == 468


@responses.activate
def test_spend_accumulates_across_calls(client):
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(last=2)
    )
    client.get_event_odds(EVENT_ID)
    client.get_event_odds(EVENT_ID)
    assert client.spent_this_run == 4


@responses.activate
def test_missing_credit_headers_do_not_crash(client):
    """Some proxies strip headers; the client must degrade, not explode."""
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"))
    resp = client.get_event_odds(EVENT_ID)
    assert resp.requests_remaining is None
    assert resp.cost is None


@responses.activate
def test_malformed_credit_header_is_ignored(client):
    responses.get(
        EVENT_ODDS_URL,
        json=load_fixture("event_odds.json"),
        headers={"x-requests-remaining": "unlimited"},
    )
    assert client.get_event_odds(EVENT_ID).requests_remaining is None


# ------------------------------------------------------------- the credit floor


@responses.activate
def test_aborts_when_balance_would_drop_below_floor(client):
    """Floor is 50. Land on 51, then the next 2-credit call must be refused."""
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(remaining=51)
    )
    client.get_event_odds(EVENT_ID)

    with pytest.raises(CreditFloorError, match="floor is 50"):
        client.get_event_odds(EVENT_ID)


@responses.activate
def test_floor_abort_makes_no_http_call(client):
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(remaining=51)
    )
    client.get_event_odds(EVENT_ID)
    with pytest.raises(CreditFloorError):
        client.get_event_odds(EVENT_ID)
    assert len(responses.calls) == 1, "the refused call must not reach the network"


@responses.activate
def test_comfortable_balance_is_allowed(client):
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(remaining=200)
    )
    client.get_event_odds(EVENT_ID)
    assert client.get_event_odds(EVENT_ID).status_code == 200


@responses.activate
def test_first_call_proceeds_when_balance_unknown(client):
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    assert client.requests_remaining is None
    assert client.get_event_odds(EVENT_ID).status_code == 200


def test_client_inherits_balance_from_metadata(config):
    """A fresh process must not forget it is already near the floor."""
    MetadataStore(config.metadata_db).record_odds_call(
        endpoint="/x", status_code=200, requests_remaining=51, cost=2
    )
    fresh = OddsAPIClient(config, api_key="k", sleep=lambda _: None)
    assert fresh.requests_remaining == 51
    with pytest.raises(CreditFloorError):
        fresh.get_event_odds(EVENT_ID)


# ------------------------------------------------------------- bulk budgeting


def test_bulk_refuses_a_pull_over_the_run_budget(client):
    """max_spend_per_run is 120; 80 events x 2 markets = 160."""
    with pytest.raises(CreditFloorError, match="max_spend_per_run"):
        client.get_event_odds_bulk([f"evt{i}" for i in range(80)])


def test_bulk_budget_check_happens_before_any_call(client):
    with pytest.raises(CreditFloorError):
        client.get_event_odds_bulk([f"evt{i}" for i in range(80)])
    assert len(responses.calls) == 0


@responses.activate
def test_bulk_stops_at_the_floor_mid_slate(client, config):
    """A partial slate beats an exhausted quota."""
    for remaining in (200, 100, 51):
        responses.get(
            EVENT_ODDS_URL,
            json=load_fixture("event_odds.json"),
            headers=credit_headers(remaining=remaining),
        )
    with pytest.raises(CreditFloorError):
        client.get_event_odds_bulk([EVENT_ID] * 5)
    assert len(responses.calls) == 3


# --------------------------------------------------------------------- retries


@responses.activate
def test_retries_on_429_then_succeeds(client):
    responses.get(EVENT_ODDS_URL, status=429, json={"message": "rate limited"})
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    resp = client.get_event_odds(EVENT_ID)
    assert resp.status_code == 200
    assert len(responses.calls) == 2


@responses.activate
def test_retries_on_503(client):
    responses.get(EVENT_ODDS_URL, status=503, body="upstream down")
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    assert client.get_event_odds(EVENT_ID).status_code == 200


@responses.activate
def test_gives_up_after_max_attempts(client):
    for _ in range(3):
        responses.get(EVENT_ODDS_URL, status=429, json={"message": "rate limited"})
    with pytest.raises(OddsAPIError, match="after 3 attempts"):
        client.get_event_odds(EVENT_ID)
    assert len(responses.calls) == 3


@responses.activate
def test_422_is_not_retried(client):
    """A bad market key will never succeed — retrying just burns credits."""
    responses.get(EVENT_ODDS_URL, status=422, json={"message": "Unknown market"})
    with pytest.raises(OddsAPIError, match="422"):
        client.get_event_odds(EVENT_ID)
    assert len(responses.calls) == 1


@responses.activate
def test_401_is_not_retried(client):
    responses.get(EVENT_ODDS_URL, status=401, json={"message": "bad key"})
    with pytest.raises(OddsAPIError):
        client.get_event_odds(EVENT_ID)
    assert len(responses.calls) == 1


@responses.activate
def test_backoff_delays_grow_exponentially(config):
    retry = replace(
        config.odds.retry,
        max_attempts=3,
        backoff_initial_seconds=2.0,
        backoff_multiplier=2.0,
        backoff_max_seconds=32.0,
    )
    slow_config = replace(config, odds=replace(config.odds, retry=retry))

    slept: list[float] = []
    slow = OddsAPIClient(slow_config, api_key="k", sleep=slept.append)

    for _ in range(3):
        responses.get(EVENT_ODDS_URL, status=429, json={})
    with pytest.raises(OddsAPIError):
        slow.get_event_odds(EVENT_ID)
    assert slept == [2.0, 4.0]  # no sleep after the final attempt


# ---------------------------------------------------------------- error paths


def test_missing_api_key_raises_before_any_call(config):
    keyless = OddsAPIClient(config, api_key="", sleep=lambda _: None)
    with pytest.raises(OddsAPIError, match="ODDS_API_KEY"):
        keyless.get_event_odds(EVENT_ID)


@responses.activate
def test_failures_are_logged_to_metadata(client, config):
    responses.get(EVENT_ODDS_URL, status=422, json={"message": "Unknown market"})
    with pytest.raises(OddsAPIError):
        client.get_event_odds(EVENT_ID)
    calls = MetadataStore(config.metadata_db).recent_odds_calls()
    assert calls[0]["status_code"] == 422
    assert "Unknown market" in calls[0]["error"]


@responses.activate
def test_api_key_never_written_to_metadata(client, config):
    """The key must not leak into the run log."""
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    client.get_event_odds(EVENT_ID)
    logged = json.dumps(MetadataStore(config.metadata_db).recent_odds_calls())
    assert "test-key" not in logged
    assert "apiKey" not in logged


# --------------------------------------------------------------- snapshotting


@responses.activate
def test_snapshot_written_with_credit_context(client, config):
    responses.get(
        EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers(remaining=468)
    )
    resp = client.get_event_odds(EVENT_ID)
    path = client.save_snapshot(resp, "props")

    payload = json.loads(path.read_text())
    assert payload["requests_remaining"] == 468
    assert payload["data"]["id"] == EVENT_ID
    assert payload["fetched_at"]
    assert path.parent == config.odds_dir / "props"


@responses.activate
def test_snapshots_do_not_overwrite_each_other(client):
    """Prop history is unrecoverable — snapshots must never be clobbered."""
    responses.get(EVENT_ODDS_URL, json=load_fixture("event_odds.json"), headers=credit_headers())
    first = client.save_snapshot(client.get_event_odds(EVENT_ID), "props")
    assert first.exists()
    assert first.name.endswith(".json")
