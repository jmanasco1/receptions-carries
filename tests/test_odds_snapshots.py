"""Snapshot logging and prop parsing tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
import responses

from nfl_usage_props.odds import OddsAPIClient
from nfl_usage_props.odds.parse import (
    add_hold,
    american_to_implied,
    parse_event_odds,
    rows_to_frame,
    two_way_hold,
)
from nfl_usage_props.odds.snapshots import (
    SnapshotStore,
    events_to_snapshot,
    make_snapshot_id,
    resolve_snapshot,
    season_for,
    take_snapshot,
)
from tests.conftest import load_fixture

EVENT_ID = "e1a2b3c4d5e6f70819a2b3c4d5e6f708"
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
EVENTS_URL = f"{BASE}/events"


def credit_headers(remaining=400, used=100, last=2):
    return {
        "x-requests-remaining": str(remaining),
        "x-requests-used": str(used),
        "x-requests-last": str(last),
    }


# ---------------------------------------------------------------------- parsing


def test_parse_extracts_every_book_market_and_side():
    rows = parse_event_odds(load_fixture("event_odds.json"), snapshot_id="s1", snapshot_kind="open")
    assert len(rows) == 16  # 2 books, both markets, both sides
    assert {r.bookmaker for r in rows} == {"draftkings", "fanduel"}
    assert {r.side for r in rows} == {"Over", "Under"}
    assert {r.market for r in rows} == {"player_receptions", "player_rush_attempts"}


def test_parse_keeps_the_raw_book_name_verbatim():
    """Resolution happens later; the raw string is the durable record."""
    rows = parse_event_odds(load_fixture("event_odds.json"), snapshot_id="s1", snapshot_kind="open")
    assert "A.J. Brown" in {r.player_name_raw for r in rows}
    assert all(r.gsis_id is None for r in rows)


def test_parse_captures_line_and_price():
    rows = parse_event_odds(load_fixture("event_odds.json"), snapshot_id="s1", snapshot_kind="open")
    brown = [
        r
        for r in rows
        if r.player_name_raw == "A.J. Brown" and r.bookmaker == "draftkings" and r.side == "Over"
    ]
    assert len(brown) == 1
    assert brown[0].point == 5.5
    assert brown[0].price == -125


def test_parse_filters_to_requested_markets():
    rows = parse_event_odds(
        load_fixture("event_odds.json"),
        snapshot_id="s1",
        snapshot_kind="open",
        markets=["player_receptions"],
    )
    assert {r.market for r in rows} == {"player_receptions"}


def test_parse_skips_outcomes_without_a_player():
    """Team markets have no `description` and are not player props."""
    payload = {
        "id": "e1",
        "bookmakers": [
            {
                "key": "dk",
                "markets": [
                    {
                        "key": "player_receptions",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 4.5},  # no description
                            {
                                "name": "Over",
                                "description": "Real Player",
                                "price": -110,
                                "point": 4.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    rows = parse_event_odds(payload, snapshot_id="s", snapshot_kind="open")
    assert len(rows) == 1
    assert rows[0].player_name_raw == "Real Player"


def test_parse_empty_payload_is_empty_not_an_error():
    assert parse_event_odds({}, snapshot_id="s", snapshot_kind="open") == []


def test_rows_to_frame_keeps_schema_when_empty():
    frame = rows_to_frame([])
    assert frame.is_empty()
    assert "player_name_raw" in frame.columns
    assert "gsis_id" in frame.columns


# -------------------------------------------------------------------- vig math


@pytest.mark.parametrize(
    ("price", "expected"),
    [(-110, 0.5238), (100, 0.5), (-200, 0.6667), (150, 0.4)],
)
def test_american_to_implied(price, expected):
    assert american_to_implied(price) == pytest.approx(expected, abs=1e-3)


def test_american_to_implied_handles_none():
    assert american_to_implied(None) is None


def test_two_way_hold_on_a_standard_price():
    """-110/-110 is about 4.8% hold."""
    assert two_way_hold(-110, -110) == pytest.approx(0.0476, abs=1e-3)


def test_two_way_hold_reflects_heavy_prop_juice():
    """Receptions props run 6-10%; this must be visible, not buried."""
    hold = two_way_hold(-125, -105)
    assert 0.04 < hold < 0.12


def test_two_way_hold_needs_both_sides():
    assert two_way_hold(-110, None) is None


def test_add_hold_pairs_over_and_under():
    frame = add_hold(
        rows_to_frame(
            parse_event_odds(load_fixture("event_odds.json"), snapshot_id="s", snapshot_kind="open")
        )
    )
    brown = frame.filter(
        (pl.col("player_name_raw") == "A.J. Brown") & (pl.col("bookmaker") == "draftkings")
    )
    assert brown["hold"].drop_nulls().n_unique() == 1
    assert brown["hold"][0] == pytest.approx(two_way_hold(-125, 100), abs=1e-9)


def test_add_hold_leaves_one_sided_quotes_null_rather_than_dropping():
    rows = parse_event_odds(
        {
            "id": "e1",
            "bookmakers": [
                {
                    "key": "dk",
                    "markets": [
                        {
                            "key": "player_receptions",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Solo Guy",
                                    "price": -110,
                                    "point": 4.5,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        snapshot_id="s",
        snapshot_kind="open",
    )
    frame = add_hold(rows_to_frame(rows))
    assert frame.height == 1
    assert frame["hold"][0] is None


# ------------------------------------------------------------- event selection


def _event(hours_from_now: float, now: datetime, event_id="e") -> dict:
    return {
        "id": event_id,
        "commence_time": (now + timedelta(hours=hours_from_now)).isoformat().replace("+00:00", "Z"),
    }


def test_selects_events_inside_the_horizon():
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    events = [_event(48, now, "soon"), _event(400, now, "far")]
    selected = events_to_snapshot(events, now=now, horizon_hours=192, close_cutoff_minutes=25)
    assert [e["id"] for e in selected] == ["soon"]


def test_excludes_events_inside_the_cutoff():
    """The closing pull must never pay for a game about to kick off."""
    now = datetime(2024, 9, 8, 12, 0, tzinfo=UTC)
    events = [_event(0.1, now, "kicking_off"), _event(3, now, "later")]
    selected = events_to_snapshot(events, now=now, horizon_hours=192, close_cutoff_minutes=25)
    assert [e["id"] for e in selected] == ["later"]


def test_excludes_events_already_started():
    now = datetime(2024, 9, 8, 12, 0, tzinfo=UTC)
    selected = events_to_snapshot(
        [_event(-2, now, "in_progress")], now=now, horizon_hours=192, close_cutoff_minutes=25
    )
    assert selected == []


def test_thursday_game_is_selected_on_thursday_not_sunday():
    """The core reason closing pulls are per game day: a Sunday-morning pull
    would capture the TNF line after the game was played."""
    thursday = datetime(2024, 9, 5, 18, 0, tzinfo=UTC)
    sunday = datetime(2024, 9, 8, 13, 0, tzinfo=UTC)
    tnf = {"id": "tnf", "commence_time": "2024-09-06T00:20:00Z"}

    assert [
        e["id"]
        for e in events_to_snapshot([tnf], now=thursday, horizon_hours=192, close_cutoff_minutes=25)
    ] == ["tnf"]
    assert events_to_snapshot([tnf], now=sunday, horizon_hours=192, close_cutoff_minutes=25) == []


def test_selection_is_ordered_by_kickoff():
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    events = [_event(72, now, "late"), _event(24, now, "early")]
    selected = events_to_snapshot(events, now=now, horizon_hours=192, close_cutoff_minutes=25)
    assert [e["id"] for e in selected] == ["early", "late"]


def test_events_without_commence_time_are_skipped():
    now = datetime(2024, 9, 4, tzinfo=UTC)
    assert (
        events_to_snapshot([{"id": "x"}], now=now, horizon_hours=192, close_cutoff_minutes=25) == []
    )


# --------------------------------------------------------------- season labels


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2024, 9, 8, tzinfo=UTC), 2024),
        (datetime(2025, 1, 12, tzinfo=UTC), 2024),
        (datetime(2025, 9, 7, tzinfo=UTC), 2025),
    ],
)
def test_season_for(when, expected):
    assert season_for(when, fallback=1999) == expected


def test_season_for_falls_back_when_unknown():
    assert season_for(None, fallback=2024) == 2024


# --------------------------------------------------------------- snapshot store


def test_store_round_trip(tmp_path):
    store = SnapshotStore(tmp_path / "props")
    frame = pl.DataFrame({"player_name_raw": ["X"], "price": [-110]})
    path = store.write(frame, 2024, "20240908T120000Z_open")
    assert path.exists()
    assert store.read_all().height == 1


def test_store_refuses_to_overwrite_a_snapshot(tmp_path):
    """Prop history is unrecoverable; a clobber is not a recoverable mistake."""
    store = SnapshotStore(tmp_path / "props")
    frame = pl.DataFrame({"player_name_raw": ["X"]})
    store.write(frame, 2024, "snap1")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        store.write(frame, 2024, "snap1")


def test_store_accumulates_snapshots(tmp_path):
    store = SnapshotStore(tmp_path / "props")
    store.write(pl.DataFrame({"p": ["a"]}), 2024, "snap_open")
    store.write(pl.DataFrame({"p": ["b"]}), 2024, "snap_close")
    assert store.read_all().height == 2
    assert store.snapshot_ids(2024) == ["snap_close", "snap_open"]


def test_store_empty_when_nothing_logged(tmp_path):
    assert SnapshotStore(tmp_path / "props").read_all().is_empty()


def test_snapshot_id_encodes_kind_and_time():
    sid = make_snapshot_id("close", datetime(2024, 9, 8, 17, 30, 0, tzinfo=UTC))
    assert sid == "20240908T173000Z_close"


# ------------------------------------------------------------- end to end pull


@responses.activate
def test_take_snapshot_persists_rows(config, monkeypatch):
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    events = [{"id": EVENT_ID, "commence_time": "2024-09-09T00:20:00Z"}]
    responses.get(EVENTS_URL, json=events, headers=credit_headers(last=0))
    responses.get(
        f"{BASE}/events/{EVENT_ID}/odds",
        json=load_fixture("event_odds.json"),
        headers=credit_headers(),
    )

    client = OddsAPIClient(config, api_key="k", sleep=lambda _: None)
    result = take_snapshot(client, config, kind="open", now=now, save_raw=False)

    assert result.events_fetched == 1
    assert result.rows == 16
    assert result.path is not None
    assert result.path.exists()

    stored = SnapshotStore(config.props_dir).read_all()
    assert stored.height == 16
    assert stored["snapshot_kind"].unique().to_list() == ["open"]
    assert "hold" in stored.columns


@responses.activate
def test_take_snapshot_survives_one_bad_event(config):
    """A slate that is 1/2 logged beats a slate that raised on event 1."""
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    events = [
        {"id": "bad", "commence_time": "2024-09-09T00:20:00Z"},
        {"id": EVENT_ID, "commence_time": "2024-09-09T00:20:00Z"},
    ]
    responses.get(EVENTS_URL, json=events, headers=credit_headers(last=0))
    responses.get(f"{BASE}/events/bad/odds", status=422, json={"message": "nope"})
    responses.get(
        f"{BASE}/events/{EVENT_ID}/odds",
        json=load_fixture("event_odds.json"),
        headers=credit_headers(),
    )

    client = OddsAPIClient(config, api_key="k", sleep=lambda _: None)
    result = take_snapshot(client, config, kind="close", now=now, save_raw=False)

    assert result.events_fetched == 1
    assert len(result.errors) == 1
    assert result.rows == 16


@responses.activate
def test_take_snapshot_stops_at_the_credit_floor(config):
    """Floor abort must stop the loop, not spin through every remaining event."""
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    events = [{"id": EVENT_ID, "commence_time": "2024-09-09T00:20:00Z"} for _ in range(5)]
    events = [{**e, "id": f"{EVENT_ID}"} for e in events]
    responses.get(EVENTS_URL, json=events, headers=credit_headers(last=0))
    responses.get(
        f"{BASE}/events/{EVENT_ID}/odds",
        json=load_fixture("event_odds.json"),
        headers=credit_headers(remaining=51),
    )

    client = OddsAPIClient(config, api_key="k", sleep=lambda _: None)
    result = take_snapshot(client, config, kind="close", now=now, save_raw=False)

    assert result.events_fetched == 1
    assert any("CreditFloorError" in e for e in result.errors)


@responses.activate
def test_snapshot_does_not_require_name_resolution(config):
    """The whole point: logging must never depend on the resolver."""
    now = datetime(2024, 9, 4, 12, 0, tzinfo=UTC)
    responses.get(
        EVENTS_URL,
        json=[{"id": EVENT_ID, "commence_time": "2024-09-09T00:20:00Z"}],
        headers=credit_headers(last=0),
    )
    responses.get(
        f"{BASE}/events/{EVENT_ID}/odds",
        json=load_fixture("event_odds.json"),
        headers=credit_headers(),
    )
    client = OddsAPIClient(config, api_key="k", sleep=lambda _: None)
    result = take_snapshot(client, config, kind="open", now=now, save_raw=False)

    stored = SnapshotStore(config.props_dir).read_all()
    assert result.rows > 0
    assert stored["gsis_id"].null_count() == stored.height


def test_invalid_snapshot_kind_rejected(config):
    with pytest.raises(ValueError, match="kind must be one of"):
        take_snapshot(None, config, kind="whenever")


# ------------------------------------------------------------------ resolution


class _StubResolver:
    def __init__(self, mapping):
        self.mapping = mapping

    def resolve(self, name, teams=None, position=None):
        if name not in self.mapping:
            raise LookupError(f"unresolved: {name}")

        class R:
            gsis_id = self.mapping[name]

        return R()


def test_resolve_snapshot_attaches_gsis_ids():
    frame = pl.DataFrame(
        {
            "player_name_raw": ["A.J. Brown", "A.J. Brown"],
            "home_team": ["Philadelphia Eagles"] * 2,
            "away_team": ["Green Bay Packers"] * 2,
        }
    )
    resolved, unresolved = resolve_snapshot(
        frame, _StubResolver({"A.J. Brown": "00-0035676"}), strict=True
    )
    assert unresolved == []
    assert resolved["gsis_id"].to_list() == ["00-0035676"] * 2


def test_resolve_snapshot_strict_raises_and_names_the_file():
    frame = pl.DataFrame(
        {
            "player_name_raw": ["Unknown Guy"],
            "home_team": ["Philadelphia Eagles"],
            "away_team": ["Green Bay Packers"],
        }
    )
    with pytest.raises(LookupError, match="player_name_overrides.csv"):
        resolve_snapshot(frame, _StubResolver({}), strict=True)


def test_resolve_snapshot_non_strict_reports_without_raising():
    frame = pl.DataFrame(
        {
            "player_name_raw": ["Known", "Unknown"],
            "home_team": ["A"] * 2,
            "away_team": ["B"] * 2,
        }
    )
    resolved, unresolved = resolve_snapshot(
        frame, _StubResolver({"Known": "00-0000001"}), strict=False
    )
    assert len(unresolved) == 1
    assert resolved.filter(pl.col("gsis_id").is_null()).height == 1
