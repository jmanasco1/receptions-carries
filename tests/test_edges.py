"""Tests for edge computation, the suppression filters, CLV and the report.

Most of the value here is in what does NOT get flagged. A filter that silently
stops working produces more bets, all of them worse, and nothing about the
output looks different.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_usage_props.edge.clv import (
    MIN_OBSERVATIONS,
    closing_line_value,
    pair_open_and_close,
    score_flagged_bets,
    summarise,
)
from nfl_usage_props.edge.edges import (
    MARKET_TO_QUANTITY,
    compute_edges,
    suppression_reasons,
)
from nfl_usage_props.model.projection import ProjectionResult
from nfl_usage_props.report import build_report, render_markdown


def make_projection(probability_over_4_5: float = 0.62, draws: int = 2000):
    """A projection whose P(receptions > 4.5) is exactly what we choose."""
    receptions = np.zeros((1, draws), dtype=int)
    receptions[0, : int(draws * probability_over_4_5)] = 6
    carries = np.zeros((1, draws), dtype=int)
    keys = pl.DataFrame(
        {
            "game_id": ["E1"],
            "team": ["AAA"],
            "gsis_id": ["00-0000001"],
            "position": ["WR"],
            "catch_rate_prior": [0.65],
        }
    )
    return ProjectionResult(keys=keys, targets=receptions, receptions=receptions, carries=carries)


def make_snapshot(over: int = -110, under: int = -110, books=("alpha", "beta", "gamma")):
    rows = []
    for book in books:
        for side, price in (("Over", over), ("Under", under)):
            rows.append(
                {
                    "event_id": "E1",
                    "market": "player_receptions",
                    "player_name_raw": "A Receiver",
                    "gsis_id": "00-0000001",
                    "point": 4.5,
                    "bookmaker": book,
                    "side": side,
                    "price": price,
                }
            )
    return pl.DataFrame(rows)


# ------------------------------------------------------------------- edges


def test_edge_is_model_minus_consensus(config):
    """A -110/-110 market devigs to 0.50, so a model at 0.62 has a 12-point
    edge. Hand-checkable on purpose."""
    edges = compute_edges(make_projection(0.62), make_snapshot(), config, week=10)
    assert edges.height == 1
    assert edges["consensus_over"][0] == pytest.approx(0.5)
    assert edges["edge"][0] == pytest.approx(0.12, abs=0.01)
    assert edges["side"][0] == "Over"


def test_the_under_is_taken_when_the_model_is_low(config):
    edges = compute_edges(make_projection(0.35), make_snapshot(), config, week=10)
    assert edges["side"][0] == "Under"
    assert edges["edge"][0] == pytest.approx(0.15, abs=0.01)


def test_edge_is_never_negative(config):
    """The sign picks the side; the magnitude is the edge. A negative edge
    would mean betting the side the model likes less."""
    for probability in (0.1, 0.35, 0.5, 0.7, 0.95):
        edges = compute_edges(make_projection(probability), make_snapshot(), config, week=10)
        assert edges["edge"][0] >= 0


def test_a_fair_model_finds_no_edge(config):
    edges = compute_edges(make_projection(0.50), make_snapshot(), config, week=10)
    assert edges["edge"][0] == pytest.approx(0.0, abs=0.01)
    assert not edges["actionable"][0]


def test_the_fair_price_matches_the_model_probability(config):
    edges = compute_edges(make_projection(0.62), make_snapshot(), config, week=10)
    assert edges["model_fair_price"][0] == pytest.approx(-100 * 0.62 / 0.38, rel=0.05)


def test_only_known_markets_are_priced():
    assert set(MARKET_TO_QUANTITY) == {"player_receptions", "player_rush_attempts"}


# ------------------------------------------------------------------ filters


def test_a_small_edge_is_suppressed(config):
    """Below the threshold, disagreement is indistinguishable from calibration
    error."""
    edges = compute_edges(make_projection(0.51), make_snapshot(), config, week=10)
    assert edges["below_threshold"][0]
    assert not edges["actionable"][0]


def test_an_extreme_price_is_suppressed(config):
    """Devigging a -400 is unstable and the limits are tiny."""
    edges = compute_edges(make_projection(0.95), make_snapshot(-400, 320), config, week=10)
    assert edges["extreme_price"][0]
    assert not edges["actionable"][0]


def test_a_thin_market_is_suppressed(config):
    edges = compute_edges(make_projection(0.70), make_snapshot(books=("alpha",)), config, week=10)
    assert edges["thin_market"][0]
    assert not edges["actionable"][0]


def test_early_weeks_are_suppressed(config):
    """Before week 4 a role estimate is mostly prior. The model will produce
    confident numbers anyway; they should not be bet."""
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=2)
    assert edges["early_season"][0]
    assert not edges["actionable"][0]

    later = compute_edges(make_projection(0.70), make_snapshot(), config, week=9)
    assert not later["early_season"][0]


def test_a_clean_market_survives_every_filter(config):
    """The complement. If this ever fails the filters have become a wall."""
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=10)
    assert edges["actionable"][0], edges.to_dicts()


def test_suppression_reasons_are_counted(config):
    edges = compute_edges(make_projection(0.51), make_snapshot(), config, week=2)
    reasons = suppression_reasons(edges)
    counted = dict(zip(reasons["reason"], reasons["rows"], strict=True))
    assert counted["below_threshold"] == 1
    assert counted["early_season"] == 1


def test_an_empty_snapshot_returns_an_empty_table(config):
    edges = compute_edges(make_projection(), pl.DataFrame(), config, week=10)
    assert edges.is_empty()


# ---------------------------------------------------------------------- CLV


def price_history(open_price: int, close_price: int) -> pl.DataFrame:
    rows = []
    for kind, price, when in (
        ("open", open_price, "2024-09-04T18:00:00Z"),
        ("close", close_price, "2024-09-08T16:00:00Z"),
    ):
        rows.append(
            {
                "event_id": "E1",
                "market": "player_receptions",
                "gsis_id": "00-0000001",
                "point": 4.5,
                "bookmaker": "alpha",
                "side": "Over",
                "price": price,
                "snapshot_kind": kind,
                "fetched_at": when,
            }
        )
    return pl.DataFrame(rows)


def test_open_and_close_are_paired():
    paired = pair_open_and_close(price_history(-110, -130))
    assert paired.height == 1
    assert paired["open_price"][0] == -110
    assert paired["close_price"][0] == -130


def test_a_line_move_is_not_a_price_move():
    """A book moving 4.5 to 5.5 replaced the market rather than repricing it.
    Treating that as CLV would manufacture edge from a different bet."""
    history = price_history(-110, -130).with_columns(
        pl.when(pl.col("snapshot_kind") == "close")
        .then(5.5)
        .otherwise(pl.col("point"))
        .alias("point")
    )
    assert pair_open_and_close(history).is_empty()


def test_clv_is_positive_when_the_market_moves_toward_the_bet():
    """Taking -110 and watching it close -130 is the definition of beating
    the close."""
    scored = closing_line_value(pair_open_and_close(price_history(-110, -130)))
    assert scored["clv"][0] > 0


def test_clv_is_negative_when_the_market_moves_away():
    scored = closing_line_value(pair_open_and_close(price_history(-130, -110)))
    assert scored["clv"][0] < 0


def test_clv_is_measured_in_probability_not_cents():
    """Twenty cents at -110 and twenty cents at -300 are very different
    amounts of edge, so cents are not comparable across price levels."""
    near = closing_line_value(pair_open_and_close(price_history(-110, -130)))["clv"][0]
    far = closing_line_value(pair_open_and_close(price_history(-300, -320)))["clv"][0]
    assert near > far


def test_a_small_sample_refuses_to_summarise():
    """The single most misleading thing this project could produce is a track
    record built from a dozen bets."""
    scored = pl.DataFrame({"clv": [0.02] * 5})
    result = summarise(scored)
    assert result["status"] != "ok"
    assert str(MIN_OBSERVATIONS) in result["status"]


def test_an_adequate_sample_summarises():
    scored = pl.DataFrame({"clv": [0.01, 0.02, -0.01] * 20})
    result = summarise(scored)
    assert result["status"] == "ok"
    assert result["n"] == 60
    assert "median_clv" in result


def test_no_scored_bets_is_reported_honestly():
    assert summarise(pl.DataFrame())["n"] == 0


def test_only_flagged_bets_are_scored(config):
    """CLV across every posted price measures the market, not the model."""
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=10)
    scored = score_flagged_bets(edges, price_history(-110, -130))
    assert scored.height == 1

    suppressed = compute_edges(make_projection(0.51), make_snapshot(), config, week=10)
    assert score_flagged_bets(suppressed, price_history(-110, -130)).is_empty()


# ------------------------------------------------------------------- report


def test_the_report_lists_flagged_rows(config):
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=10)
    report = build_report(edges)
    assert report.height == 1
    assert "edge" in report.columns


def test_the_report_hides_suppressed_rows_by_default(config):
    edges = compute_edges(make_projection(0.51), make_snapshot(), config, week=10)
    assert build_report(edges).is_empty()
    assert build_report(edges, actionable_only=False).height == 1


def test_markdown_leads_with_what_was_held_back(config):
    """A week where everything is suppressed for one reason is a broken
    pipeline, not a quiet slate, and the report has to be able to show that."""
    edges = compute_edges(make_projection(0.51), make_snapshot(), config, week=2)
    rendered = render_markdown(edges, week=2, season=2024)
    assert rendered.index("What was held back") < rendered.index("## Flagged")
    assert "below threshold" in rendered


def test_markdown_never_suggests_a_stake(config):
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=10)
    rendered = render_markdown(edges, week=10, season=2024).lower()
    for forbidden in ("units", "kelly fraction of", "stake:", "bet size"):
        assert forbidden not in rendered


def test_markdown_says_clv_is_unavailable_when_it_is(config):
    edges = compute_edges(make_projection(0.70), make_snapshot(), config, week=10)
    rendered = render_markdown(edges, week=10, season=2024, clv={"status": "no scored bets yet"})
    assert "cannot be backfilled" in rendered


def test_markdown_handles_an_empty_slate():
    rendered = render_markdown(pl.DataFrame(), week=10, season=2024)
    assert "Nothing to report" in rendered
