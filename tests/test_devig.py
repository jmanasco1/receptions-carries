"""Tests for vig removal and consensus pricing.

Devig arithmetic is checkable by hand, so most of these pin exact numbers. The
method comparisons matter as much: the three disagree in a specific direction,
and if that direction ever flips, one of them is implemented wrong.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.edge.devig import (
    DEVIG_METHODS,
    american_to_probability,
    consensus_probability,
    devig_two_way,
    multiplicative_devig,
    power_devig,
    probability_to_american,
    shin_devig,
)

# ------------------------------------------------------------ price arithmetic


@pytest.mark.parametrize(
    ("price", "expected"),
    [(-110, 110 / 210), (100, 0.5), (-200, 2 / 3), (150, 0.4), (-300, 0.75)],
)
def test_american_to_probability(price, expected):
    assert american_to_probability(price) == pytest.approx(expected)


@pytest.mark.parametrize("price", [-400, -175, -110, 100, 145, 320])
def test_price_conversion_round_trips(price):
    assert probability_to_american(american_to_probability(price)) == pytest.approx(price)


def test_minus_one_hundred_normalises_to_plus_one_hundred():
    """The one price that cannot round-trip: -100 and +100 are the same bet,
    and the convention is to quote it +100."""
    assert probability_to_american(american_to_probability(-100)) == pytest.approx(100.0)


def test_even_money_is_plus_one_hundred():
    assert probability_to_american(0.5) == pytest.approx(100.0)


# -------------------------------------------------------------------- devig


@pytest.mark.parametrize("method", sorted(DEVIG_METHODS))
def test_every_method_returns_a_distribution(method):
    over, under = DEVIG_METHODS[method](american_to_probability(-125), american_to_probability(105))
    assert over + under == pytest.approx(1.0)
    assert 0 < over < 1


@pytest.mark.parametrize("method", sorted(DEVIG_METHODS))
def test_a_symmetric_price_devigs_to_a_coin_flip(method):
    over, under = DEVIG_METHODS[method](
        american_to_probability(-110), american_to_probability(-110)
    )
    assert over == pytest.approx(0.5)
    assert under == pytest.approx(0.5)


@pytest.mark.parametrize("method", sorted(DEVIG_METHODS))
def test_a_fair_price_is_left_alone(method):
    """No hold to remove means nothing to do."""
    over, under = DEVIG_METHODS[method](0.4, 0.6)
    assert over == pytest.approx(0.4)
    assert under == pytest.approx(0.6)


def test_power_takes_more_margin_off_the_longshot_than_multiplicative():
    """The documented difference between the two, and the reason power is the
    default. If this ever flips, one of them is wrong."""
    over_raw = american_to_probability(-250)
    under_raw = american_to_probability(200)
    power_over, _ = power_devig(over_raw, under_raw)
    mult_over, _ = multiplicative_devig(over_raw, under_raw)
    assert power_over > mult_over


def test_shin_sits_between_the_other_two_on_a_lopsided_price():
    over_raw = american_to_probability(-250)
    under_raw = american_to_probability(200)
    mult, _ = multiplicative_devig(over_raw, under_raw)
    power, _ = power_devig(over_raw, under_raw)
    shin, _ = shin_devig(over_raw, under_raw)
    assert mult < shin < power


def test_hold_is_reported_and_plausible():
    result = devig_two_way(-110, -110)
    assert result.hold == pytest.approx(0.0476, abs=0.001)


def test_method_spread_widens_with_asymmetry():
    """A near-symmetric price is unambiguous; a lopsided one is not. The spread
    is the warning signal the edge filter uses."""
    assert devig_two_way(-110, -110).spread() < devig_two_way(-300, 240).spread()


def test_degenerate_prices_fall_back_rather_than_raise():
    """One odd price must not take down a whole slate's pricing."""
    over, under = power_devig(0.999999, 0.999999)
    assert over + under == pytest.approx(1.0)


def test_implausible_prices_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        power_devig(0.0, 0.5)


# ---------------------------------------------------------------- consensus


def snapshot(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "event_id": "E1",
        "market": "player_receptions",
        "player_name_raw": "A Receiver",
        "gsis_id": "00-0000001",
        "point": 4.5,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


@pytest.fixture
def two_books() -> pl.DataFrame:
    return snapshot(
        [
            {"bookmaker": "alpha", "side": "Over", "price": -110},
            {"bookmaker": "alpha", "side": "Under", "price": -110},
            {"bookmaker": "beta", "side": "Over", "price": -130},
            {"bookmaker": "beta", "side": "Under", "price": 110},
        ]
    )


def test_consensus_averages_across_books(two_books):
    result = consensus_probability(two_books, min_books=1)
    assert result.height == 1
    assert 0.5 < result["consensus_over"][0] < 0.57
    assert result["n_books"][0] == 2


def test_weights_shift_the_consensus(two_books):
    flat = consensus_probability(two_books, min_books=1)["consensus_over"][0]
    tilted = consensus_probability(two_books, weights={"beta": 10.0}, min_books=1)[
        "consensus_over"
    ][0]
    assert tilted > flat


def test_a_one_sided_price_is_dropped():
    """A single side cannot be devigged, and inventing the other half would
    manufacture the number being estimated."""
    frame = snapshot([{"bookmaker": "alpha", "side": "Over", "price": -110}])
    assert consensus_probability(frame, min_books=1).is_empty()


def test_thin_markets_are_flagged_not_dropped(two_books):
    """Two books is not a consensus, but a thin market is exactly where the
    biggest edges and the biggest mistakes both live."""
    result = consensus_probability(two_books, min_books=5)
    assert result.height == 1
    assert result["thin_market"][0]


def test_over_and_under_consensus_sum_to_one(two_books):
    result = consensus_probability(two_books, min_books=1)
    assert result["consensus_over"][0] + result["consensus_under"][0] == pytest.approx(1.0)


def test_different_lines_are_different_markets():
    """Over 4.5 and Over 5.5 are not the same bet, and averaging them would
    produce a price for a line nobody posted."""
    frame = pl.concat(
        [
            snapshot(
                [
                    {"bookmaker": "alpha", "side": "Over", "price": -110},
                    {"bookmaker": "alpha", "side": "Under", "price": -110},
                ]
            ),
            snapshot(
                [
                    {"bookmaker": "beta", "side": "Over", "price": 130, "point": 5.5},
                    {"bookmaker": "beta", "side": "Under", "price": -160, "point": 5.5},
                ]
            ),
        ]
    )
    assert consensus_probability(frame, min_books=1).height == 2


def test_grouping_prefers_the_resolved_id_over_the_raw_name():
    """Two books spelling one player differently must not become two thin
    markets once the snapshot has been resolved."""
    frame = pl.concat(
        [
            snapshot([{"bookmaker": "alpha", "side": "Over", "price": -110}]),
            snapshot([{"bookmaker": "alpha", "side": "Under", "price": -110}]),
            snapshot(
                [
                    {
                        "bookmaker": "beta",
                        "side": "Over",
                        "price": -115,
                        "player_name_raw": "A. Receiver",
                    }
                ]
            ),
            snapshot(
                [
                    {
                        "bookmaker": "beta",
                        "side": "Under",
                        "price": -105,
                        "player_name_raw": "A. Receiver",
                    }
                ]
            ),
        ]
    )
    result = consensus_probability(frame, min_books=1)
    assert result.height == 1
    assert result["n_books"][0] == 2


def test_an_unknown_method_is_rejected(two_books):
    with pytest.raises(ValueError, match="unknown devig method"):
        consensus_probability(two_books, method="vegas_magic")
