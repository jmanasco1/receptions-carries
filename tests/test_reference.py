"""Reference table tests.

The stadium table exists to separate two things that `schedules.roof` conflates:
the stadium's fixed characteristic (pre-kickoff, safe) and the observed state on
the day (post-kickoff for retractable roofs, and therefore leakage if used as a
forward feature).
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.reference import (
    is_wind_shielded,
    load_name_overrides,
    load_stadiums,
)

VALID_ROOF_TYPES = {"dome", "outdoor", "retractable"}


@pytest.fixture(scope="module")
def stadiums() -> pl.DataFrame:
    return load_stadiums()


def test_stadium_table_loads(stadiums):
    assert stadiums.height >= 40
    for column in ("stadium_id", "name", "roof_type", "latitude", "longitude"):
        assert column in stadiums.columns


def test_stadium_ids_are_unique(stadiums):
    assert stadiums["stadium_id"].n_unique() == stadiums.height


def test_roof_types_are_valid(stadiums):
    assert set(stadiums["roof_type"].unique()) <= VALID_ROOF_TYPES


def test_coordinates_are_plausible(stadiums):
    assert stadiums.filter((pl.col("latitude") < -90) | (pl.col("latitude") > 90)).is_empty()
    assert stadiums.filter((pl.col("longitude") < -180) | (pl.col("longitude") > 180)).is_empty()


def test_no_null_coordinates(stadiums):
    """A null coordinate would silently drop wind for that stadium."""
    assert stadiums["latitude"].null_count() == 0
    assert stadiums["longitude"].null_count() == 0


def test_known_roof_types_are_correct(stadiums):
    """Spot-check against roof types derived from observed nflverse data."""
    lookup = dict(zip(stadiums["stadium_id"], stadiums["roof_type"], strict=True))
    assert lookup["GNB00"] == "outdoor"  # Lambeau
    assert lookup["NOR00"] == "dome"  # Superdome
    assert lookup["DET00"] == "dome"  # Ford Field
    assert lookup["DAL00"] == "retractable"  # AT&T
    assert lookup["ATL97"] == "retractable"  # Mercedes-Benz
    assert lookup["HOU00"] == "retractable"  # NRG


# ------------------------------------------------------------- wind shielding


def test_fixed_dome_is_always_shielded():
    assert is_wind_shielded("dome", "dome")
    assert is_wind_shielded("dome", None)


def test_outdoor_is_never_shielded():
    assert not is_wind_shielded("outdoor", "outdoors")
    assert not is_wind_shielded("outdoor", None)


def test_retractable_shielded_only_when_observed_closed():
    assert is_wind_shielded("retractable", "closed")
    assert not is_wind_shielded("retractable", "open")


def test_retractable_unknown_state_is_not_shielded():
    """A forward projection does not know whether the roof will be closed.

    Returning False applies the wind forecast, which is the conservative
    direction: modelling wind that turned out to be absent is a smaller error
    than zeroing out wind that was really blowing.
    """
    assert not is_wind_shielded("retractable", None)


# ---------------------------------------------------------------- name overrides


def test_override_file_loads_even_when_empty():
    overrides = load_name_overrides()
    for column in ("book_name", "team", "gsis_id", "note"):
        assert column in overrides.columns


def test_override_file_has_no_partial_rows():
    """A row with a name but no gsis_id is silently ignored by the resolver,
    which would look like the override 'did not work'."""
    overrides = load_name_overrides()
    if overrides.is_empty():
        return
    populated = overrides.filter(pl.col("book_name").is_not_null())
    assert populated.filter(pl.col("gsis_id").is_null()).is_empty()
