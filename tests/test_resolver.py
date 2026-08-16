"""Player identity resolution tests.

The normalizer is tested against the name shapes that actually break props
pipelines — suffixes, initials, punctuation, accents — rather than against the
tidy names in the odds fixture, which resolve trivially and prove nothing.

See `test_resolver_real_names.py` for the same logic exercised against the real
nflverse player universe.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.identity import (
    PlayerResolver,
    UnresolvedPlayerError,
    normalize_name,
)


@pytest.fixture
def players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": [
                "00-0000001",
                "00-0000002",
                "00-0000003",
                "00-0000004",
                "00-0000005",
                "00-0000006",
                "00-0000007",
            ],
            "display_name": [
                "Marvin Harrison Jr.",
                "DK Metcalf",
                "Kenneth Walker III",
                "Josh Allen",  # QB
                "Josh Allen",  # edge rusher, different team
                "Amon-Ra St. Brown",
                "Michael Pittman Jr.",
            ],
            "position": ["WR", "WR", "RB", "QB", "LB", "WR", "WR"],
            "team": ["ARI", "PIT", "SEA", "BUF", "JAX", "DET", "IND"],
        }
    )


@pytest.fixture
def resolver(players) -> PlayerResolver:
    return PlayerResolver(players, skill_positions_only=False)


# ------------------------------------------------------------------ normalizer


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Marvin Harrison", "marvin harrison"),
        ("MARVIN HARRISON JR", "marvin harrison"),
        ("D.K. Metcalf", "dk metcalf"),
        ("DK Metcalf", "dk metcalf"),
        ("Kenneth Walker III", "kenneth walker"),
        ("Kenneth Walker", "kenneth walker"),
        ("Amon-Ra St. Brown", "amon ra st brown"),
        ("Michael Pittman Jr.", "michael pittman"),
        ("  Extra   Spaces  ", "extra spaces"),
        ("", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_suffix_variants_collapse_to_one_key():
    """The Jr./III mismatch is the single most common books-vs-nflverse break."""
    assert normalize_name("Marvin Harrison Jr.") == normalize_name("Marvin Harrison")
    assert normalize_name("Kenneth Walker III") == normalize_name("Kenneth Walker")


def test_punctuation_variants_collapse():
    assert normalize_name("D.K. Metcalf") == normalize_name("DK Metcalf")


def test_accents_are_folded():
    assert normalize_name("Ké'Shawn Vaughn") == normalize_name("KeShawn Vaughn")


# -------------------------------------------------------------------- matching


def test_resolves_exact_name(resolver):
    assert resolver.resolve("DK Metcalf", teams=["PIT"]).gsis_id == "00-0000002"


def test_resolves_across_suffix_mismatch(resolver):
    """Book omits the Jr., nflverse has it."""
    result = resolver.resolve("Marvin Harrison", teams=["ARI"])
    assert result.gsis_id == "00-0000001"
    assert result.method == "exact"


def test_resolves_across_punctuation_mismatch(resolver):
    assert resolver.resolve("D.K. Metcalf", teams=["PIT"]).gsis_id == "00-0000002"


def test_resolves_first_initial_form(resolver):
    result = resolver.resolve("M. Harrison", teams=["ARI"])
    assert result.gsis_id == "00-0000001"
    assert result.method == "initial"


def test_team_scoping_disambiguates_duplicate_names(resolver):
    """Two Josh Allens. Event scoping picks the right one structurally."""
    assert resolver.resolve("Josh Allen", teams=["BUF"]).gsis_id == "00-0000001"[:-1] + "4"
    assert resolver.resolve("Josh Allen", teams=["JAX"]).gsis_id == "00-0000005"


def test_ambiguous_duplicate_raises_rather_than_guessing(resolver):
    """Both Josh Allens in scope. A wrong id silently misattributes props to
    another player, which is worse than a loud failure."""
    with pytest.raises(UnresolvedPlayerError) as exc:
        resolver.resolve("Josh Allen", teams=["BUF", "JAX"])
    assert "Josh Allen" in str(exc.value)


def test_position_filter_disambiguates(resolver):
    result = resolver.resolve("Josh Allen", teams=["BUF", "JAX"], position="QB")
    assert result.gsis_id == "00-0000004"


# ---------------------------------------------------------------- never drops


def test_unknown_name_raises(resolver):
    with pytest.raises(UnresolvedPlayerError):
        resolver.resolve("Nonexistent Person", teams=["ARI"])


def test_error_message_is_pasteable_into_the_override_file(resolver):
    with pytest.raises(UnresolvedPlayerError) as exc:
        resolver.resolve("Totally Unknown Guy", teams=["ARI"])
    message = str(exc.value)
    assert "player_name_overrides.csv" in message
    assert "Totally Unknown Guy" in message


def test_fuzzy_absorbs_a_book_typo(resolver):
    """A one-or-two character misspelling should still resolve."""
    result = resolver.resolve("Marvin Harrisson", teams=["ARI"])
    assert result.gsis_id == "00-0000001"
    assert result.method == "fuzzy"


def test_error_lists_closest_candidates(resolver):
    with pytest.raises(UnresolvedPlayerError) as exc:
        resolver.resolve("Zebediah Quartermaine", teams=["ARI"])
    assert exc.value.candidates


def test_empty_name_raises(resolver):
    with pytest.raises(UnresolvedPlayerError):
        resolver.resolve("", teams=["ARI"])


def test_resolve_many_collects_failures_without_raising(resolver):
    resolved, failures = resolver.resolve_many(["DK Metcalf", "Nobody At All"], teams=["PIT"])
    assert len(resolved) == 1
    assert len(failures) == 1
    assert failures[0].book_name == "Nobody At All"


# --------------------------------------------------------------- overrides win


def test_override_beats_matching(players):
    """An override is never second-guessed, even against an exact match."""
    overrides = pl.DataFrame(
        {
            "book_name": ["DK Metcalf"],
            "team": ["PIT"],
            "gsis_id": ["00-0009999"],
            "note": ["forced"],
        }
    )
    resolver = PlayerResolver(players, overrides=overrides, skill_positions_only=False)
    result = resolver.resolve("DK Metcalf", teams=["PIT"])
    assert result.gsis_id == "00-0009999"
    assert result.method == "override"


def test_override_resolves_otherwise_unmappable_name(players):
    overrides = pl.DataFrame(
        {
            "book_name": ["Hollywood Brown"],
            "team": [None],
            "gsis_id": ["00-0000001"],
            "note": ["nickname"],
        }
    )
    resolver = PlayerResolver(players, overrides=overrides, skill_positions_only=False)
    assert resolver.resolve("Hollywood Brown", teams=["ARI"]).gsis_id == "00-0000001"


def test_override_matching_is_normalized(players):
    """You should not have to match the book's punctuation exactly."""
    overrides = pl.DataFrame(
        {
            "book_name": ["d.k. metcalf"],
            "team": [None],
            "gsis_id": ["00-0009999"],
            "note": [""],
        }
    )
    resolver = PlayerResolver(players, overrides=overrides, skill_positions_only=False)
    assert resolver.resolve("DK Metcalf").gsis_id == "00-0009999"


def test_empty_override_file_is_fine(players):
    empty = pl.DataFrame({"book_name": [], "team": [], "gsis_id": [], "note": []})
    resolver = PlayerResolver(players, overrides=empty, skill_positions_only=False)
    assert resolver.resolve("DK Metcalf", teams=["PIT"]).gsis_id == "00-0000002"


# ------------------------------------------------------------------- scoping


def test_skill_position_filter_shrinks_the_universe(players):
    skill_only = PlayerResolver(players, skill_positions_only=True)
    everyone = PlayerResolver(players, skill_positions_only=False)
    assert skill_only.universe_size < everyone.universe_size


def test_traded_player_still_resolves_via_full_universe_fallback(resolver):
    """Roster snapshot has him on his old team; the prop is for his new one."""
    result = resolver.resolve("DK Metcalf", teams=["SEA"])
    assert result.gsis_id == "00-0000002"
