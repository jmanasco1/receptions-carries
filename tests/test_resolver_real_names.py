"""Resolver tests against the REAL nflverse player universe.

The odds fixture cannot validate this, because its names are hand-written and
resolve trivially. What can be validated offline is the harder half: that the
resolver copes with the actual name distribution nflverse contains -- ~25k
players, real suffix and punctuation collisions, and genuinely duplicated names.

Marked `network` because it needs the ingested players table.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl_usage_props.config import load_config
from nfl_usage_props.identity import PlayerResolver, UnresolvedPlayerError, normalize_name
from nfl_usage_props.storage import SEASONLESS_SENTINEL, ParquetStore

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def real_players() -> pl.DataFrame:
    config = load_config(load_env=False)
    store = ParquetStore(config.raw_dir)
    if not store.exists("players", SEASONLESS_SENTINEL):
        pytest.skip("players table not ingested; run `nfl-props ingest run`")
    players = store.read("players", SEASONLESS_SENTINEL)
    return players.with_columns(pl.col("latest_team").alias("team"))


@pytest.fixture(scope="module")
def resolver(real_players) -> PlayerResolver:
    return PlayerResolver(real_players)


def test_universe_is_populated(resolver):
    assert resolver.universe_size > 2000


@pytest.mark.parametrize(
    "book_name",
    [
        "Marvin Harrison Jr.",
        "Marvin Harrison",  # book drops the suffix
        "DK Metcalf",
        "D.K. Metcalf",  # book adds the dots
        "Amon-Ra St. Brown",
        "Ja'Marr Chase",
        "JaMarr Chase",  # book drops the apostrophe
        "Kenneth Walker III",
        "Kenneth Walker",
        "Brian Robinson Jr.",
        "Travis Etienne Jr.",
        "Michael Pittman Jr.",
        "CeeDee Lamb",
        "A.J. Brown",
        "AJ Brown",
        "T.J. Hockenson",
    ],
)
def test_resolves_real_book_name_variants(resolver, book_name):
    """These are the exact shapes books actually write."""
    result = resolver.resolve(book_name)
    assert result.gsis_id
    assert result.gsis_id.startswith("00-")


def test_normalization_collides_suffix_variants_in_real_data(real_players):
    """Confirm the suffix problem is real and not theoretical."""
    names = real_players["display_name"].drop_nulls().to_list()
    suffixed = [n for n in names if n.rstrip(".").endswith((" Jr", " Sr", " II", " III"))]
    assert len(suffixed) > 100, "expected many suffixed names in nflverse"
    # And that stripping the suffix is what makes them matchable.
    sample = suffixed[0]
    assert normalize_name(sample) == normalize_name(sample.rsplit(" ", 1)[0])


def test_father_son_pairs_are_not_collapsed(resolver):
    """Two real father/son pairs, and they fail differently.

    Marvin Harrison: nflverse HAS the suffix on the son, so the suffix
    separates them. Michael Pittman: nflverse has NO suffix on either, even
    though books write "Jr.", so the suffix cannot help and recency has to.
    Both must land on the active player, never the retired father.
    """
    assert resolver.resolve("Marvin Harrison Jr.").gsis_id == "00-0039849"  # WR, ARI
    assert resolver.resolve("Michael Pittman Jr.").gsis_id == "00-0036252"  # WR, not the RB


def test_suffixless_book_name_resolves_to_the_active_player(resolver):
    """Books often drop the suffix. Among a retired father and an active son,
    the active one is the one carrying a prop line."""
    assert resolver.resolve("Marvin Harrison").gsis_id == "00-0039849"
    assert resolver.resolve("Michael Pittman").gsis_id == "00-0036252"


def test_book_suffix_absent_from_nflverse_does_not_break_matching(resolver):
    """Books write "Michael Pittman Jr."; nflverse stores "Michael Pittman".
    A suffix the universe does not have must not eliminate every candidate."""
    with_suffix = resolver.resolve("Michael Pittman Jr.")
    without = resolver.resolve("Michael Pittman")
    assert with_suffix.gsis_id == without.gsis_id


def test_duplicate_names_are_real_and_mostly_resolvable_by_recency(real_players):
    """Quantifies the collision problem against real data.

    142 duplicated skill-player names exist. Recency separates all but a
    handful, and the remainder are retired players who never carry a prop.
    """
    skill = real_players.filter(pl.col("position").is_in(["WR", "RB", "TE", "QB", "FB"]))
    dupes = (
        skill.group_by("display_name")
        .agg(pl.col("last_season").alias("seasons"), pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .with_columns(
            pl.col("seasons")
            .list.eval(pl.element() == pl.element().max())
            .list.sum()
            .alias("tied_at_newest")
        )
    )
    assert dupes.height > 50, "expected many duplicate skill names"

    unresolvable = dupes.filter(pl.col("tied_at_newest") > 1)
    # The genuinely tied ones must all be historical -- an active tie would
    # mean a real prop could silently resolve to the wrong player.
    for row in unresolvable.iter_rows(named=True):
        assert max(row["seasons"]) < 2024, (
            f"{row['display_name']} is tied at last_season={max(row['seasons'])}; "
            "an active same-name collision needs an override row"
        )


def test_genuinely_tied_duplicate_still_raises(real_players):
    """When recency cannot separate two players, refuse rather than guess."""
    skill = real_players.filter(pl.col("position").is_in(["WR", "RB", "TE", "QB", "FB"]))
    tied = (
        skill.group_by("display_name")
        .agg(pl.col("last_season").alias("seasons"), pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .with_columns(
            pl.col("seasons")
            .list.eval(pl.element() == pl.element().max())
            .list.sum()
            .alias("tied_at_newest")
        )
        .filter(pl.col("tied_at_newest") > 1)
    )
    if tied.is_empty():
        pytest.skip("no tied duplicates in this players snapshot")

    resolver = PlayerResolver(real_players, skill_positions_only=True)
    with pytest.raises(UnresolvedPlayerError):
        resolver.resolve(tied["display_name"][0])


def test_unknown_name_still_raises_against_real_universe(resolver):
    """25k candidates must not produce a spurious fuzzy match."""
    with pytest.raises(UnresolvedPlayerError):
        resolver.resolve("Zebediah Quartermaine")


def test_no_false_positive_on_a_different_real_player(resolver):
    """Two genuinely different players must not collapse into one another."""
    a = resolver.resolve("Justin Jefferson")
    b = resolver.resolve("Justin Fields")
    assert a.gsis_id != b.gsis_id
