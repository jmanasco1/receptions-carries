"""Resolve sportsbook player-name strings to nflverse gsis_ids.

The Odds API returns props keyed by book-written name strings; nflverse keys on
gsis_id. These do not match, and the mismatch is the classic silent killer of
props pipelines: a resolver that quietly drops what it cannot match looks like
it works while losing 8% of the slate, and the players it loses are not random
-- they are the rookies, the suffixed names and the just-traded players, which
is disproportionately where role change (and therefore edge) lives.

Real cases this has to survive:

    "Marvin Harrison Jr."  vs  "Marvin Harrison"
    "D.K. Metcalf"         vs  "DK Metcalf"
    "Kenneth Walker III"   vs  "Kenneth Walker"
    "Josh Allen"           -- a QB and an edge rusher, simultaneously

Design
------
1. **Scope candidates to the event.** Props come attached to a specific game,
   so the candidate pool is offensive skill players on those two teams -- a few
   dozen people, not 2,000. This resolves the "Josh Allen" collision
   structurally rather than statistically, and it is why fuzzy matching is a
   last resort here rather than the main mechanism.
2. **Manual overrides win outright**, before any matching runs.
3. **Exact normalized match**, then last-name + first-initial, then fuzzy above
   a threshold.
4. **Never drop.** An unresolved name raises `UnresolvedPlayerError`.

The hard-fail rule applies to the RESOLUTION path only. Snapshot logging stores
raw book strings and must never depend on this module succeeding -- see
`odds/snapshots.py`. Losing a week of closing lines to a resolver exception
would defeat the entire point of logging them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import polars as pl

# Generational suffixes books add or omit inconsistently.
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "vi"}

# Positions that can plausibly carry a reception or rush-attempt prop.
SKILL_POSITIONS = frozenset({"WR", "RB", "TE", "QB", "FB"})

DEFAULT_FUZZY_THRESHOLD = 0.88


class UnresolvedPlayerError(LookupError):
    """A book name could not be mapped to a gsis_id.

    Carries the raw name and team so the message is directly pasteable into
    reference/player_name_overrides.csv.
    """

    def __init__(self, book_name: str, team: str | None, candidates: list[str] | None = None):
        self.book_name = book_name
        self.team = team
        self.candidates = candidates or []
        hint = f" Closest candidates: {', '.join(self.candidates[:5])}." if self.candidates else ""
        super().__init__(
            f"could not resolve {book_name!r} (team={team or 'unknown'}) to a gsis_id."
            f"{hint} Add a row to reference/player_name_overrides.csv: "
            f"{book_name},{team or ''},<gsis_id>,<why>"
        )


@dataclass(frozen=True)
class Resolution:
    book_name: str
    gsis_id: str
    matched_name: str
    team: str | None
    method: str  # override | exact | initial | fuzzy
    score: float


def split_suffix(name: str) -> tuple[str, str | None]:
    """Fold a name to a comparable key, returning the suffix separately.

    The suffix has to come back out rather than simply being discarded, because
    nflverse contains real father/son pairs where the suffix is the ONLY thing
    telling them apart:

        Marvin Harrison   (WR, retired)  vs  Marvin Harrison Jr.  (WR, ARI)
        Michael Pittman   (RB, retired)  vs  Michael Pittman Jr.  (WR, IND)

    Books are inconsistent about writing it, so the base name is still the
    match key -- but when the base is ambiguous the suffix breaks the tie.
    Discarding it outright would silently resolve a 2024 rookie's props to his
    father.
    """
    if not name:
        return "", None
    # Decompose accents: "Ké'Shawn" -> "Ke'Shawn" -> "keshawn"
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower()
    # Drop punctuation entirely rather than mapping to space, so "d.k." -> "dk"
    # rather than "d k", matching how books write the compact form.
    folded = re.sub(r"[.'`’]", "", folded)
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()

    parts = folded.split()
    suffix = None
    while len(parts) > 1 and parts[-1] in SUFFIXES:
        suffix = parts[-1].rstrip(".")
        parts = parts[:-1]
    return " ".join(parts), suffix


def normalize_name(name: str) -> str:
    """Base match key, suffix removed. See `split_suffix`."""
    return split_suffix(name)[0]


def _initial_key(normalized: str) -> str | None:
    """`first-initial + last name`, e.g. "marvin harrison" -> "m harrison"."""
    parts = normalized.split()
    if len(parts) < 2:
        return None
    return f"{parts[0][0]} {parts[-1]}"


class PlayerResolver:
    """Resolves book name strings against an nflverse player universe.

    `players` must carry `gsis_id`, `display_name` and `position`. `team` is
    optional but strongly recommended -- without it, candidate scoping falls
    back to the whole universe and same-name collisions become possible.
    """

    def __init__(
        self,
        players: pl.DataFrame,
        *,
        overrides: pl.DataFrame | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        skill_positions_only: bool = True,
    ):
        self.fuzzy_threshold = fuzzy_threshold

        frame = players
        if skill_positions_only and "position" in frame.columns:
            frame = frame.filter(pl.col("position").is_in(list(SKILL_POSITIONS)))
        frame = frame.filter(pl.col("gsis_id").is_not_null())

        self._rows: list[dict] = []
        for row in frame.iter_rows(named=True):
            display = row.get("display_name") or row.get("full_name") or ""
            normalized, suffix = split_suffix(display)
            if not normalized:
                continue
            last_season = row.get("last_season")
            self._rows.append(
                {
                    "gsis_id": row["gsis_id"],
                    "display_name": display,
                    "normalized": normalized,
                    "suffix": suffix,
                    "initial_key": _initial_key(normalized),
                    "team": row.get("team"),
                    "position": row.get("position"),
                    "last_season": int(last_season) if last_season is not None else None,
                }
            )

        self._overrides: dict[tuple[str, str | None], str] = {}
        if overrides is not None and overrides.height:
            for row in overrides.iter_rows(named=True):
                book_name = row.get("book_name")
                gsis_id = row.get("gsis_id")
                if not book_name or not gsis_id:
                    continue
                team = (row.get("team") or "").strip().upper() or None
                self._overrides[(normalize_name(book_name), team)] = gsis_id

    # ------------------------------------------------------------------ lookup

    def _candidates(self, teams: list[str] | None) -> list[dict]:
        if not teams:
            return self._rows
        wanted = {t.upper() for t in teams if t}
        return [r for r in self._rows if (r["team"] or "").upper() in wanted]

    @staticmethod
    def _disambiguate(matches: list[dict], query_suffix: str | None) -> list[dict]:
        """Narrow several same-base-name candidates to one, if we honestly can.

        Two tie-breaks, in order:

        1. **The suffix**, when the book supplied one. "Michael Pittman Jr."
           against a Sr./Jr. pair is unambiguous.
        2. **Recency**, when it is not. Books only post props for active
           players, so among a retired father and an active son the active one
           is right. Requires a STRICT winner -- if two candidates share the
           most recent season, this gives up rather than guessing.

        Returns the surviving candidates; a list longer than one means the
        caller must raise.
        """
        if query_suffix:
            suffixed = [c for c in matches if c["suffix"] == query_suffix]
            if len(suffixed) == 1:
                return suffixed
            if suffixed:
                matches = suffixed

        seasons = [c["last_season"] for c in matches if c["last_season"] is not None]
        if len(seasons) == len(matches) and seasons:
            newest = max(seasons)
            if seasons.count(newest) == 1:
                return [c for c in matches if c["last_season"] == newest]
        return matches

    def _match(
        self,
        normalized: str,
        candidates: list[dict],
        position: str | None,
        query_suffix: str | None = None,
    ) -> tuple[Resolution | None, list[dict]]:
        """Try to match within one candidate pool.

        Returns (resolution, ambiguous_matches). A non-empty second element
        means the pool contains several equally good matches and the caller
        must raise rather than widen the search or guess.
        """
        if position:
            narrowed = [c for c in candidates if c["position"] == position]
            candidates = narrowed or candidates

        exact = [c for c in candidates if c["normalized"] == normalized]
        if len(exact) > 1:
            exact = self._disambiguate(exact, query_suffix)
        if len(exact) > 1:
            return None, exact
        if len(exact) == 1:
            c = exact[0]
            return (
                Resolution("", c["gsis_id"], c["display_name"], c["team"], "exact", 1.0),
                [],
            )

        key = _initial_key(normalized)
        if key:
            initial = [c for c in candidates if c["initial_key"] == key]
            if len(initial) > 1:
                initial = self._disambiguate(initial, query_suffix)
            if len(initial) > 1:
                return None, initial
            if len(initial) == 1:
                c = initial[0]
                return (
                    Resolution("", c["gsis_id"], c["display_name"], c["team"], "initial", 0.95),
                    [],
                )

        scored = sorted(
            ((SequenceMatcher(None, normalized, c["normalized"]).ratio(), c) for c in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if scored and scored[0][0] >= self.fuzzy_threshold:
            best_score, best = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            # A near-tie means we cannot tell two players apart. Raising is
            # correct: a wrong id silently attributes one player's props to
            # another, which is worse than a loud failure.
            if best_score - runner_up >= 0.02:
                return (
                    Resolution(
                        "", best["gsis_id"], best["display_name"], best["team"], "fuzzy", best_score
                    ),
                    [],
                )
        return None, []

    def resolve(
        self,
        book_name: str,
        *,
        teams: list[str] | None = None,
        position: str | None = None,
    ) -> Resolution:
        """Resolve one book name, or raise `UnresolvedPlayerError`."""
        normalized, query_suffix = split_suffix(book_name)
        if not normalized:
            raise UnresolvedPlayerError(book_name, None)

        # 1. Manual overrides, team-specific first, then team-agnostic.
        for key in ((normalized, t.upper()) for t in (teams or []) if t):
            if key in self._overrides:
                return Resolution(
                    book_name, self._overrides[key], book_name, key[1], "override", 1.0
                )
        if (normalized, None) in self._overrides:
            return Resolution(
                book_name, self._overrides[(normalized, None)], book_name, None, "override", 1.0
            )

        # 2-4. Match within the event's teams first.
        scoped = self._candidates(teams)
        resolution, ambiguous = self._match(normalized, scoped, position, query_suffix)
        if ambiguous:
            # Genuine collision inside the event (two same-named players on
            # these teams). Never guess and never widen -- widening can only
            # add more candidates. This is what the override file is for.
            raise UnresolvedPlayerError(
                book_name,
                ",".join(teams or []),
                [f"{c['display_name']} ({c['gsis_id']}, {c['position']})" for c in ambiguous],
            )

        # 5. Fall back to the full universe. A player traded mid-season is
        # still on his old team in the roster snapshot, so scoping to the
        # event's two teams misses him -- and the roster snapshot lags trades
        # by design. Widening only when the scoped pool produced nothing keeps
        # the disambiguation benefit of scoping while not losing traded
        # players, which is precisely the population whose role just changed.
        if resolution is None and teams:
            resolution, ambiguous = self._match(normalized, self._rows, position, query_suffix)
            if ambiguous:
                raise UnresolvedPlayerError(
                    book_name,
                    ",".join(teams or []),
                    [f"{c['display_name']} ({c['gsis_id']}, {c['position']})" for c in ambiguous],
                )

        if resolution is not None:
            return Resolution(
                book_name,
                resolution.gsis_id,
                resolution.matched_name,
                resolution.team,
                resolution.method,
                resolution.score,
            )

        scored = sorted(
            (
                (SequenceMatcher(None, normalized, c["normalized"]).ratio(), c)
                for c in (scoped or self._rows)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        raise UnresolvedPlayerError(
            book_name,
            ",".join(teams or []),
            [f"{c['display_name']} ({score:.2f})" for score, c in scored[:5]],
        )

    def resolve_many(
        self,
        names: list[str],
        *,
        teams: list[str] | None = None,
    ) -> tuple[list[Resolution], list[UnresolvedPlayerError]]:
        """Resolve a batch, collecting failures rather than raising on the first.

        Callers decide what to do with the failures. The props pipeline raises
        if any are present; the snapshot logger records them and carries on.
        """
        resolved: list[Resolution] = []
        failures: list[UnresolvedPlayerError] = []
        for name in names:
            try:
                resolved.append(self.resolve(name, teams=teams))
            except UnresolvedPlayerError as exc:
                failures.append(exc)
        return resolved, failures

    @property
    def universe_size(self) -> int:
        return len(self._rows)
