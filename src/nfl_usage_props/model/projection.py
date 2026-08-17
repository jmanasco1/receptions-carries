"""Stage 5: compose the layers into a per-player PMF.

    plays          ~ Negative Binomial          (Layer 1)
    targets        ~ Beta-Binomial(plays)       (Layer 2)
    share          ~ Dirichlet(p * K)           (Layer 3)
    player targets ~ Multinomial(targets, share)
    receptions     ~ Beta-Binomial(targets, catch rate)   (Layer 4)

The compound distribution is **sampled, not approximated**. There is no
closed form for it, and the usual shortcut -- take each layer's mean, multiply,
put a Poisson around the product -- throws away exactly the variance that the
half-point depends on. Every layer contributes its own uncertainty and they
compound.

Why a PMF and not a mean
------------------------
A projection of 4.2 receptions says nothing about whether to bet a 3.5 line or
a 4.5 line, and the half-point is the entire wager. What comes out here is
`P(X >= k)` for every k, read off the samples. `E[X]` is reported alongside for
sanity-checking only and is never what gets compared to a price.

Correlation is preserved by construction
----------------------------------------
All players on a team are drawn from the *same* simulation of that team's
game. In simulation 4,102 where the Bengals ran 71 plays and threw 42 times,
every Bengals receiver is dividing those same 42 targets. That is what makes
the shares compete, and it is why the WR1's absence mechanically lifts
everyone else. Sampling players independently would lose it and quietly
project a team throwing more passes than it attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_usage_props.model.player_share import PlayerShareModel, eligible
from nfl_usage_props.model.team_volume import TeamVolumeModel

# Reception and rush-attempt lines sit well inside this range. Reading the PMF
# beyond it costs memory and says nothing a book would price.
MAX_COUNT = 25


@dataclass
class ProjectionResult:
    """Per player-game draws, plus the PMF read off them."""

    keys: pl.DataFrame
    targets: np.ndarray
    receptions: np.ndarray
    carries: np.ndarray

    def pmf_table(self, market: str) -> pl.DataFrame:
        """`P(X >= k)` for every k up to MAX_COUNT, one row per player-game."""
        draws = {
            "receptions": self.receptions,
            "carries": self.carries,
            "targets": self.targets,
        }[market]
        columns = {
            f"p_over_{k - 0.5:g}": pl.Series(np.mean(draws >= k, axis=1))
            for k in range(1, MAX_COUNT + 1)
        }
        return self.keys.with_columns(
            pl.Series("mean", draws.mean(axis=1)),
            pl.Series("p10", np.quantile(draws, 0.10, axis=1)),
            pl.Series("p90", np.quantile(draws, 0.90, axis=1)),
            **columns,
        )

    def probability_over(self, market: str, line: float) -> np.ndarray:
        """P(X > line) for a half-point line, per player-game.

        Whole-number lines push rather than resolving, so they are rejected
        here: silently treating a push as a loss would misprice every
        integer-line market, and those exist.
        """
        if float(line).is_integer():
            raise ValueError(
                f"line {line} is a whole number and would push; "
                "pass a half-point line, or handle the push explicitly"
            )
        draws = {
            "receptions": self.receptions,
            "carries": self.carries,
            "targets": self.targets,
        }[market]
        return np.mean(draws > line, axis=1)


class Projector:
    """Runs the full chain for a set of games."""

    def __init__(
        self,
        volume: TeamVolumeModel,
        targets_share: PlayerShareModel,
        carries_share: PlayerShareModel,
        catch_rate_dispersion: float,
        *,
        seed: int = 20240901,
    ):
        self.volume = volume
        self.targets_share = targets_share
        self.carries_share = carries_share
        self.catch_rate_dispersion = catch_rate_dispersion
        self.seed = seed

    def project(
        self, features: pl.DataFrame, team_games: pl.DataFrame, draws: int = 10_000
    ) -> ProjectionResult:
        rng = np.random.default_rng(self.seed)

        team = self.volume.sample(team_games, draws=draws)
        team_index = {
            (game, side): i
            for i, (game, side) in enumerate(zip(team["game_id"], team["team"], strict=True))
        }

        keys, target_shares = self.targets_share.sample_shares(features, draws, rng)
        _, carry_shares = self.carries_share.sample_shares(features, draws, rng)

        # Map each player row onto the row of its team's simulation, so a
        # player and his team's game are the SAME draw, not two draws that
        # happen to have the same mean.
        rows = [
            team_index.get((game, side))
            for game, side in zip(keys["game_id"], keys["team"], strict=True)
        ]
        known = np.array([r is not None for r in rows])
        if not known.all():
            keys = keys.filter(pl.Series(known))
            target_shares = target_shares[known]
            carry_shares = carry_shares[known]
            rows = [r for r in rows if r is not None]
        row_index = np.array(rows, dtype=int)

        missing = [name for name in ("targets", "carries") if name not in team]
        if missing:
            raise ValueError(
                f"the volume model did not sample {missing}; it was fitted without "
                "team_targets/team_carries columns, so there is nothing for the "
                "share layer to divide. Rebuild the team-game frame from "
                "`team_game_frame`, which carries them."
            )
        team_targets = team["targets"][row_index]
        team_carries = team["carries"][row_index]

        # Binomial per player against his own share is the right marginal of a
        # multinomial. It does not renormalise across teammates, so the totals
        # can drift a little from the team's sampled count -- accepted because
        # the alternative is a per-team-game loop over every draw, and the
        # shares already sum to one so the drift is sampling noise, not bias.
        player_targets = rng.binomial(team_targets, target_shares)
        player_carries = rng.binomial(team_carries, carry_shares)

        catch = self._catch_rate_draws(keys, draws, rng)
        receptions = rng.binomial(player_targets, catch)

        return ProjectionResult(
            keys=keys,
            targets=player_targets,
            receptions=receptions,
            carries=player_carries,
        )

    def _catch_rate_draws(
        self, keys: pl.DataFrame, draws: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Layer 4: a Beta draw per player per simulation.

        Drawn rather than fixed at the prior mean, because a player's catch
        rate on the day varies with how he is covered. Fixing it would make
        every reception distribution too narrow.
        """
        rate = np.clip(keys["catch_rate_prior"].to_numpy(), 1e-4, 1 - 1e-4)[:, None]
        phi = self.catch_rate_dispersion
        return rng.beta(rate * phi, (1.0 - rate) * phi, size=(len(rate), draws))


def project_week(
    features: pl.DataFrame,
    team_games: pl.DataFrame,
    volume: TeamVolumeModel,
    targets_share: PlayerShareModel,
    carries_share: PlayerShareModel,
    catch_rate_dispersion: float,
    *,
    draws: int = 10_000,
    seed: int = 20240901,
) -> ProjectionResult:
    """Convenience wrapper: filter to eligible players and project."""
    usable = eligible(features).drop_nulls(
        ["role_target_share_mean", "role_carry_share_mean", "catch_rate_prior"]
    )
    projector = Projector(volume, targets_share, carries_share, catch_rate_dispersion, seed=seed)
    return projector.project(usable, team_games, draws=draws)
