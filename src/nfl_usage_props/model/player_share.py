"""Layer 3: how a team's targets and carries divide among its players.

This is where the edge is. Layers 1 and 2 barely beat a constant; the role
tracker predicts target share at r=0.81. So this layer gets the care.

The model
---------
For one team-game with eligible players 1..k:

    p           ~ role tracker priors, normalised to the simplex
    shares      ~ Dirichlet(p * K)
    counts      ~ Multinomial(team opportunities, shares)

Two distinct sources of variation, and collapsing them is the most common way
to get this wrong:

* **Role uncertainty** — we do not know a player's true share. The tracker
  reports a logit-normal per player, and it is *sampled*, not collapsed to its
  mean. A receiver whose role just changed has a wide prior and must produce a
  wide projection.
* **Game-to-game variation** — even a player whose true share is exactly 22%
  does not see 22% of the targets every week. That is the Dirichlet's
  concentration K, and it is fitted rather than assumed.

Why a Dirichlet-Multinomial rather than k independent Betas
-----------------------------------------------------------
Because shares compete. If the WR1 draws 35% of the targets this week, the
other nine players are dividing 65%, not their usual amounts. Independent
per-player models cannot represent that and will happily project a team
throwing 130% of its passes. The simplex constraint is the point, and it is
why the WR1's absence mechanically raises everyone else rather than needing a
"teammate out" feature.

Eligibility
-----------
The simplex covers skill positions only. `participation` lists all eleven
players on the field, so an unfiltered roster puts five offensive linemen on
the simplex with a rookie-receiver prior each, diluting every real share by
about a third. Linemen do catch passes -- tackle-eligible plays exist -- but
they account for 0.075% of targets and 0.09% of carries in 2023-24, which is
the documented cost of excluding them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import expit as expit_array
from scipy.special import gammaln

# Positions that meaningfully compete for targets and carries.
ELIGIBLE_POSITIONS = frozenset({"WR", "TE", "RB", "FB", "HB", "QB"})

# A share prior can be arbitrarily small but never zero: a zero Dirichlet
# parameter makes that player's share identically zero and the likelihood
# undefined the moment he catches a pass.
_MIN_SHARE = 1e-4


@dataclass
class ShareFit:
    """Fitted concentration for one layer."""

    layer: str
    concentration: float
    log_likelihood: float
    n_team_games: int
    converged: bool

    def summary(self) -> str:
        return (
            f"{self.layer}: K={self.concentration:.1f} "
            f"(loglik={self.log_likelihood:,.0f} over {self.n_team_games:,} team-games)"
        )


def eligible(panel: pl.DataFrame) -> pl.DataFrame:
    """Restrict to players who compete for touches."""
    return panel.filter(pl.col("position").is_in(list(ELIGIBLE_POSITIONS)))


def dirichlet_multinomial_loglik(
    counts: np.ndarray, alpha: np.ndarray, group_starts: np.ndarray
) -> np.ndarray:
    """Per-team-game log-likelihood, vectorised over ragged groups.

    `counts` is flat over every player-game; `alpha` is the same shape, or
    (rows, draws) to evaluate many role samples at once. `group_starts` gives
    each team-game's offset. Ragged rather than padded because rosters run from
    8 to 20 eligible players and padding with zeros would put fictitious
    players on every simplex.

    Returns one value per team-game (per draw), not a total, because the caller
    integrates over role uncertainty *within* each team-game before summing.
    """
    starts = group_starts[:-1]
    two_dimensional = alpha.ndim == 2
    n = counts[:, None] if two_dimensional else counts

    alpha_sum = np.add.reduceat(alpha, starts, axis=0)
    count_sum = np.add.reduceat(n, starts, axis=0)
    per_player = np.add.reduceat(gammaln(n + alpha) - gammaln(alpha), starts, axis=0)
    return gammaln(alpha_sum) - gammaln(count_sum + alpha_sum) + per_player


class PlayerShareModel:
    """Dirichlet-Multinomial share model for one layer (targets or carries)."""

    def __init__(self, layer: str, *, seed: int = 20240901):
        if layer not in ("targets", "carries"):
            raise ValueError(f"layer must be 'targets' or 'carries', got {layer!r}")
        self.layer = layer
        self.seed = seed
        self.fit_result: ShareFit | None = None

    @property
    def prior_column(self) -> str:
        return "role_target_share_mean" if self.layer == "targets" else "role_carry_share_mean"

    @property
    def prior_sd_column(self) -> str:
        return "role_target_share_sd" if self.layer == "targets" else "role_carry_share_sd"

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        features: pl.DataFrame,
        *,
        bounds: tuple[float, float] = (1.0, 2000.0),
        role_draws: int = 32,
    ) -> ShareFit:
        """Fit the concentration K by maximum MARGINAL likelihood.

        Only K is fitted. The per-player means come from the role tracker,
        which has already done the shrinkage over time -- refitting them would
        use the same data twice and break the tracker's as-of guarantee.

        The marginal matters. Fitting K against the tracker's *mean* share
        makes K absorb every source of spread, role uncertainty included; then
        sampling draws role uncertainty again on top and the variance is
        counted twice. That is not a rounding error -- it made carry
        projections cover 87% inside a nominal 50% interval against 77%
        achievable, which is a systematically too-wide distribution on every
        carry prop.

        So K is fitted against the same generative process the sampler runs:
        role is drawn from the tracker's logit-normal, the Dirichlet sits on
        top, and the likelihood is averaged over those draws inside each
        team-game before summing. K then measures only what is left after role
        uncertainty, which is what it is supposed to mean.
        """
        counts, mean_logit, sd_logit, starts = self._prepare(features)
        if len(starts) < 2:
            raise ValueError("no usable team-games; check positions and prior columns")

        rng = np.random.default_rng(self.seed)
        alpha_unit = self._role_draws(mean_logit, sd_logit, starts, role_draws, rng)

        def negative_loglik(log_k: float) -> float:
            per_draw = dirichlet_multinomial_loglik(counts, alpha_unit * np.exp(log_k), starts)
            # log-mean-exp across role draws within each team-game: the Monte
            # Carlo estimate of the marginal likelihood. Averaging the logs
            # instead would be a different (and wrong) quantity.
            peak = per_draw.max(axis=1, keepdims=True)
            marginal = peak[:, 0] + np.log(np.exp(per_draw - peak).mean(axis=1))
            value = float(np.sum(marginal))
            return np.inf if not np.isfinite(value) else -value

        outcome = minimize_scalar(
            negative_loglik,
            bounds=(np.log(bounds[0]), np.log(bounds[1])),
            method="bounded",
            options={"xatol": 1e-3},
        )
        concentration = float(np.exp(outcome.x))
        at_bound = (
            abs(outcome.x - np.log(bounds[0])) < 1e-2 or abs(outcome.x - np.log(bounds[1])) < 1e-2
        )
        self.fit_result = ShareFit(
            layer=self.layer,
            concentration=concentration,
            log_likelihood=-float(outcome.fun),
            n_team_games=len(starts) - 1,
            converged=bool(outcome.success) and not at_bound,
        )
        return self.fit_result

    def _prepare(
        self, features: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten to (counts, prior mean logit, prior sd, offsets)."""
        frame = (
            eligible(features)
            .drop_nulls([self.prior_column, self.prior_sd_column, self.layer])
            .sort("game_id", "team", "gsis_id")
        )
        counts = frame[self.layer].to_numpy().astype(float)
        keys = frame.select("game_id", "team").with_row_index("_i")
        boundaries = (
            keys.group_by("game_id", "team", maintain_order=True)
            .agg(pl.col("_i").min().alias("start"))
            .sort("start")
        )
        starts = np.append(boundaries["start"].to_numpy(), len(frame))
        return (
            counts,
            frame[self.prior_column].to_numpy(),
            frame[self.prior_sd_column].to_numpy(),
            starts,
        )

    @staticmethod
    def _role_draws(
        mean_logit: np.ndarray,
        sd_logit: np.ndarray,
        starts: np.ndarray,
        draws: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simplex-normalised role samples, shape (rows, draws).

        The tracker estimates each player independently, so his prior does not
        sum to one across the roster and the sum carries no information.
        Normalising within the team-game is what turns k independent estimates
        into a distribution over who gets the ball.
        """
        sampled = expit_array(
            mean_logit[:, None] + sd_logit[:, None] * rng.standard_normal((len(mean_logit), draws))
        )
        sampled = np.maximum(sampled, _MIN_SHARE)
        totals = np.add.reduceat(sampled, starts[:-1], axis=0)
        repeats = np.diff(starts)
        return sampled / np.repeat(totals, repeats, axis=0)

    # --------------------------------------------------------------- sample

    def sample_shares(
        self,
        features: pl.DataFrame,
        draws: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Draw player shares for every team-game.

        Returns the row keys and an array of shape (rows, draws). Role
        uncertainty is sampled from the tracker's logit-normal *before* the
        Dirichlet draw, so a player whose role just changed produces a wide
        projection rather than a confident wrong one.
        """
        self._require_fit()
        rng = rng or np.random.default_rng(self.seed)
        frame = (
            eligible(features)
            .drop_nulls([self.prior_column, self.prior_sd_column])
            .sort("game_id", "team", "gsis_id")
        )
        mean_logit = frame[self.prior_column].to_numpy()
        sd_logit = frame[self.prior_sd_column].to_numpy()

        keys = frame.select("game_id", "team").with_row_index("_i")
        boundaries = (
            keys.group_by("game_id", "team", maintain_order=True)
            .agg(pl.col("_i").min().alias("start"))
            .sort("start")
        )
        starts = np.append(boundaries["start"].to_numpy(), len(frame))

        # Parameter uncertainty: one role draw per player per simulation, and
        # exactly the process `fit` integrated over, so K means the same thing
        # here as it did there.
        alpha_unit = self._role_draws(mean_logit, sd_logit, starts, draws, rng)

        # Gamma draws normalised within each team-game are a Dirichlet,
        # vectorised over every simulation at once.
        gamma = rng.gamma(alpha_unit * self.fit_result.concentration)
        totals = np.add.reduceat(gamma, starts[:-1], axis=0)
        shares = gamma / np.repeat(totals, np.diff(starts), axis=0)

        # Carry the identifying columns plus anything downstream layers need on
        # the same rows and in the same order -- re-joining them later would
        # risk a reorder that silently pairs a player with someone else's draws.
        passthrough = [
            c
            for c in ("position", "season", "week", "catch_rate_prior", "opponent")
            if c in frame.columns
        ]
        return frame.select("game_id", "team", "gsis_id", *passthrough), shares

    def _require_fit(self) -> None:
        if self.fit_result is None:
            raise RuntimeError("model is not fitted; call fit() first")


def fit_catch_rate_dispersion(
    features: pl.DataFrame, *, bounds: tuple[float, float] = (1.0, 500.0)
) -> ShareFit:
    """Layer 4: Beta-Binomial dispersion for receptions out of targets.

    The mean comes from `catch_rate_prior`, which is already an as-of
    shrunk estimate; only the dispersion is fitted. A binomial would claim
    every target is an independent coin flip at the player's rate, which
    ignores that catch rate varies with how a defence plays him on the day.
    """
    frame = eligible(features).filter(pl.col("targets") > 0).drop_nulls(["catch_rate_prior"])
    successes = frame["receptions"].to_numpy().astype(float)
    trials = frame["targets"].to_numpy().astype(float)
    rate = np.clip(frame["catch_rate_prior"].to_numpy(), 1e-6, 1 - 1e-6)

    from scipy.special import betaln

    def negative_loglik(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        a, b = rate * phi, (1.0 - rate) * phi
        ll = betaln(successes + a, trials - successes + b) - betaln(a, b)
        total = float(np.sum(ll))
        return np.inf if not np.isfinite(total) else -total

    outcome = minimize_scalar(
        negative_loglik,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-3},
    )
    at_bound = (
        abs(outcome.x - np.log(bounds[0])) < 1e-2 or abs(outcome.x - np.log(bounds[1])) < 1e-2
    )
    return ShareFit(
        layer="catch_rate",
        concentration=float(np.exp(outcome.x)),
        log_likelihood=-float(outcome.fun),
        n_team_games=len(successes),
        converged=bool(outcome.success) and not at_bound,
    )
