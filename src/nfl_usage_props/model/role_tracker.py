"""State-space role tracker.

Replaces fixed exponential decay entirely. A half-life says "four games ago
matters half as much as last game" and says it identically for a receiver
whose role is stable, a receiver who just lost the man in front of him, and a
receiver who played six snaps in a blowout. Those are three different amounts
of evidence and a decay constant cannot tell them apart.

The model
---------
For one player and one usage layer, on the logit scale:

    state        x_t = x_{t-1} + w_t,    w_t ~ N(0, Q_t)
    observation  z_t = x_t + v_t,        v_t ~ N(0, R_t)

x_t is the player's true role. A scalar Kalman filter runs it forward. Three
things make it behave like football rather than like a textbook:

**R_t is earned, not assumed.** A share of 2/40 targets is a far weaker
statement than 12/40. The observation variance comes from the delta method on
a binomial proportion, so a game with few team opportunities moves the state
less, automatically.

**Exposure inflates R_t.** A player who left in the first quarter produced a
share over a partial game. His number is real but it describes less football,
so it is downweighted by his exposure rather than being either trusted fully
or dropped.

**Q_t is event-driven.** Role changes are not smooth. In a normal week the
process noise is the layer's base value, tuned so the effective memory matches
how fast that layer actually moves (snap share ~4 games, carries ~5-6, targets
~7-8). In a week where something known happened -- the starter ahead is out, a
trade, a return from IR -- Q_t is multiplied and the filter is allowed to move
in one step what would otherwise take a month.

**Changepoints are detected, not waited out.** When the filter is repeatedly
and consistently surprised in the same direction, that is a regime change it
was not told about. Rather than dragging the stale level along, it inflates
its own uncertainty so the next observations dominate.

What comes out
--------------
For every game, the *prior* estimate: mean and variance of the role given
strictly earlier games. That is the leakage-safe quantity, and it is the one
Stage 3 feeds into the Monte Carlo -- as a distribution, not a point, so
role uncertainty propagates into the projection instead of being asserted away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Continuity correction for shares at the boundary. logit(0) is -inf, and a
# receiver with zero targets in a game is entirely ordinary, so the raw share
# cannot be used. Jeffreys' 1/2 keeps a zero informative but finite: it says
# "low, and how low depends on how many chances there were".
JEFFREYS = 0.5

# Ceiling on a single observation's variance. Without it a one-opportunity game
# produces a number so noisy it is indistinguishable from no data, but still
# costs a filter step. With it, such a game nudges rather than nothing.
MAX_OBSERVATION_VARIANCE = 25.0

# Floor on exposure, so a two-snap cameo cannot divide the variance to infinity.
MIN_EXPOSURE = 0.05


# --------------------------------------------------------------------------
# Calibrating process noise
# --------------------------------------------------------------------------
# Process noise is not a free knob to be eyeballed. For a local level model the
# steady-state Kalman gain K solves K^2 R + qK - q = 0, so the filter's
# effective memory m = 1/K is pinned by the ratio of process to observation
# noise. Inverting gives q exactly:
#
#     q = R / (m * (m - 1))
#
# which turns "target share should behave like a seven-and-a-half game memory"
# -- a football statement someone can argue with -- into a number, instead of
# leaving a number nobody can interpret. The reference workloads below are a
# typical starter's game at each layer; they set what "typical" means for the
# calibration and nothing else. Real players deviate, and should: a low-volume
# player has a larger R, hence a longer effective memory, which is the correct
# response to a weaker signal rather than a bug in it.
REFERENCE_WORKLOAD: dict[str, tuple[float, float]] = {
    "snap_share": (45.0, 65.0),  # a starter: 45 of 65 offensive snaps
    "carry_share": (14.0, 26.0),  # a lead back: 14 of 26 team carries
    "target_share": (7.0, 35.0),  # a first read: 7 of 35 team targets
}

# How fast each layer actually moves. Snap share responds to a depth chart
# change within a month; target share is the stickiest thing a receiver owns.
TARGET_MEMORY_GAMES: dict[str, float] = {
    "snap_share": 4.0,
    "carry_share": 5.5,
    "target_share": 7.5,
}


def process_noise_for_memory(memory_games: float, observation_var: float) -> float:
    """Process noise giving a filter the requested effective memory in games."""
    if memory_games <= 1.0:
        raise ValueError(f"effective memory must exceed one game, got {memory_games}")
    return observation_var / (memory_games * (memory_games - 1.0))


def calibrated_process_noise(layer: str) -> float:
    """The process noise for `layer`, derived from its target memory."""
    try:
        successes, trials = REFERENCE_WORKLOAD[layer]
        memory = TARGET_MEMORY_GAMES[layer]
    except KeyError as exc:
        raise KeyError(
            f"no reference workload for layer {layer!r}; "
            f"expected one of {sorted(REFERENCE_WORKLOAD)}"
        ) from exc
    return process_noise_for_memory(memory, observation_variance(successes, trials))


def effective_memory_games(process_noise: float, observation_var: float) -> float:
    """Inverse of `process_noise_for_memory` -- what a given q actually buys."""
    q = process_noise
    gain = (-q + math.sqrt(q * q + 4.0 * q * observation_var)) / (2.0 * observation_var)
    return 1.0 / gain


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    # Branch to avoid overflow in exp for large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class RoleObservation:
    """One game's realised usage for one player and one layer.

    successes: the player's count (targets, carries, snaps).
    trials:    the team denominator for the same play universe.
    exposure:  1.0 for a full game's worth of availability, lower when the
               player was on the field for only part of it. This is *not* the
               share being measured -- it is how much of the game the share had
               a chance to accumulate over.
    event:     True when something known before kickoff should have changed
               this player's role (teammate ruled out, trade, activation).
    """

    successes: float
    trials: float
    exposure: float = 1.0
    event: bool = False
    season_break: bool = False

    @property
    def observed(self) -> bool:
        """False when the game carries no information about this layer."""
        return self.trials > 0


@dataclass(frozen=True)
class RoleEstimate:
    """The role as of *before* a game, given strictly earlier games."""

    mean_logit: float
    var_logit: float
    n_observations: int
    changepoint: bool = False

    @property
    def share(self) -> float:
        """Point estimate on the share scale.

        This is the median of the implied logit-normal, not its mean. The
        distinction matters: taking expit of the mean logit and calling it an
        expected share would bias every projection toward 0.5. Stage 3 samples
        the distribution rather than using this, precisely to avoid the issue.
        """
        return expit(self.mean_logit)

    @property
    def sd_logit(self) -> float:
        return math.sqrt(self.var_logit)


def corrected_share(
    successes: float,
    trials: float,
    *,
    prior_share: float = 0.5,
    prior_strength: float = 2.0 * JEFFREYS,
) -> float:
    """Continuity-corrected share, anchored on a prior rather than on 1/2.

    A raw 0/26 cannot be put on the logit scale, so something has to stand in.
    The conventional Jeffreys choice adds half a success and half a failure,
    which is exactly a uniform prior on the share -- a perfectly good default
    when nothing is known, and a badly wrong one here.

    Concretely: it floors every zero-carry tight end at (0+0.5)/(26+1) = 1.9%
    of his team's carries, every week, no matter how many weeks he goes
    without one. Since Layer 3 normalises across the roster, a dozen such
    players between them soak up a sixth of the simplex, and the backs it is
    stolen from are under-projected by the same amount. Measured on 2024: RBs
    received 68% of the modelled carry mass against 83% of the actual carries,
    while tight ends got 5.4% against 0.4%.

    Anchoring the pseudo-count on the position's own prior fixes the floor at
    something defensible -- a tight end starts near his position's real carry
    share, not near a coin flip. `prior_share=0.5` recovers Jeffreys exactly,
    so this is a generalisation rather than a change of default.
    """
    a = prior_share * prior_strength
    return (successes + a) / (trials + prior_strength)


def observation_variance(
    successes: float,
    trials: float,
    exposure: float = 1.0,
    *,
    jeffreys: float = JEFFREYS,
    prior_share: float = 0.5,
    prior_strength: float | None = None,
) -> float:
    """Variance of the observed logit share, by the delta method.

    For p_hat ~ Binomial(n, p)/n, var(p_hat) = p(1-p)/n, and the delta method
    carries that to the logit scale as 1 / (n p (1-p)). The Jeffreys-corrected
    p keeps it finite at the boundaries, where it is largest -- which is the
    correct behaviour: a zero out of five says much less than a zero out of
    forty.
    """
    if trials <= 0:
        return math.inf
    strength = prior_strength if prior_strength is not None else 2.0 * jeffreys
    p = corrected_share(successes, trials, prior_share=prior_share, prior_strength=strength)
    variance = 1.0 / (trials * p * (1.0 - p))
    variance /= max(exposure, MIN_EXPOSURE)
    return min(variance, MAX_OBSERVATION_VARIANCE)


def observed_logit(
    successes: float,
    trials: float,
    *,
    jeffreys: float = JEFFREYS,
    prior_share: float = 0.5,
    prior_strength: float | None = None,
) -> float:
    """The observation, continuity-corrected so 0 and 1 are representable."""
    strength = prior_strength if prior_strength is not None else 2.0 * jeffreys
    return logit(
        corrected_share(successes, trials, prior_share=prior_share, prior_strength=strength)
    )


def filter_series(
    observations: list[RoleObservation],
    *,
    process_noise: float,
    prior_mean_logit: float,
    prior_var_logit: float,
    event_multiplier: float = 1.0,
    season_break_weight: float = 1.0,
    prior_share: float | None = None,
    changepoint_lookback: int = 2,
    changepoint_threshold_sd: float = 2.5,
) -> list[RoleEstimate]:
    """Run the filter forward, returning the PRIOR estimate for each game.

    The returned list is aligned with `observations`: element t is what the
    filter believed about the player's role *before* game t, having seen games
    0..t-1 and nothing else. Using element t as a feature for game t is
    therefore safe by construction, which is the whole point of returning the
    prior rather than the smoothed state.

    `season_break_weight` is how much of the prior season's information
    survives the offseason, as a fraction of its precision: 0.75 means a
    player arrives at week 1 knowing three quarters of what he knew in
    January. It is a precision discount rather than added noise, so the level
    is kept and only the confidence in it is reduced -- which is what an
    offseason does to a receiver whose team, coordinator and depth chart may
    all have moved.
    """
    mean = prior_mean_logit
    variance = prior_var_logit
    seen = 0
    surprise_run = 0
    surprise_sign = 0.0
    last_innovation = 0.0
    estimates: list[RoleEstimate] = []

    for obs in observations:
        # --- predict: the state drifts, and how much depends on the week ---
        q = process_noise * (event_multiplier if obs.event else 1.0)
        carried = variance
        if obs.season_break and season_break_weight > 0.0:
            carried = variance / season_break_weight
        prior_var = carried + q
        prior_mean = mean

        # A detected changepoint means the filter's confidence is wrong, not
        # just its level. Widening the prior lets the next observation dominate
        # instead of being averaged into a level that no longer applies.
        #
        # The width is the size of the surprise itself: having been wrong by
        # `last_innovation` repeatedly and in the same direction, the honest
        # uncertainty about the level is at least that big. Anything smaller --
        # a multiple of the process noise, say -- is dominated by the variance
        # it is meant to widen and silently does nothing.
        changepoint = surprise_run >= changepoint_lookback
        if changepoint:
            prior_var = max(prior_var, last_innovation * last_innovation)
            surprise_run = 0

        estimates.append(
            RoleEstimate(
                mean_logit=prior_mean,
                var_logit=prior_var,
                n_observations=seen,
                changepoint=changepoint,
            )
        )

        if not obs.observed:
            # No information this week -- the drift still happened, so carry the
            # widened prior forward. This is how a bye or an inactive week
            # correctly increases uncertainty without inventing an observation.
            mean, variance = prior_mean, prior_var
            continue

        # --- update ---
        # The continuity correction is anchored on the same prior the filter
        # starts from, so a player who never touches the ball converges toward
        # his position's real share instead of toward a coin flip.
        anchor = expit(prior_mean_logit) if prior_share is None else prior_share
        z = observed_logit(obs.successes, obs.trials, prior_share=anchor)
        r = observation_variance(obs.successes, obs.trials, obs.exposure, prior_share=anchor)
        innovation = z - prior_mean
        innovation_var = prior_var + r

        gain = prior_var / innovation_var
        mean = prior_mean + gain * innovation
        variance = (1.0 - gain) * prior_var
        seen += 1

        # Track consistent, same-signed surprise. One shock is noise; a run of
        # them in the same direction is a role that moved.
        last_innovation = innovation
        standardised = innovation / math.sqrt(innovation_var)
        if abs(standardised) >= changepoint_threshold_sd:
            sign = 1.0 if standardised > 0 else -1.0
            surprise_run = surprise_run + 1 if sign == surprise_sign else 1
            surprise_sign = sign
        else:
            surprise_run = 0
            surprise_sign = 0.0

    return estimates
