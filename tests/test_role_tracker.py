"""Tests for the state-space role tracker.

The tracker is the piece most able to be subtly wrong while looking fine: it
will always produce plausible-looking numbers. These tests pin the properties
that make it better than a decay constant, because those are exactly the
properties that would disappear silently under a refactor.
"""

from __future__ import annotations

import math

import pytest

from nfl_usage_props.config import RoleTrackerConfig
from nfl_usage_props.model.role_tracker import (
    MAX_OBSERVATION_VARIANCE,
    REFERENCE_WORKLOAD,
    TARGET_MEMORY_GAMES,
    RoleObservation,
    calibrated_process_noise,
    effective_memory_games,
    expit,
    filter_series,
    logit,
    observation_variance,
    observed_logit,
    process_noise_for_memory,
)


def steady(observation: RoleObservation, n: int = 60, **kwargs):
    defaults = {"process_noise": 0.006, "prior_mean_logit": 0.0, "prior_var_logit": 4.0}
    return filter_series([observation] * n, **{**defaults, **kwargs})


# ------------------------------------------------------------------- basics


def test_logit_and_expit_round_trip():
    for p in (0.01, 0.2, 0.5, 0.83, 0.99):
        assert expit(logit(p)) == pytest.approx(p)


def test_expit_does_not_overflow_on_extreme_logits():
    assert expit(-800.0) == pytest.approx(0.0)
    assert expit(800.0) == pytest.approx(1.0)


def test_observed_logit_is_finite_at_both_boundaries():
    """A receiver with zero targets is ordinary. logit(0) is not."""
    assert math.isfinite(observed_logit(0, 40))
    assert math.isfinite(observed_logit(40, 40))


def test_zero_of_forty_is_a_stronger_statement_than_zero_of_five():
    assert observed_logit(0, 40) < observed_logit(0, 5)


# ------------------------------------------------------- observation variance


def test_more_trials_means_a_tighter_observation():
    assert observation_variance(5, 50) < observation_variance(1, 10)


def test_no_trials_is_infinite_variance():
    assert observation_variance(0, 0) == math.inf


def test_partial_exposure_inflates_variance():
    """A player who left in the first quarter produced a real number over less
    football. It should move the state less, not equally."""
    full = observation_variance(4, 30, exposure=1.0)
    partial = observation_variance(4, 30, exposure=0.4)
    assert partial > full


def test_observation_variance_is_capped():
    assert observation_variance(0, 1, exposure=0.01) <= MAX_OBSERVATION_VARIANCE


# ---------------------------------------------------------------- filtering


def test_filter_returns_one_prior_estimate_per_observation():
    observations = [RoleObservation(7, 35) for _ in range(5)]
    assert len(steady(observations[0], n=5)) == 5


def test_first_estimate_is_the_untouched_prior():
    """Element 0 must not have seen observation 0 -- that is the whole contract."""
    estimates = filter_series(
        [RoleObservation(30, 35)],
        process_noise=0.006,
        prior_mean_logit=-2.0,
        prior_var_logit=4.0,
    )
    assert estimates[0].mean_logit == pytest.approx(-2.0)


def test_estimates_converge_toward_the_truth():
    truth = 0.25
    estimates = steady(RoleObservation(round(truth * 40), 40), n=40)
    assert estimates[-1].share == pytest.approx(truth, abs=0.02)


def test_uncertainty_shrinks_as_evidence_accumulates():
    estimates = steady(RoleObservation(10, 40), n=30)
    assert estimates[-1].var_logit < estimates[1].var_logit


def test_uncertainty_grows_across_an_unobserved_week():
    """A bye or an inactive week is not evidence, but time still passed."""
    observations = [RoleObservation(10, 40)] * 5 + [RoleObservation(0, 0)] * 3
    estimates = filter_series(
        observations, process_noise=0.006, prior_mean_logit=0.0, prior_var_logit=4.0
    )
    assert estimates[-1].var_logit > estimates[5].var_logit
    # The level is carried, not forgotten.
    assert estimates[-1].mean_logit == pytest.approx(estimates[5].mean_logit)


def test_an_unobserved_week_does_not_count_as_evidence():
    observations = [RoleObservation(10, 40)] * 4 + [RoleObservation(0, 0)]
    estimates = filter_series(
        observations, process_noise=0.006, prior_mean_logit=0.0, prior_var_logit=4.0
    )
    assert estimates[-1].n_observations == 4


def test_share_is_the_median_not_the_mean():
    """expit(mean logit) is the median of a logit-normal. The property that
    matters is that it is NOT pulled toward 0.5 by the variance."""
    tight = filter_series(
        [RoleObservation(8, 40)],
        process_noise=0.006,
        prior_mean_logit=logit(0.2),
        prior_var_logit=0.01,
    )
    wide = filter_series(
        [RoleObservation(8, 40)],
        process_noise=0.006,
        prior_mean_logit=logit(0.2),
        prior_var_logit=4.0,
    )
    assert tight[0].share == pytest.approx(wide[0].share)


# ------------------------------------------------------------- events, breaks


def test_an_event_lets_the_state_move_faster():
    """Same data, same filter -- the only difference is being told something
    changed. The flagged run must adapt further."""
    jump = [RoleObservation(4, 40)] * 6 + [RoleObservation(16, 40)] * 2
    flagged = [RoleObservation(o.successes, o.trials, event=(i == 6)) for i, o in enumerate(jump)]
    plain = filter_series(jump, process_noise=0.006, prior_mean_logit=0.0, prior_var_logit=4.0)
    evented = filter_series(
        flagged,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        event_multiplier=6.0,
    )
    assert evented[-1].mean_logit > plain[-1].mean_logit


def test_season_break_widens_uncertainty_without_losing_the_level():
    observations = [RoleObservation(10, 40)] * 6
    with_break = [
        RoleObservation(o.successes, o.trials, season_break=(i == 5))
        for i, o in enumerate(observations)
    ]
    plain = filter_series(
        observations, process_noise=0.006, prior_mean_logit=0.0, prior_var_logit=4.0
    )
    broken = filter_series(
        with_break,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        season_break_weight=0.75,
    )
    assert broken[5].var_logit > plain[5].var_logit
    assert broken[5].mean_logit == pytest.approx(plain[5].mean_logit)


def test_season_break_weight_of_one_is_a_no_op():
    observations = [RoleObservation(10, 40, season_break=(i == 3)) for i in range(6)]
    plain = filter_series(
        [RoleObservation(10, 40) for _ in range(6)],
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
    )
    kept = filter_series(
        observations,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        season_break_weight=1.0,
    )
    assert [e.var_logit for e in kept] == pytest.approx([e.var_logit for e in plain])


# ------------------------------------------------------------- changepoints


def test_a_sustained_role_change_is_flagged_as_a_changepoint():
    observations = [RoleObservation(2, 40)] * 8 + [RoleObservation(20, 40)] * 4
    estimates = filter_series(
        observations,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        changepoint_lookback=2,
        changepoint_threshold_sd=2.5,
    )
    assert any(e.changepoint for e in estimates[8:])


def test_a_stable_player_never_trips_the_changepoint_detector():
    estimates = steady(RoleObservation(10, 40), n=30, changepoint_lookback=2)
    assert not any(e.changepoint for e in estimates)


def test_one_off_shock_is_not_a_changepoint():
    """A single outlier is noise. Treating it as a regime change is how a
    tracker ends up chasing every blowout."""
    observations = [RoleObservation(10, 40)] * 8
    observations[4] = RoleObservation(30, 40)
    estimates = filter_series(
        observations,
        process_noise=0.006,
        prior_mean_logit=logit(0.25),
        prior_var_logit=0.05,
        changepoint_lookback=2,
        changepoint_threshold_sd=2.5,
    )
    assert not any(e.changepoint for e in estimates)


def test_changepoint_makes_the_filter_recover_faster():
    observations = [RoleObservation(2, 40)] * 8 + [RoleObservation(20, 40)] * 5
    quick = filter_series(
        observations,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        changepoint_lookback=2,
        changepoint_threshold_sd=2.0,
    )
    slow = filter_series(
        observations,
        process_noise=0.006,
        prior_mean_logit=0.0,
        prior_var_logit=4.0,
        changepoint_lookback=99,
        changepoint_threshold_sd=2.0,
    )
    assert quick[-1].mean_logit > slow[-1].mean_logit


# ------------------------------------------------------------- calibration


def test_process_noise_and_memory_are_exact_inverses():
    for memory in (2.0, 4.0, 7.5, 12.0):
        for observation_var in (0.05, 0.17, 0.5):
            q = process_noise_for_memory(memory, observation_var)
            assert effective_memory_games(q, observation_var) == pytest.approx(memory)


def test_memory_below_one_game_is_rejected():
    """m <= 1 means the filter forgets faster than it observes, which is not a
    slow filter -- it is a broken one."""
    with pytest.raises(ValueError, match="must exceed one game"):
        process_noise_for_memory(1.0, 0.1)


@pytest.mark.parametrize("layer", sorted(TARGET_MEMORY_GAMES))
def test_calibrated_noise_delivers_the_documented_memory(layer):
    """The numbers in config.toml claim a memory in games. This is the test
    that makes the claim true rather than decorative."""
    successes, trials = REFERENCE_WORKLOAD[layer]
    q = calibrated_process_noise(layer)
    observed = effective_memory_games(q, observation_variance(successes, trials))
    assert observed == pytest.approx(TARGET_MEMORY_GAMES[layer], rel=1e-6)


@pytest.mark.parametrize("layer", sorted(TARGET_MEMORY_GAMES))
def test_config_defaults_match_the_calibration(layer):
    """config.toml is allowed to disagree with the calibration only on purpose.
    If this fails, either the target memory moved or someone hand-edited a
    number back to something uninterpretable."""
    assert RoleTrackerConfig().process_noise(layer) == pytest.approx(
        calibrated_process_noise(layer), rel=5e-3
    )


def test_target_share_is_the_stickiest_layer():
    """Ordering is a football claim, and it should survive re-tuning."""
    assert (
        TARGET_MEMORY_GAMES["target_share"]
        > TARGET_MEMORY_GAMES["carry_share"]
        > TARGET_MEMORY_GAMES["snap_share"]
    )


def test_unknown_layer_names_are_rejected_loudly():
    with pytest.raises(KeyError, match="no reference workload"):
        calibrated_process_noise("yards_share")
    with pytest.raises(KeyError, match="no process noise configured"):
        RoleTrackerConfig().process_noise("yards_share")
