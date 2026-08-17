"""Calibration diagnostics.

A model that is accurate on average and wrong about its own uncertainty is
worse than useless for betting, because the entire product is a probability
either side of a half-point. Accuracy metrics cannot see that failure. These
can.

Three diagnostics, each answering a question the others cannot:

* **PIT** — where does the realised value fall in the predicted distribution?
  Uniform means the shape is right. A U shape means the model is overconfident
  (too many outcomes in the tails); a hump means underconfident.
* **Coverage** — do 80% intervals contain the truth 80% of the time? The direct
  operational question, and the one that translates to money.
* **Log score** — proper, so it cannot be gamed by widening intervals until
  coverage looks good. Coverage and log score together catch what either alone
  would miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# How far observed coverage may sit from what a perfectly calibrated model
# would achieve ON THE SAME PREDICTIVES. Because the target is now the
# achievable coverage rather than the nominal level, these are genuine
# tolerances rather than an allowance for the lattice -- see
# `expected_coverage`.
COVERAGE_TOLERANCE = 0.05


@dataclass(frozen=True)
class CalibrationReport:
    pit_uniformity: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    expected_50: float
    expected_80: float
    expected_95: float
    mean_log_score: float
    mean_absolute_error: float
    n: int

    def summary(self) -> str:
        return (
            f"n={self.n}  PIT dev={self.pit_uniformity:.3f}  "
            f"cover {self.coverage_50:.2f}/{self.coverage_80:.2f}/{self.coverage_95:.2f} "
            f"vs achievable {self.expected_50:.2f}/{self.expected_80:.2f}/"
            f"{self.expected_95:.2f}  logscore={self.mean_log_score:.3f}  "
            f"MAE={self.mean_absolute_error:.2f}"
        )

    @property
    def well_calibrated(self) -> bool:
        """Observed coverage close to what these predictives can achieve.

        Compared against `expected_*`, not against the nominal level. On a
        lattice the nominal level is unreachable, and how far short it falls
        depends on the counts: for team plays near 62 the gap is a point or
        two, for receptions near 3 a nominal 50% interval genuinely holds about
        80% of the mass. Judging player-level projections against 0.50 would
        reject a perfect model, and judging them against a loosened constant
        would accept a bad one.
        """
        return (
            abs(self.coverage_50 - self.expected_50) < COVERAGE_TOLERANCE
            and abs(self.coverage_80 - self.expected_80) < COVERAGE_TOLERANCE
            and abs(self.coverage_95 - self.expected_95) < COVERAGE_TOLERANCE
        )


def randomised_pit(cdf_at_y: np.ndarray, cdf_at_y_minus_1: np.ndarray) -> np.ndarray:
    """PIT values for a DISCRETE predictive distribution.

    For counts the naive PIT is not uniform even under a perfect model, because
    the CDF jumps. Randomising within the jump restores uniformity, which is
    what makes the diagnostic readable rather than misleading. Using the
    continuous formula on count data produces a PIT histogram that looks broken
    when nothing is.
    """
    rng = np.random.default_rng(0)
    u = rng.uniform(size=len(cdf_at_y))
    return cdf_at_y_minus_1 + u * (cdf_at_y - cdf_at_y_minus_1)


def pit_deviation(pit: np.ndarray, bins: int = 10) -> float:
    """Mean absolute deviation of the PIT histogram from uniform.

    0 is perfect. Above ~0.20 the shape of the predictive distribution is
    wrong, not just its location.
    """
    counts, _ = np.histogram(np.clip(pit, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    expected = len(pit) / bins
    if expected == 0:
        return float("nan")
    return float(np.mean(np.abs(counts - expected)) / expected)


def coverage(samples: np.ndarray, actual: np.ndarray, level: float) -> float:
    """Share of outcomes inside the central interval of the sampled predictive."""
    lower = np.quantile(samples, (1 - level) / 2, axis=1)
    upper = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def expected_coverage(samples: np.ndarray, level: float) -> float:
    """Coverage a PERFECTLY calibrated model would show on these predictives.

    The nominal level is not achievable on a lattice: an interval between two
    integers includes both endpoints, so it holds strictly more than `level` of
    the mass. How much more depends entirely on how spread out the distribution
    is -- negligible for team plays around 62, enormous for receptions around
    3, where a nominal 50% interval really does contain about 80% of the mass.

    Comparing observed coverage against the nominal level therefore measures
    the lattice, not the model. Comparing it against this instead measures the
    model. Computed from the model's own draws, so it needs no assumption about
    the distribution's family.
    """
    lower = np.quantile(samples, (1 - level) / 2, axis=1)
    upper = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
    inside = (samples >= lower[:, None]) & (samples <= upper[:, None])
    return float(np.mean(inside))


def log_score(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Negative log predictive probability, estimated from draws.

    The probability of the realised integer is read off the sample histogram.
    Floored so a single unlucky outcome with zero sampled mass cannot dominate
    the mean -- the floor is reported behaviour, not a silent fudge: it caps
    the penalty at roughly the resolution the sample size can support.
    """
    draws = samples.shape[1]
    floor = 1.0 / (draws * 10)
    scores = np.empty(len(actual))
    for i, value in enumerate(actual):
        probability = np.mean(samples[i] == value)
        scores[i] = -math.log(max(probability, floor))
    return scores


def assess(samples: np.ndarray, actual: np.ndarray) -> CalibrationReport:
    """Full report from sampled draws and realised outcomes."""
    actual = np.asarray(actual)
    below = np.mean(samples < actual[:, None], axis=1)
    at_or_below = np.mean(samples <= actual[:, None], axis=1)
    pit = randomised_pit(at_or_below, below)

    return CalibrationReport(
        pit_uniformity=pit_deviation(pit),
        coverage_50=coverage(samples, actual, 0.50),
        coverage_80=coverage(samples, actual, 0.80),
        coverage_95=coverage(samples, actual, 0.95),
        expected_50=expected_coverage(samples, 0.50),
        expected_80=expected_coverage(samples, 0.80),
        expected_95=expected_coverage(samples, 0.95),
        mean_log_score=float(np.mean(log_score(samples, actual))),
        mean_absolute_error=float(np.mean(np.abs(np.mean(samples, axis=1) - actual))),
        n=len(actual),
    )
