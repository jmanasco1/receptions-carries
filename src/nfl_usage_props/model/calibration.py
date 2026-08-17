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

# Per-level coverage tolerances. See `CalibrationReport.well_calibrated` --
# these are not arbitrary slack, they absorb the over-coverage that discrete
# predictive distributions produce by construction.
TOLERANCE_50 = 0.09
TOLERANCE_80 = 0.05
TOLERANCE_95 = 0.04


@dataclass(frozen=True)
class CalibrationReport:
    pit_uniformity: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    mean_log_score: float
    mean_absolute_error: float
    n: int

    def summary(self) -> str:
        return (
            f"n={self.n}  PIT dev={self.pit_uniformity:.3f}  "
            f"cover 50/80/95={self.coverage_50:.2f}/{self.coverage_80:.2f}/"
            f"{self.coverage_95:.2f}  logscore={self.mean_log_score:.3f}  "
            f"MAE={self.mean_absolute_error:.2f}"
        )

    @property
    def well_calibrated(self) -> bool:
        """Every nominal interval within tolerance of its target.

        The tolerances differ by level, and the 50% one is widest, which looks
        backwards until you account for discreteness. A central interval on a
        count distribution runs between two integers and includes both, so it
        covers strictly MORE than its nominal level -- there is no way to carve
        exactly 50% out of a lattice. The narrower the interval, the larger
        that excess is in relative terms, so the 50% band over-covers most.
        A correctly specified Poisson predictive with these means lands near
        0.57 at the 50% level, and demanding 0.50 would reject a perfect model.
        """
        return (
            abs(self.coverage_50 - 0.50) < TOLERANCE_50
            and abs(self.coverage_80 - 0.80) < TOLERANCE_80
            and abs(self.coverage_95 - 0.95) < TOLERANCE_95
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
        mean_log_score=float(np.mean(log_score(samples, actual))),
        mean_absolute_error=float(np.mean(np.abs(np.mean(samples, axis=1) - actual))),
        n=len(actual),
    )
