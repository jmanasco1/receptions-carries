"""Stage 7: turning a price into a fair probability, and a fair probability
into an edge.

Deliberately separate from `model/`. The model says what it thinks will happen;
this says what the market thinks and where the two disagree. Keeping them apart
means a change to the devig cannot quietly alter a projection, and the edge
calculation can be tested against hand-worked prices with no model in sight.
"""

from nfl_usage_props.edge.devig import (
    DEVIG_METHODS,
    consensus_probability,
    devig_two_way,
    multiplicative_devig,
    power_devig,
    shin_devig,
)

__all__ = [
    "DEVIG_METHODS",
    "consensus_probability",
    "devig_two_way",
    "multiplicative_devig",
    "power_devig",
    "shin_devig",
]
