"""Put MetaMo's PDS budget on the scale of the environment's deltas.

THE PROBLEM
-----------
The PDS gate admits a candidate when

    delta_r + min_i delta_n_i  >=  -epsilon        (certification/pds_test.py:68-85)

and MetaMo sets epsilon in [0, 0.5] (bridge/budget.py:144-160). That only
works when the deltas are of order one. On MO-LunarLander they are in
environment units -- delta_r is 0 or +-100 / +-200, objective changes are a few
hundred -- so no margin falls inside [-0.5, 0] and epsilon never changes an
admission (tests/test_lunar_lander_integration.py,
`test_epsilon_cannot_change_a_lunar_lander_admission`).

WHAT THIS MODULE DOES
---------------------
`GuardedEpsilonScale` converts MetaMo's epsilon into environment units before
the controller stamps it on each candidate record:

    epsilon_env = epsilon * scale           scale = s_r + s_n

where s_r is the typical |delta_r| and s_n the typical |delta_n| (the same
statistics as the governor's appraisal scales), measured once on calibration
episodes. The records' deltas are never touched, so selection
(library/skill_selector.py) and stored certificates keep their meaning.

THE FAILURE GUARD
-----------------
Scaling alone admits crashes. When the baseline itself fails the task (the
LunarLander noop baseline crashes on every seed), a candidate that also fails
has delta_r = 0 and its margin is just its worst objective change -- often a
small fuel cost. Measured on seeds 1-40, the best crash margin was -1.0 while
ten landings scored lower, so no threshold on a rescaled margin can make
epsilon matter AND keep every crash out.

The guard uses the candidate's OWN task payoff, which the controller already
has: baseline_payoff + delta_r. Only candidates whose own payoff is at least
`failure_threshold` get the scaled tolerance; a candidate that fails the task
keeps MetaMo's raw epsilon, i.e. exactly today's gate. MetaMo may then trade
objectives against each other for skills that do the task, but cannot trade a
task failure for anything.

Generic: nothing here depends on the number of objectives, and on an
environment whose task reward is never negative (the Minecraft stub) the guard
never fires. Monotonic: epsilon_env is non-decreasing in epsilon for every
candidate, so lowering epsilon never admits more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# Own task payoff below this counts as a failure. LunarLander: a crash scores
# -100, a timeout 0 and a landing +100.
DEFAULT_FAILURE_THRESHOLD = 0.0


@dataclass(frozen=True)
class GuardedEpsilonScale:
    """Converts a per-step PDS epsilon into environment units, per candidate.

    Attributes:
        scale: environment units per unit of epsilon (> 0). Use
            `calibrate_epsilon_scale` to measure it.
        failure_threshold: candidates whose own task payoff
            (baseline_payoff + delta_r) is below this keep the raw epsilon.
            None disables the guard, so every candidate is scaled.
    """

    scale: float
    failure_threshold: Optional[float] = DEFAULT_FAILURE_THRESHOLD

    def __post_init__(self) -> None:
        if not (np.isfinite(self.scale) and self.scale > 0.0):
            raise ValueError(f"scale must be a positive finite number, got {self.scale}")

    def is_guarded(self, *, delta_r: Optional[float], baseline_payoff: Optional[float]) -> bool:
        """True when the candidate fails the task and so keeps the raw epsilon.

        A candidate whose own payoff cannot be computed (no delta_r or no
        baseline payoff) is treated as guarded: without evidence that it does
        the task, it does not get the wider tolerance.
        """
        if self.failure_threshold is None:
            return False
        if delta_r is None or baseline_payoff is None:
            return True
        return float(baseline_payoff) + float(delta_r) < float(self.failure_threshold)

    def epsilon_for(
        self,
        epsilon: float,
        *,
        delta_r: Optional[float],
        baseline_payoff: Optional[float],
    ) -> float:
        """The epsilon to stamp on one candidate record."""
        epsilon = float(epsilon)
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if self.is_guarded(delta_r=delta_r, baseline_payoff=baseline_payoff):
            return epsilon
        return epsilon * float(self.scale)


def calibrate_epsilon_scale(
    records: Iterable[Any],
    *,
    floor: float = 0.0,
) -> float:
    """s_r + s_n over calibration records, pooled.

    s_r is the mean |delta_r| and s_n the mean over records of the mean
    |delta_n_i| -- the statistics `appraisal_scales` computes in
    env/lunar_lander_rollout.py and env/minecraft_rollout.py -- each at least
    `floor`. Works for any number of objectives.

    Pool the candidates of several calibration seeds: on LunarLander the
    per-seed estimate of s_r + s_n has a coefficient of variation of 0.24 over
    seeds 1-20, while the pooled value moves by at most 5% when any one seed
    is left out (experiments/epsilon_scale/e2_scale_stability.py).
    """
    records = list(records)
    if not records:
        raise ValueError("calibrate_epsilon_scale needs at least one record")
    s_r = float(np.mean([abs(float(r.delta_r)) for r in records])) or 1.0
    s_n = float(np.mean([np.mean(np.abs(np.asarray(r.delta_n, dtype=np.float64))) for r in records])) or 1.0
    floor = float(floor)
    return max(s_r, floor) + max(s_n, floor)


def scaled_epsilons(
    scaler: GuardedEpsilonScale,
    epsilon: float,
    candidates: Sequence[Any],
    baseline_payoff: Optional[float],
) -> list:
    """`scaler.epsilon_for` for every candidate, in order (for logging)."""
    return [
        scaler.epsilon_for(
            epsilon,
            delta_r=getattr(c, "delta_r", None),
            baseline_payoff=baseline_payoff,
        )
        for c in candidates
    ]
