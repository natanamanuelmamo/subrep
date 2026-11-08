"""Candidate sets for the epsilon-scale experiments.

* `ci`        -- the fixed-action and random candidates (no PPO checkpoint),
                 as in tests/test_lunar_lander_integration.py.
* `default`   -- data_collector.collect_candidate_sets.build_default_candidate_policies.
* `extended`  -- build_extended_candidate_policies (default + two PPO variants).
* `borderline`-- candidates defined HERE ONLY, to probe whether an option lets
                 epsilon decide skills that are neither clear landings nor
                 clear crashes. They are degraded PPO pilots:
                   ppo_greedy_p{10,20,30}   PPO, but a uniformly random action
                                            with probability p each step;
                   ppo_then_noop_{k}        PPO for k steps, then engines off.
* `tradeoff`  -- also defined here only, added after the first E1 run showed
                 the borderline candidates are mostly crash-type: PPO pilots
                 that land but burn extra fuel (ppo_waste_{main,side}_{p}).

Every policy with state exposes `reset(seed=...)`, which
`env.lunar_lander_rollout.run_episode` calls, so each episode replays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from common import REPO_ROOT  # noqa: F401  (puts the repo on sys.path)
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    _fixed_action_policy,
    _random_policy,
    build_default_candidate_policies,
    build_extended_candidate_policies,
)

PILOT_CHECKPOINT = str(REPO_ROOT / "models" / "pilot_ppo.pt")
NUM_ACTIONS = 4


def _ppo_action(pilot, obs) -> int:
    out = pilot.predict(obs, deterministic=True, return_probability=True)
    return int(out[0] if isinstance(out, tuple) else out)


class PpoEpsGreedy:
    """Deterministic PPO, replaced by a random action with probability p."""

    def __init__(self, pilot, p: float) -> None:
        self.pilot = pilot
        self.p = float(p)
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None, **_kwargs) -> None:
        self.rng = np.random.default_rng(0 if seed is None else int(seed) + 7919)

    def __call__(self, obs):
        if self.rng.random() < self.p:
            return int(self.rng.integers(NUM_ACTIONS)), 1.0
        return _ppo_action(self.pilot, obs), 1.0


class PpoThenNoop:
    """Deterministic PPO for `k` steps, then all engines off."""

    def __init__(self, pilot, k: int) -> None:
        self.pilot = pilot
        self.k = int(k)
        self.t = 0

    def reset(self, **_kwargs) -> None:
        self.t = 0

    def __call__(self, obs):
        self.t += 1
        if self.t <= self.k:
            return _ppo_action(self.pilot, obs), 1.0
        return 0, 1.0


class PpoFuelWaste:
    """Deterministic PPO that burns extra fuel when it would coast.

    Whenever PPO picks noop, with probability p it fires instead:
      mode "main" -- the main engine for one step;
      mode "side" -- the left then the right side engine (net torque ~0).
    The pilot still flies the descent, so these candidates mostly land, but at
    a fuel cost: the "lands, but loses on an objective" trade-off that epsilon
    is meant to govern.
    """

    def __init__(self, pilot, p: float, mode: str) -> None:
        if mode not in ("main", "side"):
            raise ValueError(mode)
        self.pilot = pilot
        self.p = float(p)
        self.mode = mode
        self.rng = np.random.default_rng(0)
        self.pending = None

    def reset(self, seed=None, **_kwargs) -> None:
        self.rng = np.random.default_rng((0 if seed is None else int(seed)) + 104729)
        self.pending = None

    def __call__(self, obs):
        if self.pending is not None:
            action, self.pending = self.pending, None
            return action, 1.0
        action = _ppo_action(self.pilot, obs)
        if action == 0 and self.rng.random() < self.p:
            if self.mode == "main":
                return 2, 1.0
            self.pending = 3
            return 1, 1.0
        return action, 1.0


def tradeoff_candidates(env) -> Tuple[CandidatePolicy, ...]:
    from pilot.rl_pilot import RLPilot

    pilot = RLPilot.load(PILOT_CHECKPOINT, map_location="cpu")
    pilot.eval()
    out = []
    for p in (0.12, 0.15, 0.18, 0.20):
        out.append(CandidatePolicy(f"ppo_waste_main_{int(round(p * 100))}", PpoFuelWaste(pilot, p, "main")))
    for p in (0.15, 0.20):
        out.append(CandidatePolicy(f"ppo_waste_side_{int(round(p * 100))}", PpoFuelWaste(pilot, p, "side")))
    return tuple(out)


def ci_candidates(env) -> Tuple[CandidatePolicy, ...]:
    return (
        CandidatePolicy("noop", _fixed_action_policy(0)),
        CandidatePolicy("left_engine", _fixed_action_policy(1)),
        CandidatePolicy("main_engine", _fixed_action_policy(2)),
        CandidatePolicy("right_engine", _fixed_action_policy(3)),
        CandidatePolicy("random", _random_policy(env)),
    )


def borderline_candidates(env) -> Tuple[CandidatePolicy, ...]:
    from pilot.rl_pilot import RLPilot

    pilot = RLPilot.load(PILOT_CHECKPOINT, map_location="cpu")
    pilot.eval()
    out = []
    for p in (0.1, 0.2, 0.3):
        out.append(CandidatePolicy(f"ppo_greedy_p{int(p * 100)}", PpoEpsGreedy(pilot, p)))
    for k in (50, 100, 150):
        out.append(CandidatePolicy(f"ppo_then_noop_{k}", PpoThenNoop(pilot, k)))
    return tuple(out)


def build_candidate_set(name: str, env) -> Tuple[CandidatePolicy, ...]:
    if name == "ci":
        return ci_candidates(env)
    if name == "default":
        return build_default_candidate_policies(env, pilot_checkpoint=PILOT_CHECKPOINT)
    if name == "extended":
        return build_extended_candidate_policies(env, pilot_checkpoint=PILOT_CHECKPOINT)
    if name == "borderline":
        return borderline_candidates(env)
    if name == "tradeoff":
        return tradeoff_candidates(env)
    raise ValueError(f"unknown candidate set {name!r}")
