"""Tests for bridge/epsilon_scale.py and the controller's `epsilon_scale` hook.

Certification runs through the real `RuntimeCertificationPipeline` (PDS only,
CVaR off) driven by `FakeGovernor`, so no MetaMo checkout is needed. The
LunarLander-sized records mirror what experiments/epsilon_scale measured:
the noop baseline crashes (payoff -100), a landing scores delta_r = +200, and
a crash against the crashing baseline scores delta_r = 0 with a small
objective loss. The m = 6 tests use the real Minecraft stub.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest

from bridge.controller import MetaMoController
from bridge.epsilon_scale import (
    GuardedEpsilonScale,
    calibrate_epsilon_scale,
    scaled_epsilons,
)
from bridge.governor import FakeGovernor
from bridge.protocol import GovernorSignal
from utils.mdn_contracts import CandidateSkillRecord

EPS_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
EPS_SWEEP = tuple(float(e) for e in np.linspace(0.0, 0.5, 26))
CONTEXT_DIM = 8

# LunarLander-sized fixture: the fixed scale measured on seeds 1-20.
LUNAR_SCALE = 300.0
LUNAR_BASELINE_PAYOFF = -100.0  # the noop baseline crashed


def record(skill_id: str, delta_r: float, delta_n: Sequence[float]) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=float(delta_r),
        delta_n=tuple(float(v) for v in delta_n),
        is_certified=False,
        gate_type="PDS",
    )


def lunar_records() -> List[CandidateSkillRecord]:
    return [
        # Lands, objectives fine: margin +100.
        record("clean_landing", 200.0, (300.0, -100.0)),
        # Lands, but burns a lot of fuel: margin -40, the trade-off epsilon
        # is meant to decide.
        record("fuel_heavy_landing", 200.0, (250.0, -240.0)),
        # Crashes like the baseline but burns almost nothing: margin -1.0,
        # the best crash margin measured on seeds 1-40.
        record("cheap_crash", 0.0, (5.0, -1.0)),
        # Crashes and loses a lot: margin -400.
        record("costly_crash", 0.0, (-400.0, -60.0)),
    ]


def make_pipeline(num_objectives: int):
    import torch

    from generator.mdn import MotiveDecompositionNetwork
    from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
    from utils.weight_set_store import WeightSetStore

    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=CONTEXT_DIM, num_objectives=num_objectives)
    model.eval()
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=num_objectives),
        config=RuntimePipelineConfig(gate_type="PDS", use_cvar=False, train_support_after_certify=False),
    )


def admitted(
    records: Sequence[CandidateSkillRecord],
    epsilon: float,
    *,
    scaler: Optional[GuardedEpsilonScale],
    baseline_payoff: float,
    num_objectives: int,
) -> frozenset:
    """Certify `records` once at `epsilon`; return the admitted skill ids."""
    signal = GovernorSignal(
        weights=np.full(num_objectives, 1.0 / num_objectives),
        pds_epsilon=epsilon,
        cvar_tail_level=0.1,
    )
    controller = MetaMoController(
        FakeGovernor([signal]), make_pipeline(num_objectives), seed=0, epsilon_scale=scaler
    )
    out = controller.certify(
        context=np.zeros(CONTEXT_DIM),
        candidate_skills=list(records),
        baseline_stats={
            "baseline_payoff": baseline_payoff,
            "baseline_motives": np.zeros(num_objectives),
        },
        signal=signal,
    )
    return frozenset(r.skill_id for r in out if r.is_certified)


def lunar_admitted(epsilon: float, scaler: Optional[GuardedEpsilonScale]) -> frozenset:
    return admitted(
        lunar_records(), epsilon, scaler=scaler,
        baseline_payoff=LUNAR_BASELINE_PAYOFF, num_objectives=2,
    )


# ---------------------------------------------------------------------------
# GuardedEpsilonScale on its own
# ---------------------------------------------------------------------------


def test_epsilon_is_scaled_for_a_candidate_that_does_the_task():
    scaler = GuardedEpsilonScale(scale=300.0)
    assert scaler.epsilon_for(0.1, delta_r=200.0, baseline_payoff=-100.0) == pytest.approx(30.0)
    # A timeout against a crashing baseline: own payoff 0 is not a failure.
    assert scaler.epsilon_for(0.1, delta_r=100.0, baseline_payoff=-100.0) == pytest.approx(30.0)


def test_a_candidate_that_fails_the_task_keeps_the_raw_epsilon():
    scaler = GuardedEpsilonScale(scale=300.0)
    assert scaler.epsilon_for(0.5, delta_r=0.0, baseline_payoff=-100.0) == 0.5
    # Unknown payoff: no evidence the task is done, so no wider tolerance.
    assert scaler.epsilon_for(0.5, delta_r=None, baseline_payoff=-100.0) == 0.5
    assert scaler.epsilon_for(0.5, delta_r=0.0, baseline_payoff=None) == 0.5


def test_the_guard_can_be_disabled():
    scaler = GuardedEpsilonScale(scale=300.0, failure_threshold=None)
    assert scaler.epsilon_for(0.5, delta_r=0.0, baseline_payoff=-100.0) == pytest.approx(150.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError):
        GuardedEpsilonScale(scale=bad)


def test_negative_epsilon_is_rejected():
    with pytest.raises(ValueError):
        GuardedEpsilonScale(scale=1.0).epsilon_for(-0.1, delta_r=1.0, baseline_payoff=0.0)


@pytest.mark.parametrize("m", [2, 6])
def test_calibration_is_s_r_plus_s_n_at_any_m(m):
    rng = np.random.default_rng(m)
    recs = [record(f"s{i}", rng.normal(0, 50), rng.normal(0, 100, size=m)) for i in range(30)]
    s_r = np.mean([abs(r.delta_r) for r in recs])
    s_n = np.mean([np.mean(np.abs(r.delta_n)) for r in recs])
    assert calibrate_epsilon_scale(recs) == pytest.approx(s_r + s_n)
    assert calibrate_epsilon_scale(recs, floor=1e6) == pytest.approx(2e6)


def test_calibration_needs_records():
    with pytest.raises(ValueError):
        calibrate_epsilon_scale([])


def test_scaled_epsilons_lists_one_value_per_candidate():
    scaler = GuardedEpsilonScale(scale=LUNAR_SCALE)
    values = scaled_epsilons(scaler, 0.1, lunar_records(), LUNAR_BASELINE_PAYOFF)
    assert values == pytest.approx([30.0, 30.0, 0.1, 0.1])


# ---------------------------------------------------------------------------
# Through the controller and the real PDS pipeline, m = 2
# ---------------------------------------------------------------------------


def test_without_a_scale_epsilon_changes_nothing():
    """The default path: margins are in environment units, epsilon <= 0.5."""
    verdicts = {lunar_admitted(e, None) for e in EPS_GRID}
    assert verdicts == {frozenset({"clean_landing"})}


def test_epsilon_flips_a_borderline_admission_with_the_scale():
    scaler = GuardedEpsilonScale(scale=LUNAR_SCALE)
    # margin -40 needs epsilon_env >= 40, i.e. epsilon >= 0.1333.
    assert "fuel_heavy_landing" not in lunar_admitted(0.0, scaler)
    assert "fuel_heavy_landing" not in lunar_admitted(0.1, scaler)
    assert "fuel_heavy_landing" in lunar_admitted(0.2, scaler)
    assert "fuel_heavy_landing" in lunar_admitted(0.5, scaler)


@pytest.mark.parametrize("epsilon", EPS_GRID)
def test_crashes_stay_rejected_at_every_epsilon(epsilon):
    scaler = GuardedEpsilonScale(scale=LUNAR_SCALE)
    got = lunar_admitted(epsilon, scaler)
    assert "cheap_crash" not in got
    assert "costly_crash" not in got
    assert "clean_landing" in got


def test_without_the_guard_scaling_would_admit_a_crash():
    """Why the guard exists: the -1.0 crash passes once epsilon * scale >= 1."""
    unguarded = GuardedEpsilonScale(scale=LUNAR_SCALE, failure_threshold=None)
    assert "cheap_crash" in lunar_admitted(0.05, unguarded)


def test_admissions_are_monotonic_in_epsilon():
    scaler = GuardedEpsilonScale(scale=LUNAR_SCALE)
    previous = frozenset()
    for epsilon in EPS_SWEEP:
        current = lunar_admitted(epsilon, scaler)
        assert previous <= current, f"lowering epsilon admitted more below {epsilon}"
        previous = current


def test_selection_scores_the_unchanged_deltas():
    """The scale only changes epsilon; records keep their environment units."""
    scaler = GuardedEpsilonScale(scale=LUNAR_SCALE)
    signal = GovernorSignal(weights=np.array([0.5, 0.5]), pds_epsilon=0.5, cvar_tail_level=0.1)
    controller = MetaMoController(FakeGovernor([signal]), make_pipeline(2), seed=0, epsilon_scale=scaler)
    records = controller.certify(
        context=np.zeros(CONTEXT_DIM),
        candidate_skills=lunar_records(),
        baseline_stats={"baseline_payoff": LUNAR_BASELINE_PAYOFF, "baseline_motives": np.zeros(2)},
        signal=signal,
    )
    by_id = {r.skill_id: r for r in records}
    assert by_id["fuel_heavy_landing"].delta_n == (250.0, -240.0)
    assert by_id["fuel_heavy_landing"].epsilon == pytest.approx(150.0)
    assert controller.select(records, signal)[0] == "clean_landing"


def test_step_records_the_scale_and_defaults_to_none():
    signal = GovernorSignal(weights=np.array([0.5, 0.5]), pds_epsilon=0.1, cvar_tail_level=0.1)
    from bridge.protocol import SkillOutcome

    def outcome_for(_selected):
        return SkillOutcome(delta_r=0.0, delta_n=np.zeros(2), admitted=False, effort=0.0)

    kwargs = dict(
        context=np.zeros(CONTEXT_DIM),
        candidate_skills=lunar_records(),
        baseline_stats={"baseline_payoff": LUNAR_BASELINE_PAYOFF, "baseline_motives": np.zeros(2)},
        outcome_for=outcome_for,
    )
    plain = MetaMoController(FakeGovernor([signal]), make_pipeline(2), seed=0).step(**kwargs)
    scaled = MetaMoController(
        FakeGovernor([signal]), make_pipeline(2), seed=0,
        epsilon_scale=GuardedEpsilonScale(scale=LUNAR_SCALE),
    ).step(**kwargs)
    assert plain.epsilon_scale is None
    assert scaled.epsilon_scale == pytest.approx(LUNAR_SCALE)


# ---------------------------------------------------------------------------
# m = 6: the Minecraft stub
# ---------------------------------------------------------------------------


def minecraft_steps() -> List[tuple]:
    """(baseline_stats, records) at every decision of one noiseless episode."""
    from env.minecraft_rollout import DEFAULT_HORIZON, evaluate_candidates
    from env.minecraft_stub import MinecraftStubEnv

    env = MinecraftStubEnv(seed=42, noise_scale=0.0)
    env.reset(seed=42)
    steps = []
    while not env.episode_over:
        steps.append(evaluate_candidates(env, gamma=0.99, horizon=DEFAULT_HORIZON))
        for _ in range(DEFAULT_HORIZON):
            env.step(0)
    return steps


@pytest.fixture(scope="module")
def mc_steps():
    return minecraft_steps()


def test_minecraft_guard_never_fires(mc_steps):
    scaler = GuardedEpsilonScale(scale=1.0)
    for baseline_stats, records in mc_steps:
        for r in records:
            assert not scaler.is_guarded(
                delta_r=r.delta_r, baseline_payoff=baseline_stats["baseline_payoff"]
            )


def test_epsilon_still_changes_minecraft_admissions_and_stays_monotonic(mc_steps):
    from env.minecraft_rollout import APPRAISAL_SCALE_FLOOR

    pooled = [r for _, records in mc_steps for r in records]
    scaler = GuardedEpsilonScale(scale=calibrate_epsilon_scale(pooled, floor=APPRAISAL_SCALE_FLOOR))
    flips = 0
    for baseline_stats, records in mc_steps:
        previous = frozenset()
        verdicts = []
        for epsilon in EPS_GRID:
            current = admitted(
                records, epsilon, scaler=scaler,
                baseline_payoff=baseline_stats["baseline_payoff"], num_objectives=6,
            )
            assert previous <= current
            previous = current
            verdicts.append(current)
        flips += len(verdicts[-1] - verdicts[0])
    assert flips >= 1
