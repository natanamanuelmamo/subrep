"""E4 -- the real MetaMo -> SubRep loop on MO-LunarLander, before and after the fix.

    python experiments/epsilon_scale/e4_end_to_end.py

The setup is the one in tests/test_lunar_lander_integration.py: the real
MetaMoController (bridge/controller.py), the real certification pipeline (PDS,
use_cvar=False, untrained seeded MDN), and the rollout adapter
(env/lunar_lander_rollout.py). Each step scores every candidate against the
noop baseline, certifies, selects under the governor's weights, flies the
chosen policy (or the baseline when nothing is admitted) and feeds the outcome
back to the governor.

Candidates: the default set plus two fuel-wasting PPO pilots
(experiments/epsilon_scale/candidates.py, tradeoff set).

  Run 1 -- scripted epsilon (FakeGovernor), fixed weights (0.5, 0.5):
     1a. the same start (seed 8) every step, eps 0.5 -> 0.3 -> 0.1 -> 0.0,
         with a fresh pipeline per step, so only epsilon changes;
     1b. a new held-out seed every step (21-40), eps cycling 0.5/0.3/0.1/0.0;
     1c. as 1a with ONE pipeline: on a repeated context the pipeline grows
         the W_x support region from earlier certified weights
         (utils/mdn_runtime_pipeline.py, observe_certified_weight), and the
         gate's worst case is then taken over W_x instead of min_i -- a
         pre-existing effect, independent of epsilon, shown for the record.
  Run 2 -- the real MetaMo governor (env/lunar_lander_metamo.py), seeds 21-30.

Each run is done with the fix off (raw epsilon, today's behaviour) and on
(GuardedEpsilonScale with env.lunar_lander_rollout.EPSILON_SCALE). Printed per
step: epsilon, the weights, every candidate's margin and its tolerance
(epsilon in env units after the fix), the admitted set, the selected skill and
the flown outcome.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from common import parse_common_args, print_header, print_table, seed_everything

CONTROLLER_SEED = 42
WEIGHTS = np.array([0.5, 0.5])
SCRIPT = (0.5, 0.3, 0.1, 0.0)


class CapturingPipeline:
    """Passes through to the real pipeline and keeps the last certified records."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.last: List = []

    def certify_candidate_skills(self, **kwargs):
        self.last = self.inner.certify_candidate_skills(**kwargs)
        return self.last


def make_pipeline():
    import torch

    from generator.mdn import MotiveDecompositionNetwork
    from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
    from utils.weight_set_store import WeightSetStore

    torch.manual_seed(CONTROLLER_SEED)
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    model.eval()
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=2),
        config=RuntimePipelineConfig(gate_type="PDS", use_cvar=False, train_support_after_certify=False),
    )


def build_candidates(env):
    from candidates import build_candidate_set

    default = build_candidate_set("default", env)
    tradeoff = {c.skill_id: c for c in build_candidate_set("tradeoff", env)}
    return tuple(default) + (tradeoff["ppo_waste_side_15"], tradeoff["ppo_waste_main_15"])


def run(governor, seeds: Sequence[int], *, fix: bool, title: str,
        fresh_pipeline_each_step: bool = False) -> Dict[str, list]:
    from bridge.controller import MetaMoController
    from bridge.epsilon_scale import GuardedEpsilonScale
    from env.lunar_lander_rollout import (
        EPSILON_SCALE,
        NOOP_BASELINE,
        episode_outcome,
        evaluate_candidates,
        execute_candidate,
    )
    from env.lunar_lander_wrapper import SubRepEnv

    seed_everything()
    scaler = GuardedEpsilonScale(scale=EPSILON_SCALE) if fix else None
    env = SubRepEnv(seed=seeds[0])
    candidates = build_candidates(env)
    policies = {c.skill_id: c for c in candidates}
    pipeline = CapturingPipeline(make_pipeline())
    controller = MetaMoController(governor, pipeline, seed=CONTROLLER_SEED, epsilon_scale=scaler)

    print_header(f"{title} -- fix {'ON (guarded scale %.1f)' % EPSILON_SCALE if fix else 'OFF (raw epsilon)'}")
    summary = []
    history = {"eps": [], "admitted": []}
    try:
        for step, seed in enumerate(seeds):
            baseline_stats, records = evaluate_candidates(env, candidates, seed=seed)
            if fresh_pipeline_each_step and step > 0:
                # Same context every step: a shared pipeline would grow W_x
                # from the weights certified at earlier steps, and the gate's
                # worst case would move with it (see run 1c). A fresh pipeline
                # keeps W_x empty, so only epsilon changes between steps.
                pipeline.inner = make_pipeline()
            signal = governor.signal()
            flown = {}

            def outcome_for(selected, _seed=seed):
                policy = NOOP_BASELINE if selected is None else policies[selected.skill_id]
                dr, dn, info = execute_candidate(env, policy, seed=_seed)
                flown["info"] = info
                return episode_outcome(dr, dn, info, admitted=selected is not None)

            rec = controller.step(
                context=np.asarray(baseline_stats["info"]["start_obs"]),
                candidate_skills=records,
                baseline_stats=baseline_stats,
                outcome_for=outcome_for,
            )
            certified = {r.skill_id: r for r in pipeline.last}
            admitted = sorted(k for k, r in certified.items() if r.is_certified)
            info = flown["info"]
            result = "landed" if info["landing_success"] else ("crashed" if info["crashed"] else "timeout")
            print()
            print(f"step {step}  seed {seed}  eps={signal.pds_epsilon:.4f}  "
                  f"weights=[{signal.weights[0]:.3f}, {signal.weights[1]:.3f}]  "
                  f"baseline payoff={baseline_stats['baseline_payoff']:+.0f}")
            rows = []
            for r in records:
                margin = float(r.delta_r) + float(np.min(r.delta_n))
                tol = (signal.pds_epsilon if scaler is None else scaler.epsilon_for(
                    signal.pds_epsilon, delta_r=r.delta_r, baseline_payoff=baseline_stats["baseline_payoff"]))
                outcome = ("landed" if r.metadata["landing_success"]
                           else "crashed" if r.metadata["crashed"] else "timeout")
                rows.append([r.skill_id, outcome, margin, tol, margin + tol, r.skill_id in admitted])
            print_table(["candidate", "outcome", "margin", "tolerance eps_env", "margin+tol", "admitted"], rows)
            print(f"admitted: {admitted or '-'}   selected: {rec.selected_skill_id or '- (fly baseline)'}   "
                  f"flown outcome: {result}")
            summary.append([step, seed, signal.pds_epsilon, len(admitted), ", ".join(admitted) or "-",
                            rec.selected_skill_id or "-", result])
            history["eps"].append(signal.pds_epsilon)
            history["admitted"].append(tuple(admitted))
    finally:
        env.close()
    print()
    print("summary:")
    print_table(["step", "seed", "eps", "#adm", "admitted", "selected", "flown"], summary)
    return history


def scripted(epsilons):
    from bridge.governor import FakeGovernor
    from bridge.protocol import GovernorSignal

    return FakeGovernor([GovernorSignal(weights=WEIGHTS, pds_epsilon=e, cvar_tail_level=0.1) for e in epsilons])


def real_metamo(first_seed: int):
    from env.lunar_lander_metamo import make_lunar_lander_governor
    from env.lunar_lander_rollout import appraisal_scales, evaluate_candidates
    from env.lunar_lander_wrapper import SubRepEnv

    env = SubRepEnv(seed=first_seed)
    try:
        _, records = evaluate_candidates(env, build_candidates(env), seed=first_seed)
    finally:
        env.close()
    payoff_scale, motive_scale = appraisal_scales(records)
    print(f"real MetaMo governor: appraisal scales from seed {first_seed}: "
          f"payoff_scale={payoff_scale:.1f} motive_scale={motive_scale:.1f}")
    return make_lunar_lander_governor(payoff_scale=payoff_scale, motive_scale=motive_scale)


def changes(history) -> int:
    adm = history["admitted"]
    return sum(int(a != b) for a, b in zip(adm, adm[1:]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parse_common_args(argv, __doc__).parse_args(argv)
    from bridge._loader import is_available

    out = {}
    print_header("E4 run 1a -- scripted epsilon, same start every step (seed 8), fresh pipeline per step")
    for fix in (False, True):
        out[("1a", fix)] = run(scripted(SCRIPT), [8] * len(SCRIPT), fix=fix, title="Run 1a",
                               fresh_pipeline_each_step=True)
    print_header("E4 run 1b -- scripted epsilon cycling 0.5/0.3/0.1/0.0, a new held-out seed every step (21-40)")
    seeds_1b = list(range(21, 41))
    eps_1b = [SCRIPT[i % len(SCRIPT)] for i in range(len(seeds_1b))]
    for fix in (False, True):
        out[("1b", fix)] = run(scripted(eps_1b), seeds_1b, fix=fix, title="Run 1b")
    print_header("E4 run 1c -- as 1a but ONE pipeline for all steps (W_x grows on the repeated context)")
    for fix in (False, True):
        out[("1c", fix)] = run(scripted(SCRIPT), [8] * len(SCRIPT), fix=fix, title="Run 1c")

    if is_available():
        print_header("E4 run 2 -- the real MetaMo governor, seeds 21-30")
        seeds_2 = list(range(21, 31))
        for fix in (False, True):
            out[("2", fix)] = run(real_metamo(seeds_2[0]), seeds_2, fix=fix, title="Run 2")
    else:
        print("MetaMo checkout not found -- run 2 skipped")

    print_header("E4 summary -- does the admitted set change with epsilon?")
    rows = []
    for (name, fix), hist in out.items():
        eps = hist["eps"]
        rows.append([name, "on" if fix else "off", f"{min(eps):.3f}..{max(eps):.3f}",
                     changes(hist), len(eps) - 1,
                     f"{sum(len(a) for a in hist['admitted'])}"])
    print_table(["run", "fix", "eps range", "admitted-set changes between steps", "step pairs",
                 "total admissions"], rows)

    print()
    print("fix on vs fix off, step by step (same seeds, same governor script):")
    rows = []
    for name in ("1a", "1b", "1c", "2"):
        if (name, False) not in out:
            continue
        off, on = out[(name, False)], out[(name, True)]
        n = min(len(off["admitted"]), len(on["admitted"]))
        diff = [i for i in range(n) if off["admitted"][i] != on["admitted"][i]]
        rows.append([name, f"{len(diff)}/{n}", ", ".join(str(i) for i in diff) or "-"])
    print_table(["run", "steps whose admitted set differs", "steps"], rows)
    off_1a = out[("1a", False)]["admitted"]
    on_1a = out[("1a", True)]["admitted"]
    print(f"run 1a (only epsilon varies): fix off -> {len(set(off_1a))} distinct admitted set(s); "
          f"fix on -> {len(set(on_1a))} distinct admitted sets over eps {list(SCRIPT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
