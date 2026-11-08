"""E5 -- does the chosen fix work on the Minecraft stub (m = 6)?

    python experiments/epsilon_scale/e5_minecraft_generality.py [--no-tests]

Applies the same `bridge.epsilon_scale.GuardedEpsilonScale` used on
LunarLander to the Minecraft stub (env/minecraft_stub.py), with the
candidates and baseline of env/minecraft_rollout.py (Idle baseline, horizon 3,
gamma 0.99, as demo/run_metamo_pipeline.py uses them). Certification goes
through the real MetaMoController and RuntimeCertificationPipeline (PDS only,
CVaR off, a fresh pipeline per decision so W_x stays empty).

  * scale: s_r + s_n pooled over every decision of episodes with seeds 1-20;
  * evaluation: every decision of episodes with seeds 21-40;
  * reported for raw epsilon and with the fix: flips between eps = 0 and 0.5,
    admissions over the eps grid, monotonicity, and how often the guard fires;
  * finally runs the Minecraft and bridge test files and prints the counts.

Note: the stub multiplies its rewards by a hand-calibrated _REWARD_SCALE = 0.08
(env/minecraft_stub.py:69-88) so that its margins already sit inside a raw
epsilon band. The fix therefore stacks a second calibration on top of the
first; the numbers below show what that does.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Optional, Sequence

import numpy as np

from common import EPS_FINE, EPS_GRID, REPO_ROOT, parse_common_args, pct, print_header, print_table

CAL_EPISODES = tuple(range(1, 21))
EVAL_EPISODES = tuple(range(21, 41))
M = 6


def episode_decisions(seed: int) -> List[tuple]:
    """(baseline_stats, records) at every decision of one episode (Idle between)."""
    from env.minecraft_rollout import DEFAULT_HORIZON, evaluate_candidates
    from env.minecraft_stub import MinecraftStubEnv

    env = MinecraftStubEnv(seed=seed)
    env.reset(seed=seed)
    out = []
    while not env.episode_over:
        out.append(evaluate_candidates(env, gamma=0.99, horizon=DEFAULT_HORIZON))
        for _ in range(DEFAULT_HORIZON):
            if env.episode_over:
                break
            env.step(0)
    return out


class Certifier:
    def __init__(self) -> None:
        import torch

        from generator.mdn import MotiveDecompositionNetwork

        torch.manual_seed(42)
        self.model = MotiveDecompositionNetwork(input_dim=8, num_objectives=M)
        self.model.eval()

    def admitted(self, baseline_stats, records, epsilon: float, scaler) -> frozenset:
        from bridge.controller import MetaMoController
        from bridge.governor import FakeGovernor
        from bridge.protocol import GovernorSignal
        from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
        from utils.weight_set_store import WeightSetStore

        pipeline = RuntimeCertificationPipeline(
            model=self.model,
            weight_store=WeightSetStore(num_objectives=M),
            config=RuntimePipelineConfig(gate_type="PDS", use_cvar=False, train_support_after_certify=False),
        )
        signal = GovernorSignal(weights=np.full(M, 1.0 / M), pds_epsilon=epsilon, cvar_tail_level=0.1)
        controller = MetaMoController(FakeGovernor([signal]), pipeline, seed=42, epsilon_scale=scaler)
        out = controller.certify(context=np.zeros(8), candidate_skills=records,
                                 baseline_stats=baseline_stats, signal=signal)
        return frozenset(r.skill_id for r in out if r.is_certified)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = parse_common_args(argv, __doc__)
    parser.add_argument("--no-tests", action="store_true", help="skip running the test files")
    args = parser.parse_args(argv)

    from bridge.epsilon_scale import GuardedEpsilonScale, calibrate_epsilon_scale
    from env.minecraft_rollout import APPRAISAL_SCALE_FLOOR

    print_header("E5 -- the guarded epsilon scale on the Minecraft stub (m = 6)")
    cal = [d for s in CAL_EPISODES for d in episode_decisions(s)]
    ev = [(s, i, d) for s in EVAL_EPISODES for i, d in enumerate(episode_decisions(s))]
    pooled = [r for _, recs in cal for r in recs]
    scale = calibrate_epsilon_scale(pooled, floor=APPRAISAL_SCALE_FLOOR)
    s_r = float(np.mean([abs(r.delta_r) for r in pooled]))
    s_n = float(np.mean([np.mean(np.abs(r.delta_n)) for r in pooled]))
    print(f"calibration: {len(cal)} decisions from episodes 1-20; mean|dr|={s_r:.4f}, mean|dn|={s_n:.4f}, "
          f"floor={APPRAISAL_SCALE_FLOOR}; scale = max(s_r,floor)+max(s_n,floor) = {scale:.4f}")
    margins = np.array([r.delta_r + min(r.delta_n) for _, _, (_, recs) in ev for r in recs])
    print(f"evaluation: {len(ev)} decisions x {len(ev[0][2][1])} candidates from episodes 21-40; "
          f"raw margins min={margins.min():.4f} median={np.median(margins):.4f} max={margins.max():.4f}")
    print(f"with the fix, eps in [0, 0.5] maps to a tolerance in [0, {0.5 * scale:.4f}]")

    scaler = GuardedEpsilonScale(scale=scale)
    guard_fires = sum(
        scaler.is_guarded(delta_r=r.delta_r, baseline_payoff=bs["baseline_payoff"])
        for _, _, (bs, recs) in ev for r in recs
    )
    print(f"guard fired (candidate's own task payoff < 0): {guard_fires} times")

    cert = Certifier()
    table = []
    for label, sc in (("raw eps (fix off)", None), ("guarded scale (fix on)", scaler)):
        counts = {e: 0 for e in EPS_GRID}
        flips = mono = pairs = 0
        flip_seeds = set()
        flip_cands = set()
        for seed, _, (bs, recs) in ev:
            verdicts = {e: cert.admitted(bs, recs, e, sc) for e in EPS_GRID}
            for e in EPS_GRID:
                counts[e] += len(verdicts[e])
            pairs += len(recs)
            changed = verdicts[EPS_GRID[-1]] - verdicts[EPS_GRID[0]]
            flips += len(changed)
            if changed:
                flip_seeds.add(seed)
                flip_cands |= changed
            for a, b in zip(EPS_GRID, EPS_GRID[1:]):
                mono += int(not verdicts[a] <= verdicts[b])
        table.append([label] + [f"{counts[e]}/{pairs}" for e in EPS_GRID]
                     + [f"{flips} ({pct(flips, pairs)})", len(flip_seeds), len(flip_cands), mono])
    print()
    print_table(["gate"] + [f"eps={e:g}" for e in EPS_GRID] + ["flips 0->0.5", "seeds", "cands", "mono viol"],
                table)

    # Where the band sits for each, as a share of margins epsilon can decide.
    raw_band = int(np.sum((margins >= -0.5) & (margins < 0)))
    fix_band = int(np.sum((margins >= -0.5 * scale) & (margins < 0)))
    print(f"margins epsilon can decide: raw band [-0.5, 0): {raw_band}/{len(margins)}; "
          f"fixed band [-{0.5 * scale:.4f}, 0): {fix_band}/{len(margins)}")

    if not args.no_tests:
        print_header("Minecraft and bridge test files")
        files = ["tests/test_minecraft_stub.py", "tests/test_minecraft_rollout.py",
                 "tests/test_run_metamo_pipeline.py", "tests/test_bridge_budget.py",
                 "tests/test_bridge_controller.py", "tests/test_bridge_e2e.py",
                 "tests/test_bridge_governor.py", "tests/test_bridge_weights.py",
                 "tests/test_bridge_epsilon_scale.py"]
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        print("command: python -m pytest " + " ".join(files) + " -q -p no:cacheprovider")
        print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-500:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
