"""Shared evaluation logic for E1-E3.

An "evaluation" applies one option to every seed's candidate set and returns,
per seed, the candidates' normalised margins `mu` (admit iff mu >= -eps).
Fixed scales are measured once, pooled over the calibration seeds 1-20 of the
task's extended candidate set -- the same pooling as
env.lunar_lander_rollout.appraisal_scales. Per-step scales are measured on
the candidate set being certified at that seed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (
    ALL_SEEDS,
    CAL_SEEDS,
    EPS_FINE,
    EPS_GRID,
    EPS_MAX,
    EPS_NEUTRAL,
    EVAL_SEEDS,
    Row,
    collect_rows,
    group_by_seed,
)
from options import (
    LUNAR_SCALE_FLOOR,
    Option,
    Scales,
    StepInput,
    compute_scales,
    scales_for,
    step_inputs,
)


def load_sets(refresh: bool = False) -> Dict[str, List[Row]]:
    """The candidate sets compared everywhere.

    * `task`     -- the extended set (default + two PPO variants);
    * `task+bl`  -- the extended set with the borderline candidates added to
                    the same candidate set on each seed;
    * `all`      -- task + borderline + tradeoff (fuel-wasting PPO) candidates.
    """
    extended = collect_rows("extended", ALL_SEEDS, refresh=refresh, verbose=False)
    borderline = collect_rows("borderline", ALL_SEEDS, refresh=refresh, verbose=False)
    tradeoff = collect_rows("tradeoff", ALL_SEEDS, refresh=refresh, verbose=False)
    return {
        "task": extended,
        "task+bl": extended + borderline,
        "all": extended + borderline + tradeoff,
    }


def fixed_scales(task_rows: Sequence[Row], seeds: Sequence[int] = CAL_SEEDS,
                 floor: float = LUNAR_SCALE_FLOOR) -> Scales:
    cal = [r for r in task_rows if r.seed in set(seeds)]
    return compute_scales([r.delta_r for r in cal], [r.delta_n for r in cal], floor=floor)


@dataclass
class SeedResult:
    seed: int
    rows: List[Row]
    mu: np.ndarray
    scales: Optional[Scales]
    step: StepInput


def evaluate(option: Option, rows: Sequence[Row], fixed: Scales,
             seeds: Sequence[int] = EVAL_SEEDS, floor: float = LUNAR_SCALE_FLOOR) -> List[SeedResult]:
    by_seed = group_by_seed([r for r in rows if r.seed in set(seeds)])
    steps = step_inputs(by_seed)
    out = []
    for seed in sorted(by_seed):
        step = steps[seed]
        sc = scales_for(option, step, fixed, floor)
        out.append(SeedResult(seed, by_seed[seed], np.asarray(option.mu(step, sc), dtype=np.float64), sc, step))
    return out


@dataclass
class Metrics:
    pairs: int
    flips: int
    flip_seeds: int
    flip_candidates: Counter
    flip_outcomes: Counter
    crash_pairs: int
    crash_admitted_max_eps: int     # crashing pairs admitted at eps = 0.5
    crash_admitted_any: int         # crashing pairs admitted at any grid eps
    det_landed: int
    det_landed_admitted_neutral: int
    mono_violations: int
    admitted_by_eps: Dict[float, int]

    @property
    def flip_share(self) -> float:
        return self.flips / self.pairs if self.pairs else 0.0

    def crit1(self) -> bool:
        return self.flip_share >= 0.10 and self.flip_seeds >= 5 and len(self.flip_candidates) >= 2

    def crit2(self) -> bool:
        return self.crash_admitted_any == 0

    def crit3(self) -> bool:
        return self.det_landed > 0 and self.det_landed_admitted_neutral / self.det_landed >= 0.95

    def crit4(self) -> bool:
        return self.mono_violations == 0


def is_crash(r: Row) -> bool:
    return r.crashed and not r.landed


def metrics(results: Sequence[SeedResult], only: Optional[Sequence[str]] = None) -> Metrics:
    """Criteria 1-4 over `results`; `only` restricts the counted candidates."""
    pairs = flips = 0
    flip_seeds = set()
    flip_cands: Counter = Counter()
    flip_out: Counter = Counter()
    crash_pairs = crash_max = crash_any = 0
    det = det_ok = 0
    mono = 0
    adm_by_eps = {e: 0 for e in EPS_GRID}
    for res in results:
        for r, mu in zip(res.rows, res.mu):
            if only is not None and r.skill_id not in only:
                continue
            pairs += 1
            a0, a5 = mu >= -EPS_GRID[0], mu >= -EPS_MAX
            for e in EPS_GRID:
                adm_by_eps[e] += int(mu >= -e)
            if a0 != a5:
                flips += 1
                flip_seeds.add(res.seed)
                flip_cands[r.skill_id] += 1
                flip_out[r.outcome] += 1
            if is_crash(r):
                crash_pairs += 1
                crash_max += int(a5)
                crash_any += int(any(mu >= -e for e in EPS_GRID))
            if r.skill_id == "ppo_deterministic" and r.landed:
                det += 1
                det_ok += int(mu >= -EPS_NEUTRAL)
            # Monotonicity over the fine sweep: admitted(eps) must never shrink
            # as eps grows. Checked numerically, not assumed.
            prev = False
            for e in EPS_FINE:
                now = bool(mu >= -e)
                if prev and not now:
                    mono += 1
                prev = now
    return Metrics(pairs, flips, len(flip_seeds), flip_cands, flip_out, crash_pairs, crash_max,
                   crash_any, det, det_ok, mono, adm_by_eps)


def admitted_sets(results: Sequence[SeedResult], eps: float) -> Dict[int, Tuple[str, ...]]:
    return {res.seed: tuple(r.skill_id for r, mu in zip(res.rows, res.mu) if mu >= -eps)
            for res in results}


def outcome_class_agreement(results: Sequence[SeedResult], eps: float = EPS_NEUTRAL) -> Tuple[int, int]:
    """Criterion 5: pairs agreeing with the majority admission of their
    (candidate, outcome class) group. Returns (agreeing, total)."""
    groups: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
    for res in results:
        for r, mu in zip(res.rows, res.mu):
            groups[(r.skill_id, r.outcome)].append(bool(mu >= -eps))
    agree = total = 0
    for vals in groups.values():
        k = sum(vals)
        agree += max(k, len(vals) - k)
        total += len(vals)
    return agree, total
