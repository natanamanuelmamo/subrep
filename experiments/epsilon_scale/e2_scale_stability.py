"""E2 -- scale stability: fixed (calibration) vs per-step scales.

    python experiments/epsilon_scale/e2_scale_stability.py [--refresh]

Prints:
  * per-seed scale estimates on the calibration seeds (task's extended set),
    with mean / std / coefficient of variation (criterion 5: CV <= 0.25);
  * per-step scales on the evaluation seeds, for each candidate set, showing
    that per-step scales move with whichever candidates are in the set;
  * leave-one-seed-out sensitivity of the fixed scales and of the admissions;
  * per option: agreement between fixed and per-step admissions, and the
    criterion-5 outcome-class agreement at eps = 0.1.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import numpy as np

from analysis import evaluate, fixed_scales, load_sets, outcome_class_agreement
from common import CAL_SEEDS, EPS_GRID, EPS_NEUTRAL, EVAL_SEEDS, group_by_seed, parse_common_args, pct, print_header, print_table
from options import LUNAR_SCALE_FLOOR, build_options, compute_scales


def cv(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.std(v) / np.mean(v)) if np.mean(v) else float("nan")


def per_seed_scales(rows, seeds):
    out = []
    for seed, sub in sorted(group_by_seed([r for r in rows if r.seed in set(seeds)]).items()):
        out.append((seed, compute_scales([r.delta_r for r in sub], [r.delta_n for r in sub], floor=LUNAR_SCALE_FLOOR)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_common_args(argv, __doc__).parse_args(argv)
    sets = load_sets(args.refresh)
    fixed = fixed_scales(sets["task"])
    options = build_options()

    print_header("E2 -- scale stability")
    print(f"fixed scales (seeds 1-20, extended set, pooled): {fixed.as_text()}")

    print_header("Per-seed scale estimates on the calibration seeds (task's extended set)")
    est = per_seed_scales(sets["task"], CAL_SEEDS)
    print_table(["seed", "s_r", "s_n", "s_Safety", "s_Fuel"],
                [[s, sc.s_r, sc.s_n, sc.s_vec[0], sc.s_vec[1]] for s, sc in est])
    cols = {
        "s_r": [sc.s_r for _, sc in est],
        "s_n": [sc.s_n for _, sc in est],
        "s_r+s_n": [sc.s_r + sc.s_n for _, sc in est],
        "s_Safety": [sc.s_vec[0] for _, sc in est],
        "s_Fuel": [sc.s_vec[1] for _, sc in est],
    }
    print()
    print_table(["scale", "mean", "std", "min", "max", "CV", "CV <= 0.25"],
                [[k, float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)), cv(v), cv(v) <= 0.25]
                 for k, v in cols.items()])

    print_header("Per-step scales on the evaluation seeds, by candidate set")
    table = []
    for name in ("task", "task+bl", "all"):
        est = per_seed_scales(sets[name], EVAL_SEEDS)
        for key, fn in (("s_n", lambda sc: sc.s_n), ("s_r+s_n", lambda sc: sc.s_r + sc.s_n)):
            v = [fn(sc) for _, sc in est]
            table.append([name, key, float(np.mean(v)), float(np.min(v)), float(np.max(v)), cv(v)])
    print_table(["candidate set", "scale", "mean", "min", "max", "CV"], table)
    print("(the same seeds give different per-step scales depending on which candidates are certified together)")

    print_header("Leave-one-seed-out: fixed scales and eval admissions (all candidates, eps grid)")
    loso = []
    for left in CAL_SEEDS:
        keep = [s for s in CAL_SEEDS if s != left]
        loso.append((left, fixed_scales(sets["task"], seeds=keep)))
    s_n = [sc.s_n for _, sc in loso]
    s_rn = [sc.s_r + sc.s_n for _, sc in loso]
    print(f"s_n over 20 leave-one-out fits: min={min(s_n):.2f} max={max(s_n):.2f} (full fit {fixed.s_n:.2f})")
    print(f"s_r+s_n over 20 leave-one-out fits: min={min(s_rn):.2f} max={max(s_rn):.2f} (full fit {fixed.s_r + fixed.s_n:.2f})")
    rows_all = sets["all"]
    table = []
    for opt in options:
        if opt.scale_source != "fixed":
            continue
        base = evaluate(opt, rows_all, fixed)
        worst = 0
        for _, sc in loso:
            alt = evaluate(opt, rows_all, sc)
            changed = sum(
                int((a >= -e) != (b >= -e))
                for e in EPS_GRID
                for ra, rb in zip(base, alt)
                for a, b in zip(ra.mu, rb.mu)
            )
            worst = max(worst, changed)
        total = len(EPS_GRID) * sum(len(r.rows) for r in base)
        table.append([opt.key, f"{worst}/{total}", pct(worst, total)])
    print_table(["option", "max verdicts changed by dropping one cal seed", "share"], table)

    print_header("Fixed vs per-step admissions, and outcome-class agreement at eps=0.1 (all candidates, seeds 21-40)")
    table = []
    by_key = {o.key: o for o in options}
    for opt in options:
        res = evaluate(opt, rows_all, fixed)
        agree, total = outcome_class_agreement(res, EPS_NEUTRAL)
        fixed_vs_step = "-"
        if opt.scale_source == "fixed":
            twin = by_key.get(opt.key.replace("fixed", "per-step"))
            if twin is not None:
                res_s = evaluate(twin, rows_all, fixed)
                same = sum(int((a >= -e) == (b >= -e)) for e in EPS_GRID
                           for ra, rb in zip(res, res_s) for a, b in zip(ra.mu, rb.mu))
                tot = len(EPS_GRID) * sum(len(r.rows) for r in res)
                fixed_vs_step = pct(same, tot)
        table.append([opt.key, f"{agree}/{total}", pct(agree, total), agree / total >= 0.90, fixed_vs_step])
    print_table(["option", "outcome-class agree", "share", ">= 90%", "fixed==per-step (grid)"], table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
