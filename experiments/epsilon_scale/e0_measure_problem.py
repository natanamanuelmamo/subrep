"""E0 -- measure the epsilon-scale problem in raw environment units.

    python experiments/epsilon_scale/e0_measure_problem.py [--refresh] [--no-pairs]

Scores the default and extended candidate sets against the noop baseline with
`env.lunar_lander_rollout.evaluate_candidates`, seeds 1-40, and prints:

  * per candidate-seed pair: delta_r, delta_n[Safety], delta_n[Fuel], the PDS
    margin delta_r + min(delta_n), and landed / crashed / timeout;
  * per candidate: outcome shares and the margin distribution;
  * the margin distribution (min / median / max) and the share inside the
    epsilon band [-0.5, 0];
  * the admission table at epsilon in {0, 0.05, 0.1, 0.2, 0.3, 0.5}.

The default set is a subset of the extended set (same policies, same seeds,
every episode seeded), so the extended set is flown once and the default set
is read from it.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

import numpy as np

from common import (
    ALL_SEEDS,
    EPS_GRID,
    EPS_MAX,
    OBJECTIVES_LUNAR,
    Row,
    collect_rows,
    parse_common_args,
    pct,
    print_header,
    print_table,
    skill_ids,
)

DEFAULT_IDS = (
    "ppo_deterministic",
    "ppo_stochastic",
    "left_engine",
    "main_engine",
    "right_engine",
    "random",
)


def margin_summary(rows: Sequence[Row]) -> List[object]:
    m = np.array([r.raw_margin for r in rows])
    in_band = int(np.sum((m >= -EPS_MAX) & (m <= 0.0)))
    return [
        len(rows),
        float(m.min()),
        float(np.median(m)),
        float(m.max()),
        f"{in_band}/{len(rows)}",
        pct(in_band, len(rows)),
    ]


def admission_table(rows: Sequence[Row]) -> None:
    ids = skill_ids(rows)
    headers = ["candidate"] + [f"eps={e:g}" for e in EPS_GRID] + ["flips 0->0.5"]
    table = []
    total_flips = 0
    for sid in ids:
        sub = [r for r in rows if r.skill_id == sid]
        counts = [sum(r.raw_margin >= -e for r in sub) for e in EPS_GRID]
        flips = sum((r.raw_margin >= -EPS_GRID[-1]) != (r.raw_margin >= -EPS_GRID[0]) for r in sub)
        total_flips += flips
        table.append([sid] + [f"{c}/{len(sub)}" for c in counts] + [flips])
    counts = [sum(r.raw_margin >= -e for r in rows) for e in EPS_GRID]
    table.append(["ALL"] + [f"{c}/{len(rows)}" for c in counts] + [total_flips])
    print_table(headers, table)
    print(f"pairs that change admission between eps=0 and eps=0.5: {total_flips}/{len(rows)} ({pct(total_flips, len(rows))})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = parse_common_args(argv, __doc__)
    parser.add_argument("--no-pairs", action="store_true", help="skip the per-pair table")
    args = parser.parse_args(argv)

    extended = collect_rows("extended", ALL_SEEDS, refresh=args.refresh)
    default = [r for r in extended if r.skill_id in DEFAULT_IDS]

    print_header("E0 -- the epsilon-scale problem in raw units (seeds 1-40, baseline = noop)")
    base = {r.seed: r for r in extended}
    b_land = sum(r.baseline_landed for r in base.values())
    b_crash = sum(r.baseline_crashed for r in base.values())
    print(f"baseline (noop) outcomes over {len(base)} seeds: landed={b_land} crashed={b_crash} "
          f"timeout={len(base) - b_land - b_crash}")
    print(f"objectives: {list(OBJECTIVES_LUNAR)}; margin = delta_r + min_i delta_n_i; "
          f"admit iff margin >= -eps; eps band = [-{EPS_MAX}, 0]")

    if not args.no_pairs:
        print_header("Per candidate-seed pair (extended set)")
        print_table(
            ["seed", "candidate", "dr", "dn_Safety", "dn_Fuel", "margin", "outcome", "steps"],
            [[r.seed, r.skill_id, r.delta_r, r.delta_n[0], r.delta_n[1], r.raw_margin, r.outcome, r.steps]
             for r in extended],
        )

    print_header("Per candidate (seeds 1-40)")
    table = []
    for sid in skill_ids(extended):
        sub = [r for r in extended if r.skill_id == sid]
        m = np.array([r.raw_margin for r in sub])
        table.append([
            sid,
            pct(sum(r.landed for r in sub), len(sub)),
            pct(sum(r.crashed and not r.landed for r in sub), len(sub)),
            pct(sum(r.outcome == "timeout" for r in sub), len(sub)),
            float(np.mean([r.delta_r for r in sub])),
            float(np.mean([r.delta_n[0] for r in sub])),
            float(np.mean([r.delta_n[1] for r in sub])),
            float(m.min()), float(np.median(m)), float(m.max()),
        ])
    print_table(
        ["candidate", "landed", "crashed", "timeout", "mean dr", "mean dn_Saf", "mean dn_Fuel",
         "margin min", "median", "max"],
        table,
    )

    print_header("Margin distribution and share inside the eps band [-0.5, 0]")
    print_table(
        ["set", "pairs", "min", "median", "max", "in band", "share"],
        [["default"] + margin_summary(default), ["extended"] + margin_summary(extended)],
    )
    abs_m = np.abs([r.raw_margin for r in extended])
    print(f"smallest |margin| over all extended pairs: {abs_m.min():.3f}; "
          f"pairs with |margin| < 5: {int(np.sum(abs_m < 5))}; < 50: {int(np.sum(abs_m < 50))}")

    print_header("Admission table, raw units -- default set")
    admission_table(default)
    print_header("Admission table, raw units -- extended set")
    admission_table(extended)
    return 0


if __name__ == "__main__":
    sys.exit(main())
