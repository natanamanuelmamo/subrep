"""E1 -- does epsilon start to matter under each option?

    python experiments/epsilon_scale/e1_option_comparison.py [--refresh]

Fixed scales are calibrated on seeds 1-20 (task's extended set, pooled);
everything below is measured on the held-out seeds 21-40.

For every option (A-H, fixed and per-step scales where they apply) it prints,
for the task's extended candidate set and for that set with the borderline
candidates added:

  * the flips between eps = 0 and eps = 0.5 (count, share, seeds, candidates);
  * criterion 2: crashing pairs admitted at any eps in the grid;
  * criterion 3: ppo_deterministic landings admitted at eps = 0.1;
  * criterion 4: monotonicity violations over a 51-point eps sweep;
  * the admission table over the eps grid;
  * which candidates flip, with their outcome (landed / crashed / timeout).
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from analysis import evaluate, fixed_scales, load_sets, metrics
from common import EPS_GRID, parse_common_args, pct, print_header, print_table
from options import build_options

BORDERLINE_IDS = (
    "ppo_greedy_p10", "ppo_greedy_p20", "ppo_greedy_p30",
    "ppo_then_noop_50", "ppo_then_noop_100", "ppo_then_noop_150",
)
TRADEOFF_IDS = (
    "ppo_waste_main_12", "ppo_waste_main_15", "ppo_waste_main_18", "ppo_waste_main_20",
    "ppo_waste_side_15", "ppo_waste_side_20",
)
SETS = (
    ("task", "task's extended set", None),
    ("task+bl", "extended + borderline candidates", BORDERLINE_IDS),
    ("all", "extended + borderline + tradeoff candidates", TRADEOFF_IDS),
)


def yn(flag: bool) -> str:
    return "PASS" if flag else "fail"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_common_args(argv, __doc__).parse_args(argv)
    sets = load_sets(args.refresh)
    fixed = fixed_scales(sets["task"])
    options = build_options()

    print_header("E1 -- option comparison on held-out seeds 21-40")
    print(f"fixed scales (seeds 1-20, extended set, pooled): {fixed.as_text()}")
    print("admit iff mu >= -eps; flips = pairs whose admission differs between eps=0 and eps=0.5")
    print("crit1: flips >= 10% of pairs, >= 5 seeds, >= 2 candidates | crit2: no crash admitted at any eps")
    print("crit3: >= 95% of ppo_deterministic landings admitted at eps=0.1 | crit4: 0 monotonicity violations")

    for set_name, label, added_ids in SETS:
        rows = sets[set_name]
        print_header(f"Summary -- {label}")
        table = []
        detail = {}
        for opt in options:
            res = evaluate(opt, rows, fixed)
            m = metrics(res)
            detail[opt.key] = (res, m)
            table.append([
                opt.key,
                f"{m.flips}/{m.pairs}",
                pct(m.flips, m.pairs),
                m.flip_seeds,
                len(m.flip_candidates),
                f"{m.crash_admitted_any}/{m.crash_pairs}",
                f"{m.det_landed_admitted_neutral}/{m.det_landed}",
                m.mono_violations,
                yn(m.crit1()), yn(m.crit2()), yn(m.crit3()), yn(m.crit4()),
            ])
        print_table(
            ["option", "flips", "share", "seeds", "cands", "crash adm", "det@0.1", "mono viol",
             "c1", "c2", "c3", "c4"],
            table,
        )
        if added_ids is not None:
            print()
            print(f"flips counted on the added candidates only ({', '.join(added_ids)}):")
            sub = []
            for opt in options:
                m = metrics(detail[opt.key][0], only=added_ids)
                sub.append([opt.key, f"{m.flips}/{m.pairs}", pct(m.flips, m.pairs), m.flip_seeds,
                            len(m.flip_candidates), f"{m.crash_admitted_any}/{m.crash_pairs}"])
            print_table(["option", "flips", "share", "seeds", "cands", "crash adm"], sub)

        print_header(f"Admissions over the eps grid -- {label}")
        print_table(
            ["option"] + [f"eps={e:g}" for e in EPS_GRID],
            [[k] + [f"{m.admitted_by_eps[e]}/{m.pairs}" for e in EPS_GRID] for k, (_, m) in detail.items()],
        )

        print_header(f"Which candidates flip (count by candidate; outcome of the flipping pairs) -- {label}")
        flip_rows = []
        for k, (_, m) in detail.items():
            if not m.flips:
                flip_rows.append([k, "-", "-"])
                continue
            cands = ", ".join(f"{c}:{n}" for c, n in sorted(m.flip_candidates.items()))
            outs = ", ".join(f"{o}:{n}" for o, n in sorted(m.flip_outcomes.items()))
            flip_rows.append([k, cands, outs])
        print_table(["option", "flipping candidates", "flip outcomes"], flip_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
