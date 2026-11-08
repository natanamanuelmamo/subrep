"""E3 -- selection impact of each option.

    python experiments/epsilon_scale/e3_selection_impact.py [--refresh]

Selection scores admitted records with score = dr + w . dn
(library/skill_selector.py:21-25), ties broken by the smaller skill_id, as
MetaMoController.select does (bridge/controller.py:130-157). For every option,
evaluation seed (21-40), Safety weight in {0.2, 0.5, 0.8} and eps in
{0.1, 0.5}, the selected candidate is compared with the raw-units selection
(option F at the same eps). A change is classified as:

  * admission-driven -- the option admitted a different set and selection
    over that set with the RAW scores gives the new choice: epsilon doing its
    job (wanted);
  * score-driven     -- the option rewrote the records (A, B) and the
    rewritten scores pick a different skill from the same admitted set: a
    side effect of normalising the deltas.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import numpy as np

from analysis import evaluate, fixed_scales, load_sets
from common import parse_common_args, pct, print_header, print_table
from options import build_options

SAFETY_WEIGHTS = (0.2, 0.5, 0.8)
EPS_CHECK = (0.1, 0.5)


def select(ids, dr, dn, admitted, w):
    best = None
    for i in np.flatnonzero(admitted):
        score = float(dr[i] + np.dot(w, dn[i]))
        if best is None or score > best[0] or (score == best[0] and ids[i] < best[1]):
            best = (score, ids[i])
    return None if best is None else best[1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_common_args(argv, __doc__).parse_args(argv)
    sets = load_sets(args.refresh)
    fixed = fixed_scales(sets["task"])
    options = build_options()
    raw_opt = options[0]
    assert raw_opt.key == "F"

    print_header("E3 -- selection impact vs raw units (seeds 21-40)")
    print(f"weights w = (w_Safety, 1 - w_Safety), w_Safety in {SAFETY_WEIGHTS}; eps in {EPS_CHECK}")
    for set_name in ("task", "all"):
        rows = sets[set_name]
        raw_res = evaluate(raw_opt, rows, fixed)
        print_header(f"Candidate set: {set_name}")
        table = []
        for opt in options:
            res = evaluate(opt, rows, fixed)
            for eps in EPS_CHECK:
                changed = adm_driven = score_driven = total = 0
                for r_raw, r_opt in zip(raw_res, res):
                    ids = [r.skill_id for r in r_opt.rows]
                    adm_raw = r_raw.mu >= -eps
                    adm_opt = r_opt.mu >= -eps
                    tdr, tdn = opt.transform(r_opt.step, r_opt.scales)
                    for ws in SAFETY_WEIGHTS:
                        w = np.array([ws, 1.0 - ws])
                        total += 1
                        s_raw = select(ids, r_raw.step.delta_r, r_raw.step.delta_n, adm_raw, w)
                        s_opt = select(ids, tdr, tdn, adm_opt, w)
                        if s_opt == s_raw:
                            continue
                        changed += 1
                        s_mix = select(ids, r_opt.step.delta_r, r_opt.step.delta_n, adm_opt, w)
                        if s_mix == s_opt:
                            adm_driven += 1
                        else:
                            score_driven += 1
                table.append([opt.key, eps, f"{changed}/{total}", pct(changed, total), adm_driven, score_driven,
                              "yes" if opt.changes_records else "no"])
        print_table(["option", "eps", "selection changed", "share", "admission-driven", "score-driven",
                     "rewrites records"], table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
