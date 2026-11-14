"""Measure what MetaMo is told after a LunarLander episode, and where that drives caution.

Three measurements, all from one set of rollouts on the development seeds:

1. Stimulus by outcome type. The risk, novelty, conduciveness and effort the
   bridge builds for a landing, a crash, a timeout and an abstention (flying
   the noop baseline, or flying nothing).
2. Fixed points. A fresh governor is fed 30 identical outcomes of one type;
   Securing, Threshold, eps and the Safety weight are recorded per step.
3. Sequences. 8 crashes then 16 landings, and the reverse.

Everything is computed under two Safety conventions:

    with_landing_reward   Safety = landing reward + shaping   (the wrapper today)
    shaping_only          Safety = shaping

Each episode is flown once and its summed landing reward, shaping and fuel are
stored separately, so both conventions come from the same runs. Nothing in
bridge/ or env/ is changed: the sums are taken by a pass-through around the
environment, and the existing `run_episode` does the flying.

Run:
    python experiments/caution_recovery/measure.py
    python experiments/caution_recovery/measure.py --refresh   # fly again
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge._loader import find_metamo_root  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from bridge.stimulus import build_stimulus_values  # noqa: E402
from data_collector.collect_candidate_sets import (  # noqa: E402
    build_extended_candidate_policies,
)
from env.lunar_lander_metamo import make_lunar_lander_governor  # noqa: E402
from env.lunar_lander_rollout import (  # noqa: E402
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    run_episode,
)
from env.lunar_lander_wrapper import SubRepEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
PILOT_CHECKPOINT = REPO_ROOT / "models" / "pilot_ppo.pt"

# Development seeds. Anything outside this range is refused, so a mistyped
# argument cannot touch seeds held back for evaluation.
ALLOWED_SEEDS = range(1, 41)

# The candidates the demo scores by default; the appraisal scales come from
# these on the first seed, as in demo/run_lunar_lander_metamo.py.
SCALE_POLICIES = (
    "ppo_deterministic",
    "ppo_stochastic",
    "left_engine",
    "main_engine",
    "right_engine",
    "random",
)

SAFETY = 0
FIXED_POINT_STEPS = 30
SEQUENCE_SHORT = 8
SEQUENCE_LONG = 16
# A policy needs this many episodes of a type to get its own fixed point.
MIN_EPISODES_PER_POLICY = 5

LANDING = "landing"
CRASH = "crash"
TIMEOUT = "timeout"
ABSTAIN_NOOP = "abstention (noop)"
ABSTAIN_STOP = "abstention (stop)"
FLOWN_TYPES = (LANDING, CRASH, TIMEOUT)
ALL_TYPES = FLOWN_TYPES + (ABSTAIN_NOOP, ABSTAIN_STOP)

# Safety of one stored episode, per convention.
CONVENTIONS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "with_landing_reward": lambda e: e["terminal"] + e["shaping"],
    "shaping_only": lambda e: e["shaping"],
}
CONVENTION_TITLES = {
    "with_landing_reward": "A: Safety = landing reward + shaping (today)",
    "shaping_only": "B: Safety = shaping only",
}
SHORT = {"with_landing_reward": "A", "shaping_only": "B"}

STIMULUS_FIELDS = ("risk", "novelty", "conduciveness", "effort")
TRACE_FIELDS = ("securing", "threshold", "epsilon", "safety_weight")


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


class SummingEnv:
    """Passes every call through and sums the per-step reward parts of an episode."""

    def __init__(self, inner: SubRepEnv) -> None:
        self._inner = inner
        self.env = inner.env
        self.sums = {"terminal": 0.0, "shaping": 0.0, "fuel": 0.0}

    def reset(self, seed: Optional[int] = None) -> Any:
        self.sums = {"terminal": 0.0, "shaping": 0.0, "fuel": 0.0}
        return self._inner.reset(seed=seed)

    def step(self, action: int) -> Any:
        out = self._inner.step(action)
        info = out[4]
        self.sums["terminal"] += float(info["terminal_reward"])
        self.sums["shaping"] += float(info["shaping_reward"])
        self.sums["fuel"] += float(info["fuel_usage"])
        return out


def episode_result(info: Dict[str, Any]) -> str:
    """How an episode ended, by the rule the demo uses."""
    if info.get("landing_success"):
        return LANDING
    if info.get("crashed"):
        return CRASH
    if info.get("stop_reason") in ("truncated", "max_steps"):
        return TIMEOUT
    return "other"


def fly_all(seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Fly the noop baseline and every candidate once per seed."""
    inner = SubRepEnv(seed=seeds[0])
    env = SummingEnv(inner)
    episodes: List[Dict[str, Any]] = []
    try:
        candidates = build_extended_candidate_policies(
            inner, pilot_checkpoint=str(PILOT_CHECKPOINT)
        )
        policies = [NOOP_BASELINE] + [
            c for c in candidates if c.skill_id != NOOP_BASELINE.skill_id
        ]
        for seed in seeds:
            for policy in policies:
                r_hat, n_hat, info = run_episode(env, policy, seed=seed)
                sums = dict(env.sums)
                # The stored parts must rebuild what the rollout adapter reports.
                if abs(sums["terminal"] - r_hat) > 1e-6:
                    raise RuntimeError(f"landing reward mismatch: {policy.skill_id} seed {seed}")
                if abs(sums["terminal"] + sums["shaping"] - float(n_hat[0])) > 1e-2:
                    raise RuntimeError(f"Safety mismatch: {policy.skill_id} seed {seed}")
                if abs(sums["fuel"] - float(n_hat[1])) > 1e-2:
                    raise RuntimeError(f"Fuel mismatch: {policy.skill_id} seed {seed}")
                episodes.append(
                    {
                        "seed": int(seed),
                        "policy": policy.skill_id,
                        "result": episode_result(info),
                        "steps": int(info["steps"]),
                        **sums,
                    }
                )
            print(f"flown seed {seed}", file=sys.stderr)
    finally:
        inner.close()
    return episodes


def load_or_fly(path: Path, seeds: Sequence[int], refresh: bool) -> List[Dict[str, Any]]:
    """The stored rollouts for exactly these seeds, flying them if needed."""
    if path.is_file() and not refresh:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("seeds") == list(seeds):
            return stored["episodes"]
    episodes = fly_all(seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seeds": list(seeds), "episodes": episodes}, indent=1),
        encoding="utf-8",
    )
    return episodes


# ---------------------------------------------------------------------------
# Outcomes under one convention
# ---------------------------------------------------------------------------


def motives(episode: Dict[str, Any], convention: str) -> np.ndarray:
    """The [Safety, Fuel] sums of a stored episode under `convention`."""
    return np.array([CONVENTIONS[convention](episode), episode["fuel"]], dtype=np.float64)


def flown_outcomes(
    episodes: Sequence[Dict[str, Any]], convention: str
) -> List[Dict[str, Any]]:
    """Every non-baseline episode as improvements over noop from the same seed."""
    baseline = {e["seed"]: e for e in episodes if e["policy"] == NOOP_BASELINE.skill_id}
    rows = []
    for e in episodes:
        if e["policy"] == NOOP_BASELINE.skill_id:
            continue
        base = baseline[e["seed"]]
        delta_r = e["terminal"] - base["terminal"]
        delta_n = motives(e, convention) - motives(base, convention)
        outcome = episode_outcome(
            delta_r, delta_n, {"steps": e["steps"], "max_steps": None}, admitted=True
        )
        rows.append(
            {"seed": e["seed"], "policy": e["policy"], "type": e["result"], "outcome": outcome}
        )
    return rows


def abstention_outcomes(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the demo feeds back when nothing is admitted, in both modes.

    `noop` flies the baseline, so the improvement over the baseline is exactly
    zero and only the flight time differs by seed. `stop` flies nothing.
    """
    rows = []
    for e in episodes:
        if e["policy"] != NOOP_BASELINE.skill_id:
            continue
        outcome = episode_outcome(
            0.0, np.zeros(2), {"steps": e["steps"], "max_steps": None}, admitted=False
        )
        rows.append(
            {"seed": e["seed"], "policy": "noop", "type": ABSTAIN_NOOP, "outcome": outcome}
        )
    rows.append(
        {
            "seed": None,
            "policy": "-",
            "type": ABSTAIN_STOP,
            "outcome": SkillOutcome(
                delta_r=0.0, delta_n=np.zeros(2), admitted=False, effort=0.0
            ),
        }
    )
    return rows


def demo_scales(rows: Sequence[Dict[str, Any]], seed: int) -> Tuple[float, float]:
    """(payoff_scale, motive_scale) as the demo derives them on its first seed."""
    records = [
        SimpleNamespace(delta_r=r["outcome"].delta_r, delta_n=r["outcome"].delta_n)
        for r in rows
        if r["seed"] == seed and r["policy"] in SCALE_POLICIES
    ]
    return appraisal_scales(records)


def representative(rows: Sequence[Dict[str, Any]], kind: str) -> Optional[SkillOutcome]:
    """The componentwise median outcome of a type, or None if it never occurred."""
    outcomes = [r["outcome"] for r in rows if r["type"] == kind]
    if not outcomes:
        return None
    return SkillOutcome(
        delta_r=float(np.median([o.delta_r for o in outcomes])),
        delta_n=np.median(np.stack([o.delta_n for o in outcomes]), axis=0),
        admitted=outcomes[0].admitted,
        effort=float(np.median([o.effort for o in outcomes])),
    )


# ---------------------------------------------------------------------------
# Governor traces
# ---------------------------------------------------------------------------


def snapshot(governor: Any) -> Dict[str, float]:
    mods = governor.modulator_summary()
    signal = governor.signal()
    return {
        "securing": mods["securing"],
        "threshold": mods["threshold"],
        "approach": mods["approach"],
        "epsilon": float(signal.pds_epsilon),
        "safety_weight": float(signal.weights[SAFETY]),
    }


def trace(outcomes: Sequence[SkillOutcome], scales: Tuple[float, float]) -> List[Dict[str, float]]:
    """State of a fresh governor before any outcome (step 0) and after each one."""
    governor = make_lunar_lander_governor(payoff_scale=scales[0], motive_scale=scales[1])
    states = [snapshot(governor)]
    for outcome in outcomes:
        governor.step(outcome)
        states.append(snapshot(governor))
    return states


def measure(episodes: Sequence[Dict[str, Any]], convention: str, first_seed: int) -> Dict[str, Any]:
    """All three measurements under one convention."""
    rows = flown_outcomes(episodes, convention)
    scales = demo_scales(rows, first_seed)
    rows = rows + abstention_outcomes(episodes)

    initial_weights = make_lunar_lander_governor(
        payoff_scale=scales[0], motive_scale=scales[1]
    ).signal().weights
    for row in rows:
        row["stimulus"] = build_stimulus_values(
            row["outcome"],
            weights=initial_weights,
            payoff_scale=scales[0],
            motive_scale=scales[1],
        )

    reps = {kind: representative(rows, kind) for kind in ALL_TYPES}
    fixed = {
        kind: trace([rep] * FIXED_POINT_STEPS, scales)
        for kind, rep in reps.items()
        if rep is not None
    }
    # The same, per policy, so the pooled median does not hide how much the
    # fixed point depends on which policy produced the outcome.
    fixed_by_policy = {}
    for kind in FLOWN_TYPES:
        for policy in sorted({r["policy"] for r in rows if r["type"] == kind}):
            of_both = [r for r in rows if r["type"] == kind and r["policy"] == policy]
            if len(of_both) >= MIN_EPISODES_PER_POLICY:
                rep = representative(of_both, kind)
                fixed_by_policy[(kind, policy)] = (
                    len(of_both),
                    trace([rep] * FIXED_POINT_STEPS, scales),
                )
    sequences = {}
    if reps[CRASH] is not None and reps[LANDING] is not None:
        sequences = {
            "crashes_then_landings": trace(
                [reps[CRASH]] * SEQUENCE_SHORT + [reps[LANDING]] * SEQUENCE_LONG, scales
            ),
            "landings_then_crashes": trace(
                [reps[LANDING]] * SEQUENCE_LONG + [reps[CRASH]] * SEQUENCE_SHORT, scales
            ),
        }
    return {
        "rows": rows,
        "scales": scales,
        "initial_weights": initial_weights,
        "representatives": reps,
        "fixed": fixed,
        "fixed_by_policy": fixed_by_policy,
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def table(header: Sequence[str], body: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return lines + [""]


def spread(values: Sequence[float]) -> str:
    """Median, with the range when the values differ."""
    lo, hi, mid = float(np.min(values)), float(np.max(values)), float(np.median(values))
    if hi - lo < 5e-4:
        return f"{mid:.3f}"
    return f"{mid:.3f} ({lo:.3f}–{hi:.3f})"


def stimulus_rows(rows: Sequence[Dict[str, Any]], label: str) -> List[str]:
    cells = [label, str(len(rows))]
    cells += [spread([getattr(r["stimulus"], f) for r in rows]) for f in STIMULUS_FIELDS]
    cells.append(f"{np.median([r['outcome'].delta_r for r in rows]):+.0f}")
    cells.append(f"{np.median([r['outcome'].delta_n[0] for r in rows]):+.1f}")
    cells.append(f"{np.median([r['outcome'].delta_n[1] for r in rows]):+.1f}")
    return cells


STIMULUS_HEADER = (
    "n", "risk", "novelty", "conduciveness", "effort", "Δr", "ΔSafety", "ΔFuel",
)


def stimulus_section(result: Dict[str, Any]) -> List[str]:
    rows = result["rows"]
    body = []
    for kind in ALL_TYPES:
        of_kind = [r for r in rows if r["type"] == kind]
        if of_kind:
            body.append(stimulus_rows(of_kind, kind))
        else:
            body.append([kind, "0"] + ["not observed"] + [""] * (len(STIMULUS_HEADER) - 2))
    lines = table(("outcome",) + STIMULUS_HEADER, body)

    by_policy = []
    for kind in FLOWN_TYPES:
        for policy in sorted({r["policy"] for r in rows if r["type"] == kind}):
            of_both = [r for r in rows if r["type"] == kind and r["policy"] == policy]
            by_policy.append(stimulus_rows(of_both, f"{kind}: {policy}"))
    lines += ["By policy:", ""]
    lines += table(("outcome: policy",) + STIMULUS_HEADER, by_policy)
    return lines


def risk_comparison(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How well risk tells a crash from a landing, or None if either is missing."""
    land = [r["stimulus"].risk for r in result["rows"] if r["type"] == LANDING]
    crash = [r["stimulus"].risk for r in result["rows"] if r["type"] == CRASH]
    if not land or not crash:
        return None
    return {
        "land": land,
        "crash": crash,
        "overlap": min(land) <= max(crash) and min(crash) <= max(land),
        # Chance that a crash drawn at random reads as riskier than a landing
        # drawn at random: 1.0 is perfect separation, 0.5 is none.
        "separation": float(np.mean([c > l for c in crash for l in land])),
        "crashes_below_landing_median": float(np.mean(np.array(crash) <= np.median(land))),
    }


def risk_verdict(result: Dict[str, Any]) -> str:
    cmp = risk_comparison(result)
    if cmp is None:
        return "landing or crash not observed; no comparison."
    land, crash = cmp["land"], cmp["crash"]
    return (
        f"landing risk median {np.median(land):.3f} "
        f"({min(land):.3f}–{max(land):.3f}), crash risk median {np.median(crash):.3f} "
        f"({min(crash):.3f}–{max(crash):.3f}); ranges "
        f"{'overlap' if cmp['overlap'] else 'do not overlap'}. "
        f"A random crash has a higher risk than a random landing "
        f"{cmp['separation']:.0%} of the time; "
        f"{cmp['crashes_below_landing_median']:.0%} of crashes have a risk at or below "
        "the median landing's."
    )


def number(field: str, value: float) -> str:
    return f"{value:.4f}" if field == "epsilon" else f"{value:.3f}"


def side_by_side(traces: Dict[str, List[Dict[str, float]]], fed: Optional[Sequence[str]]) -> List[str]:
    """One row per step, the two conventions next to each other."""
    names = list(traces)
    header = ["step"] + (["fed"] if fed else [])
    for name in names:
        header += [f"{SHORT[name]} {f}" for f in TRACE_FIELDS]
    body = []
    for step in range(len(traces[names[0]])):
        cells = [str(step)] + ([fed[step]] if fed else [])
        for name in names:
            cells += [number(f, traces[name][step][f]) for f in TRACE_FIELDS]
        body.append(cells)
    return table(header, body)


def metamo_commit() -> str:
    root = find_metamo_root()
    if root is None or not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def summary_section(results: Dict[str, Dict[str, Any]]) -> List[str]:
    """The headline numbers, computed from the measurements below."""
    body = []
    for name, result in results.items():
        cmp = risk_comparison(result)
        fixed, seq = result["fixed"], result["sequences"].get("crashes_then_landings")
        if cmp is None or seq is None:
            continue
        crash_policies = {
            key[1]: states[1][-1]["securing"]
            for key, states in result["fixed_by_policy"].items()
            if key[0] == CRASH
        }
        calmest = min(crash_policies, key=crash_policies.get)
        after_crashes, at_end = seq[SEQUENCE_SHORT], seq[-1]
        first_open = next(
            (i - SEQUENCE_SHORT for i in range(SEQUENCE_SHORT + 1, len(seq))
             if seq[i]["epsilon"] > 0.0),
            None,
        )
        body += [
            [SHORT[name], "risk, landing / crash (median)",
             f"{np.median(cmp['land']):.3f} / {np.median(cmp['crash']):.3f}"],
            [SHORT[name], "a random crash reads riskier than a random landing",
             f"{cmp['separation']:.0%}"],
            [SHORT[name], "securing fixed point, landing / crash",
             f"{fixed[LANDING][-1]['securing']:.3f} / {fixed[CRASH][-1]['securing']:.3f}"],
            [SHORT[name], "eps fixed point, landing / crash",
             f"{fixed[LANDING][-1]['epsilon']:.4f} / {fixed[CRASH][-1]['epsilon']:.4f}"],
            [SHORT[name], f"calmest crash policy ({calmest}): securing fixed point",
             f"{crash_policies[calmest]:.3f}"],
            [SHORT[name], f"securing after {SEQUENCE_SHORT} crashes, then after "
             f"{SEQUENCE_LONG} landings",
             f"{after_crashes['securing']:.3f} → {at_end['securing']:.3f}"],
            [SHORT[name], "eps over that recovery",
             f"{after_crashes['epsilon']:.4f} → {at_end['epsilon']:.4f}"],
            [SHORT[name], "landings before eps leaves 0",
             "never" if first_open is None else str(first_open)],
        ]
    return ["## Summary", ""] + table(("convention", "quantity", "value"), body)


def report(episodes: Sequence[Dict[str, Any]], results: Dict[str, Dict[str, Any]], seeds: Sequence[int]) -> str:
    names = list(results)
    lines = ["# Caution stimulus measurement", ""]

    counts: Dict[str, int] = {}
    for e in episodes:
        counts[e["result"]] = counts.get(e["result"], 0) + 1
    noop = [e["result"] for e in episodes if e["policy"] == NOOP_BASELINE.skill_id]
    lines += [
        "Generated by `experiments/caution_recovery/measure.py`. Not hand-edited.",
        "",
        "## Setup",
        "",
        f"- Seeds {seeds[0]}–{seeds[-1]}, {len(episodes)} episodes flown once "
        f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}). "
        f"The noop baseline: {', '.join(f'{k} {noop.count(k)}' for k in sorted(set(noop)))}.",
        f"- MetaMo commit `{metamo_commit()}`; governor from `make_lunar_lander_governor`.",
        "- Two Safety conventions, both computed from the same episodes:",
    ]
    for name in names:
        payoff, motive = results[name]["scales"]
        lines.append(
            f"  - **{CONVENTION_TITLES[name]}**: payoff scale {payoff:.1f}, "
            f"motive scale {motive:.1f}"
        )
    lines += [
        f"- Scales are derived as in the demo: from the default candidates on seed {seeds[0]}.",
        "- Improvements are against the noop baseline flown from the same seed. "
        "Flown outcomes are appraised as admitted; a rejected one adds 0.25 to risk.",
        "- Conduciveness in section 1 uses a fresh governor's weights "
        f"(Safety {results[names[0]]['initial_weights'][SAFETY]:.3f}).",
        "- eps is clipped to [0, 0.5]; 0.0000 means it is at the lower bound.",
        "",
    ]
    lines += summary_section(results)
    lines += [
        "## 1. Stimulus by outcome type",
        "",
        "Cells are the median, with the range in brackets when the episodes differ.",
        "",
    ]
    for name in names:
        lines += [f"### {CONVENTION_TITLES[name]}", ""]
        lines += stimulus_section(results[name])
        lines += [f"Crash against landing: {risk_verdict(results[name])}", ""]

    lines += [
        "## 2. Fixed points",
        "",
        f"A fresh governor fed {FIXED_POINT_STEPS} identical outcomes: the componentwise "
        "median outcome of each type. The outcome fed, per convention:",
        "",
    ]
    body = []
    for kind in ALL_TYPES:
        for name in names:
            rep = results[name]["representatives"][kind]
            if rep is None:
                body.append([kind, SHORT[name], "not observed", "", "", ""])
                continue
            body.append(
                [kind, SHORT[name], f"{rep.delta_r:+.0f}", f"{rep.delta_n[0]:+.1f}",
                 f"{rep.delta_n[1]:+.1f}", f"{rep.effort:.3f}"]
            )
    lines += table(("outcome", "convention", "Δr", "ΔSafety", "ΔFuel", "effort"), body)

    lines += [
        f"State after {FIXED_POINT_STEPS} steps. `last change` is the Securing change "
        "over the final step.",
        "",
    ]
    body = []
    for kind in ALL_TYPES:
        for name in names:
            states = results[name]["fixed"].get(kind)
            if states is None:
                continue
            last = states[-1]
            body.append(
                [kind, SHORT[name]]
                + [number(f, last[f]) for f in TRACE_FIELDS]
                + [f"{last['securing'] - states[-2]['securing']:+.1e}",
                   f"{max(s['securing'] for s in states):.3f}",
                   str(sum(1 for s in states[1:] if s["epsilon"] == 0.0))]
            )
    lines += table(
        ("outcome", "convention") + TRACE_FIELDS
        + ("last change", "max securing", "steps with eps = 0"),
        body,
    )

    lines += [
        f"The same per policy (its own median outcome; policies with at least "
        f"{MIN_EPISODES_PER_POLICY} episodes of the type), state after "
        f"{FIXED_POINT_STEPS} steps:",
        "",
    ]
    body = []
    for key in results[names[0]]["fixed_by_policy"]:
        for name in names:
            count, states = results[name]["fixed_by_policy"][key]
            body.append(
                [f"{key[0]}: {key[1]}", str(count), SHORT[name]]
                + [number(f, states[-1][f]) for f in TRACE_FIELDS]
            )
    lines += table(("outcome: policy", "n", "convention") + TRACE_FIELDS, body)

    lines += ["Per step (step 0 is the fresh governor):", ""]
    for kind in ALL_TYPES:
        traces = {n: results[n]["fixed"][kind] for n in names if kind in results[n]["fixed"]}
        if not traces:
            continue
        lines += [f"<details><summary>{kind}</summary>", ""]
        lines += side_by_side(traces, None)
        lines += ["</details>", ""]

    lines += ["## 3. Sequences", ""]
    plans = {
        "crashes_then_landings": (
            f"{SEQUENCE_SHORT} crashes then {SEQUENCE_LONG} landings",
            [CRASH] * SEQUENCE_SHORT + [LANDING] * SEQUENCE_LONG,
        ),
        "landings_then_crashes": (
            f"{SEQUENCE_LONG} landings then {SEQUENCE_SHORT} crashes",
            [LANDING] * SEQUENCE_LONG + [CRASH] * SEQUENCE_SHORT,
        ),
    }
    for key, (title, fed) in plans.items():
        traces = {n: results[n]["sequences"][key] for n in names if key in results[n]["sequences"]}
        if not traces:
            lines += [f"### {title}", "", "Not run: landing or crash not observed.", ""]
            continue
        lines += [f"### {title}", ""]
        switch = fed.index(fed[-1])
        body = []
        for name in names:
            states = traces[name]
            body.append(
                [SHORT[name]]
                + [f"{states[i]['securing']:.3f}" for i in (0, switch, len(fed))]
                + [f"{states[i]['epsilon']:.4f}" for i in (0, switch, len(fed))]
            )
        lines += table(
            ("convention", "securing start", f"securing after step {switch}",
             f"securing after step {len(fed)}", "eps start",
             f"eps after step {switch}", f"eps after step {len(fed)}"),
            body,
        )
        lines += side_by_side(traces, ["-"] + fed)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_seeds(text: str) -> List[int]:
    """`1-40` or `1,2,5`. Only development seeds are accepted."""
    if "-" in text:
        first, last = (int(part) for part in text.split("-", 1))
        seeds = list(range(first, last + 1))
    else:
        seeds = [int(part) for part in text.split(",")]
    outside = [s for s in seeds if s not in ALLOWED_SEEDS]
    if not seeds or outside:
        raise argparse.ArgumentTypeError(
            f"seeds must lie in {ALLOWED_SEEDS.start}-{ALLOWED_SEEDS.stop - 1}, got {text!r}"
        )
    return seeds


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=parse_seeds, default=list(ALLOWED_SEEDS))
    parser.add_argument("--rollouts", type=Path, default=HERE / "rollouts.json",
                        help="where the flown episodes are stored and reused from")
    parser.add_argument("--refresh", action="store_true", help="fly again even if stored")
    parser.add_argument("--out", type=Path, default=HERE / "results.md")
    args = parser.parse_args(argv)

    if not PILOT_CHECKPOINT.is_file():
        print(f"error: needs the PPO checkpoint at {PILOT_CHECKPOINT}", file=sys.stderr)
        return 2

    episodes = load_or_fly(args.rollouts, args.seeds, args.refresh)
    results = {name: measure(episodes, name, args.seeds[0]) for name in CONVENTIONS}
    args.out.write_text(report(episodes, results, args.seeds), encoding="utf-8")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
