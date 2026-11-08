"""Shared helpers for the epsilon-scale experiments.

Everything here is local to `experiments/epsilon_scale/`; nothing in the
shared code base imports it.

Rollouts are the expensive part (every candidate flies a full, uncapped
LunarLander episode per seed), so `collect_rows` caches them as JSON under
`experiments/epsilon_scale/outputs/`. The cache is a pure function of the
candidate set and the seeds -- every episode is seeded -- so deleting it and
re-running reproduces the same numbers. Pass `--refresh` to any script to
recompute.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

EPS_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
EPS_FINE = tuple(float(x) for x in np.linspace(0.0, 0.5, 51))
EPS_NEUTRAL = 0.1
EPS_MAX = 0.5

CAL_SEEDS = tuple(range(1, 21))
EVAL_SEEDS = tuple(range(21, 41))
ALL_SEEDS = CAL_SEEDS + EVAL_SEEDS

GLOBAL_SEED = 0

OBJECTIVES_LUNAR = ("Safety", "Fuel")


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Rows: one candidate scored against the baseline on one seed
# ---------------------------------------------------------------------------


@dataclass
class Row:
    set_name: str
    seed: int
    skill_id: str
    delta_r: float
    delta_n: List[float]
    landed: bool
    crashed: bool
    truncated: bool
    steps: int
    baseline_payoff: float
    baseline_motives: List[float]
    baseline_landed: bool
    baseline_crashed: bool
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def dn(self) -> np.ndarray:
        return np.asarray(self.delta_n, dtype=np.float64)

    @property
    def raw_margin(self) -> float:
        return float(self.delta_r) + float(np.min(self.dn))

    @property
    def outcome(self) -> str:
        if self.landed:
            return "landed"
        if self.crashed:
            return "crashed"
        return "timeout"


def _cache_path(set_name: str) -> Path:
    return OUTPUT_DIR / f"rollouts_{set_name}.json"


def _load_cache(set_name: str) -> Dict[int, List[Row]]:
    path = _cache_path(set_name)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_seed: Dict[int, List[Row]] = {}
    for item in payload["rows"]:
        row = Row(**item)
        by_seed.setdefault(row.seed, []).append(row)
    return by_seed


def _save_cache(set_name: str, by_seed: Dict[int, List[Row]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for seed in sorted(by_seed) for r in by_seed[seed]]
    _cache_path(set_name).write_text(
        json.dumps({"set_name": set_name, "rows": rows}, indent=1),
        encoding="utf-8",
    )


def collect_rows(
    set_name: str,
    seeds: Iterable[int],
    *,
    refresh: bool = False,
    verbose: bool = True,
) -> List[Row]:
    """Score candidate set `set_name` against the noop baseline on `seeds`.

    Wraps `env.lunar_lander_rollout.evaluate_candidates` unchanged (same
    seed, gamma = 1, uncapped). Cached per set; missing seeds are added.
    """
    from env.lunar_lander_rollout import evaluate_candidates
    from env.lunar_lander_wrapper import SubRepEnv

    from candidates import build_candidate_set  # local module

    seeds = [int(s) for s in seeds]
    by_seed = {} if refresh else _load_cache(set_name)
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        seed_everything()
        env = SubRepEnv(seed=missing[0])
        try:
            policies = build_candidate_set(set_name, env)
            for seed in missing:
                if verbose:
                    print(f"  [rollout] set={set_name} seed={seed}", file=sys.stderr, flush=True)
                baseline_stats, records = evaluate_candidates(env, policies, seed=seed)
                binfo = baseline_stats["info"]
                by_seed[seed] = [
                    Row(
                        set_name=set_name,
                        seed=seed,
                        skill_id=r.skill_id,
                        delta_r=float(r.delta_r),
                        delta_n=[float(v) for v in r.delta_n],
                        landed=bool(r.metadata["landing_success"]),
                        crashed=bool(r.metadata["crashed"]),
                        truncated=bool(r.metadata["truncated"]),
                        steps=int(r.metadata["steps"]),
                        baseline_payoff=float(baseline_stats["baseline_payoff"]),
                        baseline_motives=[float(v) for v in baseline_stats["baseline_motives"]],
                        baseline_landed=bool(binfo["landing_success"]),
                        baseline_crashed=bool(binfo["crashed"]),
                    )
                    for r in records
                ]
        finally:
            env.close()
        _save_cache(set_name, by_seed)
    return [r for s in seeds for r in by_seed[s]]


def group_by_seed(rows: Sequence[Row]) -> Dict[int, List[Row]]:
    out: Dict[int, List[Row]] = {}
    for r in rows:
        out.setdefault(r.seed, []).append(r)
    return out


def skill_ids(rows: Sequence[Row]) -> List[str]:
    seen: List[str] = []
    for r in rows:
        if r.skill_id not in seen:
            seen.append(r.skill_id)
    return seen


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.rjust(widths[i]) if _numeric(c) else c.ljust(widths[i]) for i, c in enumerate(row)))


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:.3g}"
        return f"{value:.3f}"
    return str(value)


def _numeric(text: str) -> bool:
    try:
        float(text.rstrip("%"))
        return True
    except ValueError:
        return False


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def parse_common_args(argv: Optional[Sequence[str]], description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--refresh", action="store_true", help="recompute rollouts instead of using the cache")
    return parser
