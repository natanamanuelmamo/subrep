"""The epsilon-scale options, as pure functions over candidate deltas.

Every option is reduced to ONE normalised margin `mu` per candidate, with the
same rule for all of them:

    admit  <=>  mu >= -epsilon

so the admission tables, flip counts and monotonicity checks are computed the
same way for every option. All functions are generic over the number of
objectives m: `delta_n` is any 1-D array, and per-objective scales are
vectors of length m.

    F  raw            mu = dr + min_i dn_i
    A  shared scales  mu = dr / s_r + min_i dn_i / s_n
    B  per-objective  mu = dr / s_r + min_i (dn_i / s_i)
    C1 scale eps      mu = (dr + min_i dn_i) / s_n          (eps_env = eps * s_n)
    C2 scale eps      mu = (dr + min_i dn_i) / (s_r + s_n)  (eps_env = eps * (s_r + s_n))
    D  relative eps   mu = (dr + min_i dn_i) / max(|R_base| + mean_i |N_base,i|, floor)
    E  rank           mu = rank / (K - 1) - 1   within the step's candidate set
    G  tanh           mu = tanh((dr + min_i dn_i) / s_n)    (eps_env = s_n * atanh(eps))
    H  declared unit  mu = (dr + min_i dn_i) / u_env        (u_env declared by the env)
    I  guarded        any of C1/C2/G/D/H, but only for candidates whose own task
                      payoff (baseline payoff + dr) is not a failure (>= 0);
                      failing candidates keep the raw gate.

Options C, D, G and H divide the raw margin by a positive number, so they never
change the records -- equivalently, they convert epsilon into environment
units. A and B rewrite the deltas themselves, which also changes selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Same floor as env/lunar_lander_rollout.APPRAISAL_SCALE_FLOOR.
LUNAR_SCALE_FLOOR = 10.0
# One landing result (+100 / -100): what LunarLander treats as one outcome.
LUNAR_DECLARED_UNIT = 100.0


@dataclass(frozen=True)
class Scales:
    s_r: float
    s_n: float
    s_vec: Tuple[float, ...]

    def as_text(self) -> str:
        vec = ", ".join(f"{v:.1f}" for v in self.s_vec)
        return f"s_r={self.s_r:.2f} s_n={self.s_n:.2f} s_i=[{vec}]"


def compute_scales(
    delta_r: Sequence[float],
    delta_n: Sequence[Sequence[float]],
    *,
    floor: float = LUNAR_SCALE_FLOOR,
) -> Scales:
    """Typical |dr|, |dn| and per-objective |dn_i|, each at least `floor`.

    s_r and s_n match env.lunar_lander_rollout.appraisal_scales exactly.
    """
    dr = np.abs(np.asarray(delta_r, dtype=np.float64))
    dn = np.abs(np.asarray(delta_n, dtype=np.float64))
    s_r = float(np.mean(dr)) or 1.0
    s_n = float(np.mean(np.mean(dn, axis=1))) or 1.0
    s_vec = [float(v) or 1.0 for v in np.mean(dn, axis=0)]
    return Scales(
        s_r=max(s_r, floor),
        s_n=max(s_n, floor),
        s_vec=tuple(max(v, floor) for v in s_vec),
    )


@dataclass
class StepInput:
    """One step's candidate set (same seed / same baseline)."""

    delta_r: np.ndarray            # shape (K,)
    delta_n: np.ndarray            # shape (K, m)
    baseline_payoff: float
    baseline_motives: np.ndarray   # shape (m,)


MuFn = Callable[[StepInput, Optional[Scales]], np.ndarray]
TransformFn = Callable[[StepInput, Optional[Scales]], Tuple[np.ndarray, np.ndarray]]


def _raw(step: StepInput) -> np.ndarray:
    return step.delta_r + np.min(step.delta_n, axis=1)


def mu_F(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    return _raw(step)


def mu_A(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    return step.delta_r / scales.s_r + np.min(step.delta_n, axis=1) / scales.s_n


def mu_B(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    s_vec = np.asarray(scales.s_vec, dtype=np.float64)
    return step.delta_r / scales.s_r + np.min(step.delta_n / s_vec, axis=1)


def mu_C1(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    return _raw(step) / scales.s_n


def mu_C2(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    return _raw(step) / (scales.s_r + scales.s_n)


def make_mu_D(floor: float = LUNAR_SCALE_FLOOR) -> MuFn:
    def mu_D(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
        size = abs(step.baseline_payoff) + float(np.mean(np.abs(step.baseline_motives)))
        return _raw(step) / max(size, floor)

    return mu_D


def mu_E(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    raw = _raw(step)
    k = len(raw)
    if k == 1:
        return np.zeros(1)
    # Average rank for ties, 0 = worst, k-1 = best.
    order = np.argsort(raw, kind="mergesort")
    ranks = np.empty(k, dtype=np.float64)
    ranks[order] = np.arange(k, dtype=np.float64)
    for value in np.unique(raw):
        idx = raw == value
        ranks[idx] = ranks[idx].mean()
    return ranks / (k - 1) - 1.0


def mu_G(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
    return np.tanh(_raw(step) / scales.s_n)


def make_mu_H(unit: float = LUNAR_DECLARED_UNIT) -> MuFn:
    def mu_H(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
        return _raw(step) / float(unit)

    return mu_H


# Option I -- guarded epsilon scaling (own addition, after the first E1 run).
# The scaled tolerance applies only to candidates whose OWN task payoff is not
# a failure: own payoff = baseline payoff + delta_r >= FAILURE_THRESHOLD.
# Failing candidates keep the raw gate (mu = raw margin, eps in raw units),
# i.e. exactly today's behaviour for them. On LunarLander a crash has own
# payoff -100, a timeout 0 and a landing +100; on the Minecraft stub the task
# reward is never negative, so the guard never fires there.
FAILURE_THRESHOLD = 0.0


def guarded(base: MuFn, threshold: float = FAILURE_THRESHOLD) -> MuFn:
    def mu_guarded(step: StepInput, scales: Optional[Scales]) -> np.ndarray:
        scaled = np.asarray(base(step, scales), dtype=np.float64)
        own_payoff = step.baseline_payoff + step.delta_r
        return np.where(own_payoff >= threshold, scaled, _raw(step))

    return mu_guarded


def identity_transform(step: StepInput, scales: Optional[Scales]):
    return step.delta_r, step.delta_n


def transform_A(step: StepInput, scales: Optional[Scales]):
    return step.delta_r / scales.s_r, step.delta_n / scales.s_n


def transform_B(step: StepInput, scales: Optional[Scales]):
    return step.delta_r / scales.s_r, step.delta_n / np.asarray(scales.s_vec)


@dataclass(frozen=True)
class Option:
    key: str            # e.g. "A-fixed"
    family: str         # A..H
    scale_source: str   # "fixed" | "per-step" | "none"
    mu: MuFn
    transform: TransformFn
    changes_records: bool
    description: str


def build_options(
    *,
    floor: float = LUNAR_SCALE_FLOOR,
    declared_unit: float = LUNAR_DECLARED_UNIT,
) -> List[Option]:
    """Every option x scale source compared in E1-E3."""
    scaled = [
        ("A", mu_A, transform_A, True, "dr/s_r + min dn/s_n"),
        ("B", mu_B, transform_B, True, "dr/s_r + min dn_i/s_i"),
        ("C1", mu_C1, identity_transform, False, "eps_env = eps*s_n"),
        ("C2", mu_C2, identity_transform, False, "eps_env = eps*(s_r+s_n)"),
        ("G", mu_G, identity_transform, False, "tanh(M/s_n) >= -eps"),
    ]
    options: List[Option] = [
        Option("F", "F", "none", mu_F, identity_transform, False, "raw: M >= -eps (control)"),
    ]
    for family, mu, tr, changes, desc in scaled:
        for source in ("fixed", "per-step"):
            options.append(Option(f"{family}-{source}", family[0], source, mu, tr, changes, desc))
    options.append(
        Option("D", "D", "none", make_mu_D(floor), identity_transform, False,
               "eps_env = eps*(|R_base|+mean|N_base|)")
    )
    options.append(Option("E", "E", "none", mu_E, identity_transform, False, "rank within set"))
    options.append(
        Option("H", "H", "none", make_mu_H(declared_unit), identity_transform, False,
               f"eps_env = eps*{declared_unit:g} (declared)")
    )
    # Option I: the epsilon-scaling families (records unchanged) behind the
    # failure guard.
    for family, mu in (("C1", mu_C1), ("C2", mu_C2), ("G", mu_G)):
        for source in ("fixed", "per-step"):
            options.append(Option(f"I:{family}-{source}", "I", source, guarded(mu), identity_transform,
                                  False, f"guarded {family}"))
    options.append(Option("I:D", "I", "none", guarded(make_mu_D(floor)), identity_transform, False,
                          "guarded D"))
    options.append(Option("I:H", "I", "none", guarded(make_mu_H(declared_unit)), identity_transform,
                          False, "guarded H"))
    return options


def step_inputs(rows_by_seed: Dict[int, list]) -> Dict[int, StepInput]:
    """Turn experiment `Row`s grouped by seed into `StepInput`s."""
    out: Dict[int, StepInput] = {}
    for seed, rows in rows_by_seed.items():
        out[seed] = StepInput(
            delta_r=np.asarray([r.delta_r for r in rows], dtype=np.float64),
            delta_n=np.asarray([r.delta_n for r in rows], dtype=np.float64),
            baseline_payoff=float(rows[0].baseline_payoff),
            baseline_motives=np.asarray(rows[0].baseline_motives, dtype=np.float64),
        )
    return out


def scales_for(option: Option, step: StepInput, fixed: Scales, floor: float) -> Optional[Scales]:
    if option.scale_source == "fixed":
        return fixed
    if option.scale_source == "per-step":
        return compute_scales(step.delta_r, step.delta_n, floor=floor)
    return None


def admitted(mu: np.ndarray, eps: float) -> np.ndarray:
    return mu >= -float(eps)
