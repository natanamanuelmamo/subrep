# Making ε meaningful on MO-LunarLander

## Summary

**The problem.** MetaMo's PDS budget ε lies in [0, 0.5], while LunarLander's PDS margins are in environment units of order ±100. As a result, ε changes **0 of 320** admissions on seeds 1–40 (E0).

**Why rescaling alone does not work.** Any option that rescales the margin or ε until ε matters also admits crashes. The noop baseline crashes on every seed, so a crashing candidate has Δr = 0. Its margin is then only a small objective loss: the best crash margin is −1.0, and ten landings score lower than that.

**Recommendation: guarded ε scaling (option I:C2-fixed).**
- The rule is ε_env = ε·(s_r + s_n), with s_r + s_n = 300 measured once on calibration seeds.
- It applies only to candidates whose own task payoff (baseline payoff + Δr) is ≥ 0.
- Candidates that fail the task keep the raw ε, so they are judged exactly as today.

**Key numbers.** On held-out seeds 21–40, compared with raw units:
- On candidates that land but burn extra fuel, ε now flips **27/120 pairs (22.5%)**, up from 0. The flips span 15 seeds and 6 candidates.
- Over every candidate set pooled, it flips 34/400 (8.5%), up from 0/400.
- On the task's own candidate set it flips **1/160** pairs: that set has nothing between a clean landing and a crash.
- Crashes stay rejected at every ε: **0/219**, against 122–126/219 for the unguarded scale options.
- Clear landings stay admitted at ε = 0.1 (18/18).
- Admissions are monotonic in ε, selection never changes (0/60), and the scale is stable (per-seed CV 0.24).

**Prototype.** The prototype is an opt-in hook in the bridge. The full suite passes: 1171 passed, 2 xfailed, against a baseline of 1144 passed, 2 xfailed.

## 1. The problem

**The gate.** The PDS gate admits a candidate when

```
Δr + min_i Δn_i ≥ −ε
```

(`certification/pds_test.py:68-85`). When a support region W_x exists, the worst case is taken over W_x instead of over min_i (`pds_test.py:70-72`).

**Where ε comes from.** MetaMo sets ε every step:

```
ε = clip(0.1 − 0.4·dev(securing) + 0.2·dev(approach), 0, 0.5)
```

(`bridge/budget.py:144-160`, with `epsilon_max = 0.5` at `:86`). The controller writes ε onto every candidate record (`_with_epsilon`, `bridge/controller.py:26-41`). The pipeline then reads it back from the record (`_candidate_epsilon`, `utils/mdn_runtime_pipeline.py:385-392`).

**E0: the problem in raw units.** E0 flew the default and extended candidate sets on seeds 1–40 (`e0_measure_problem.py`):
- The noop baseline crashed on **40/40** seeds.
- Margins were spread over a huge range (min / median / max):
  - default set: −3470 / −282 / +127;
  - extended set: −3470 / −164 / +127.
- **0/240** margins (default set) and **0/320** (extended set) fell inside the ε band [−0.5, 0]. The smallest |margin| was 26.
- Admissions were the same at every ε in the grid {0, 0.05, 0.1, 0.2, 0.3, 0.5}: 73/240 for the default set and 74/320 for the extended set, so **0 flips**.
- PPO landed on 92.5% of seeds. The fixed-engine candidates, `random` and `ppo_then_side_tradeoff` crashed on 100%.

## 2. Options

Every option is written as a normalised margin μ, with the same rule for all: admit iff μ ≥ −ε (`experiments/epsilon_scale/options.py`). Let M = Δr + min_i Δn_i be the raw margin.

| Option | Gate | What changes in the gate's meaning | Where it would live | Generic? |
|---|---|---|---|---|
| F (control) | M ≥ −ε | nothing | — | yes |
| A shared scales | Δr/s_r + min Δn/s_n | The Δr:Δn exchange rate becomes s_n/s_r ≈ 5 instead of 1 | rewrite the records (bridge or adapter) | yes |
| B per-objective | Δr/s_r + min_i Δn_i/s_i | The worst-case objective can change (Fuel, with s = 96, binds before Safety, with s = 404) | rewrite the records | yes |
| C1 / C2 scale ε | M ≥ −ε·s_n / −ε·(s_r+s_n) | ε becomes a fraction of a typical change; the records are unchanged. C1 equals A only if s_r = s_n (here 50 vs 250) | controller ε hook | yes |
| D relative ε | M ≥ −ε·max(\|R_base\| + mean\|N_base\|, 10) | The tolerance follows the luck of this seed's baseline | controller ε hook | yes |
| E rank | rank/(K−1) − 1 ≥ −ε | Admits the top share of the set. The absolute guarantee is lost | wrapper over the whole set | yes |
| G (ours) tanh | tanh(M/s_n) ≥ −ε | Same as C1 up to atanh(ε)/ε ≤ 1.10 | controller ε hook | yes |
| H (ours) declared unit | M ≥ −ε·u, u = 100 | ε is a share of one landing result; the unit takes judgment per environment | adapter constant + ε hook | code yes, value per env |
| **I (ours) guarded** | C1/C2/G/D/H if R_base + Δr ≥ 0, otherwise M ≥ −ε | Objectives can be traded only by skills that do not fail the task | controller ε hook (per candidate) | yes |

Option I was added after the first E1 run (`outputs/e1_option_comparison_v1.txt`), for the reason given in §5.

## 3. Success criteria (written before the comparison)

The criteria were committed before E1 ran (`experiments/epsilon_scale/SUCCESS_CRITERIA.md`) and have not been amended. They are judged on held-out seeds 21–40. "Our threshold" is how we made each original wording measurable.

1. **ε matters.**
   - *Original wording:* "a meaningful share of candidate-seed pairs change between admitted and rejected as ε moves from 0 to 0.5; flips from one seed don't count".
   - *Our threshold:* at least 10% of pairs flip, spanning at least 5 seeds and at least 2 candidates.
   - If the task's candidate set is bimodal, the criterion is judged on borderline candidates, and the report says which result comes from the option and which from the candidate set.
2. **Safety.**
   - *Original wording:* "crash-only candidates are rejected at every ε in [0, 0.5]".
   - *Our threshold:* 100% of crashing pairs are rejected at every ε in the grid.
3. **Good skills pass.**
   - *Original wording:* "clear landings (PPO deterministic) are admitted at ε = 0.1".
   - *Our threshold:* at least 95% of `ppo_deterministic` landings are admitted.
4. **Monotonic.**
   - *Original wording:* "lowering ε never admits more".
   - *Our threshold:* 0 violations over the grid plus a 51-point sweep.
5. **Stable.**
   - *Original wording:* "admissions don't swing wildly between similar seeds".
   - *Our threshold:* the per-seed coefficient of variation (CV) of each fixed scale is ≤ 0.25, and at least 90% of pairs agree with the majority verdict of their (candidate, outcome class) group at ε = 0.1.
6. **Generic.**
   - *Original wording:* "same code at m = 2 and m = 6; Minecraft tests don't break".
   - *Our threshold:* plus at least 1 ε flip on the Minecraft stub with the fix on.

## 4. Method

**Candidate sets** (`experiments/epsilon_scale/candidates.py`):
- **task**: the extended set, 8 candidates (`data_collector/collect_candidate_sets.py:120-135`). The default set is a subset of it.
- **borderline** (defined only in the experiment scripts):
  - PPO with a random action at probability p ∈ {0.1, 0.2, 0.3};
  - PPO that cuts its engines after k ∈ {50, 100, 150} steps.
- **tradeoff** (added after the first E1 run): PPO that fires the main engine (p ∈ {0.12, 0.15, 0.18, 0.20}) or a left/right engine pair (p ∈ {0.15, 0.20}) instead of coasting. These skills land, but burn extra fuel.

**Rollouts.**
- Every rollout uses `evaluate_candidates` unchanged (`env/lunar_lander_rollout.py:215`): noop baseline, γ = 1, uncapped, same seed for every candidate.
- All randomness is seeded. Rollouts are cached as JSON, which `--refresh` regenerates.

**Seeds.** Seeds 1–20 are used for calibration and seeds 21–40 for evaluation.

**ε grid.** {0, 0.05, 0.1, 0.2, 0.3, 0.5}, plus a 51-point sweep for the monotonicity check.

**Scales.**
- **Fixed scales** are computed exactly as `appraisal_scales` does (`env/lunar_lander_rollout.py:109`), pooled over the task set on seeds 1–20, with a floor of 10. The result is s_r = 50.0, s_n = 250.1 and s_i = [404.5, 95.8].
- **Per-step scales** are the same statistic computed on the candidate set certified at that step.

**Scripts.** E0–E5 are in `experiments/epsilon_scale/` (§9).

## 5. Results

### Option comparison (E1, seeds 21–40)

The first E1 run compared options A–H on the task set (160 pairs) and on task + borderline (280 pairs). **Every option that let ε flip anything did so by admitting crashes.** On the task set, for example:
- A-fixed flipped 39/160 and admitted 38/121 crashes;
- C2-fixed flipped 43/160 and admitted 42/121 crashes;
- F flipped 0.

**Diagnosis**, over all 40 seeds and all candidates:
- The noop baseline always crashes, so a crash scored against it has Δr = 0, and the terminal penalty cancels inside Safety as well.
- Noop burns no fuel, so **every** candidate has ΔFuel ≤ 0. Fuel is the binding objective in 196/402 crash pairs.
- The best crash margin is therefore **−1.0**, while 10 landings score lower, down to −233.
- In raw units, crashes are rejected only because of their units. No threshold on a rescaled margin can make ε matter and still reject every crash. Option I uses the candidate's own task payoff instead.

The final comparison pools all candidates (task + borderline + tradeoff = 400 pairs, 219 of them crashes); the same output also reports each set on its own (`outputs/e1_option_comparison.txt`).

| Option | Flip share, all 400 pairs (flips among the 120 tradeoff pairs) | Crashes rejected at all ε | Landings admitted at ε=0.1 | Monotonic | Stable (scale CV ≤ 0.25) | Selection changes (E3) | Generic (E5) |
|---|---|---|---|---|---|---|---|
| F | 0.0% (0/120) | ✓ 0/219 admitted | 18/18 | ✓ | — | 0/60 | ✓ |
| A-fixed | 31.0% (1/120) | ✗ 122/219 | 18/18 | ✓ | ✗ s_n 0.289 | 0/60 (rewrites records) | ✓ |
| A-per-step | 33.5% (22/120) | ✗ 111/219 | 18/18 | ✓ | ✗ | 0/60 | ✓ |
| B-fixed | 9.2% (2/120) | ✗ 33/219 | 18/18 | ✓ | ✗ s_Safety 0.342 | 10/60, all score-driven | ✓ |
| C1-fixed | 36.5% (19/120) | ✗ 122/219 | 18/18 | ✓ | ✗ 0.289 | 0/60 | ✓ |
| C2-fixed | 40.0% (27/120) | ✗ 126/219 | 18/18 | ✓ | ✓ 0.240 | 0/60 | ✓ |
| D | 16.0% (13/120) | ✗ 50/219 | 18/18 | ✓ | follows the baseline | 0/60 | ✓ |
| E | 45.2% (47/120) | ✗ 87/219 | ✗ 14/18 | ✓ | — | 8/60 | ✓ |
| G-fixed | 38.2% (21/120) | ✗ 125/219 | 18/18 | ✓ | ✗ 0.289 | 0/60 | ✓ |
| H (u = 100) | 8.8% (11/120) | ✗ 23/219 | 18/18 | ✓ | declared | 0/60 | value per env |
| I:C1-fixed | 6.0% (19/120) | ✓ 0/219 | 18/18 | ✓ | ✗ 0.289 | 0/60 | ✓ |
| **I:C2-fixed** | **8.5% (27/120)** | **✓ 0/219** | **18/18** | **✓** | **✓ 0.240** | **0/60** | **✓** |
| I:G-fixed | 7.0% (21/120) | ✓ 0/219 | 18/18 | ✓ | ✗ 0.289 | 0/60 | ✓ |
| I:D / I:H | 3.5% / 3.0% | ✓ 0/219 | 18/18 | ✓ | — | 0/60 | ✓ |

**Which candidates flip under I:C2-fixed.** All of them are realistic trade-off skills:
- 27 of the 34 flips are landings with a fuel overrun;
- the other 7 are timeouts, where the lander hovered until the limit.

**Criterion 1, per candidate set:**
- *Task set:* 1/160 flips, so criterion 1 fails. The task set is bimodal: margins around +100 for clean landings, and crashes below that, with nothing in between.
- *Tradeoff set:* 27/120 (22.5%) flips over 15 seeds and 6 candidates, so criterion 1 passes.
- *Pooled:* 8.5%. The crash-type borderline candidates dilute it, and the guard deliberately keeps them out.

The flip shares above therefore come from the **candidate set**. Admitting no crashes comes from the **option**.

**Option A changes the gate's meaning.** Dividing Δr by s_r = 50 turns a landing into +4, which dominates the margin. A therefore flips only 1/120 trade-off landings, against 27 for C2.

### E2: scale stability

Per-seed CV on the calibration seeds:

| Scale | CV | Criterion 5 (≤ 0.25) |
|---|---|---|
| s_r | 0.137 | pass |
| s_n | 0.289 | fail |
| **s_r + s_n** | **0.240** | **pass** |
| s_Safety | 0.342 | fail |
| s_Fuel | 0.244 | pass |

- **Pooled fixed scale.** In leave-one-seed-out refits, s_r + s_n stays in [289.9, 305.6], against 300.1 on the full fit. The worst case changes 7/2400 I:C2-fixed verdicts.
- **Per-step scales depend on which candidates are certified together.** On the same seeds 21–40, the mean s_n is 194 for the task set, 155 for task + borderline and 184 for all candidates.
- **Fixed vs per-step agreement.** Under I:C2, fixed and per-step scales give the same verdict in 99.5% of pairs. Under A they agree in 87.5%.
- **Outcome-class agreement** at ε = 0.1 is at least 93.8% for every option, so all pass that part of criterion 5. I:C2-fixed scores 96.0%.

### E3: selection impact

E3 checks 20 seeds × 3 weight vectors (w_Safety ∈ {0.2, 0.5, 0.8}) at ε ∈ {0.1, 0.5}:
- **Options that leave the records unchanged** (C, D, G, H, I): the selected skill never changes, 0/60. The best admitted skill is always a clean PPO landing. Newly admitted trade-off skills score lower.
- **A:** also 0/60.
- **B:** 4/60 changes on the task set and 10/60 on all candidates. Every one is **score-driven**: rewriting Δn by 404 vs 96 re-weights the objectives. This is an unwanted side effect.
- **E:** 8/60 changes at ε = 0.1, admission-driven: it ranks clean landings out.

## 6. Recommended option and prototype

**Option I:C2-fixed.** It is the only option that meets criteria 2–6 and gives ε a real effect. Under the frozen ranking rule it also wins on every count: it does not change records, it uses a fixed scale, it has the highest guarded flip share, and its code is small.

**What was implemented:**
- **`bridge/epsilon_scale.py`** (new):
  - `GuardedEpsilonScale` (`:63`), whose `epsilon_for` (`:94`) returns ε·scale. When the guard fires (`is_guarded`, `:81`) it returns the raw ε instead: the guard fires when the candidate's own payoff is below `failure_threshold` (default 0, `:59`), or when that payoff cannot be computed.
  - `calibrate_epsilon_scale` (`:110`) returns s_r + s_n for any m.
- **`bridge/controller.py`:**
  - `MetaMoController.__init__` takes `epsilon_scale=None` (`:89`).
  - `certify` stamps a per-candidate ε when a scale is set (`:130-150`); the old path is unchanged when it is None.
  - `StepRecord.epsilon_scale` (`:60`) logs the scale used.
  - Because `_candidate_epsilon` already reads ε per record (`utils/mdn_runtime_pipeline.py:385-392`), the core needed **no change**.
- **`env/lunar_lander_rollout.py:93`:** `EPSILON_SCALE = 300.0`, with its calibration documented. Nothing uses it unless a controller is given a scale.

**Why it belongs in the bridge, not the core.** The tolerance is a property of how MetaMo's budget maps onto an environment's units, and it needs the baseline payoff, which the controller already holds (`bridge/controller.py:111`). A core version would have to add a parameter to `certify_candidate_skills`. That is worth doing only if non-MetaMo callers need it too, and we leave it as a team decision.

**Tests added:**
- **`tests/test_bridge_epsilon_scale.py`**, 17 test functions (26 test cases), all through the real PDS pipeline. They cover:
  - ε flipping a borderline admission (`:183`);
  - crashes rejected at every ε (`:193`);
  - why the guard is needed (`:201`);
  - monotonicity over 26 ε values (`:207`);
  - selection seeing the unchanged deltas (`:216`);
  - the scale being off by default (`:177`, `:233`);
  - m = 6 on the real Minecraft stub (`:280`, `:289`);
  - calibration at m = 2 and m = 6 (`:152`).
- **`tests/test_lunar_lander_integration.py`:**
  - `test_epsilon_cannot_change_a_lunar_lander_admission` (`:252`) is kept, and still passes with the fix off.
  - The new `test_epsilon_can_change_a_lunar_lander_admission_with_the_scale` (`:345`) flies a real fuel-wasting PPO landing on seed 8 (margin −43). It checks that the landing is admitted exactly when ε·300 ≥ 43, that no crash is admitted with or without the scale, and that without the scale ε changes nothing.

## 7. Evidence it works (before/after)

**E4 run 1a: scripted ε, same start (seed 8), fresh pipeline per step.** Weights are fixed at (0.5, 0.5). In every step the selected skill is `ppo_deterministic`, and it lands.

| Step | ε | Fix off: admitted | Fix on: tolerance | Fix on: admitted |
|---|---|---|---|---|
| 0 | 0.5 | ppo_deterministic | 150 | ppo_deterministic, **ppo_waste_side_15** |
| 1 | 0.3 | ppo_deterministic | 90 | ppo_deterministic, **ppo_waste_side_15** |
| 2 | 0.1 | ppo_deterministic | 30 | ppo_deterministic |
| 3 | 0.0 | ppo_deterministic | 0 | ppo_deterministic |

- **Fix off:** 1 distinct admitted set.
- **Fix on:** 2 distinct admitted sets, following ε. The flipping skill has margin −43, so it needs a tolerance of at least 43.
- **Crashes:** the crashing candidates (left/main/right engine, random) are rejected in every step, with the fix off and on.

**E4 run 1b: scripted ε cycling 0.5/0.3/0.1/0.0 over held-out seeds 21–40.** The admitted set differs between fix on and fix off on 3 of 20 steps:

| Step | Seed | ε | Fix off: admitted | Fix on: admitted |
|---|---|---|---|---|
| 12 | 33 | 0.5 | ppo_deterministic | ppo_deterministic, **ppo_waste_main_15** |
| 13 | 34 | 0.3 | ppo_deterministic, ppo_stochastic | ppo_deterministic, ppo_stochastic, **ppo_waste_main_15**, **ppo_waste_side_15** |
| 18 | 39 | 0.1 | ppo_deterministic, ppo_stochastic | ppo_deterministic, ppo_stochastic, **ppo_waste_main_15** |

No crash is admitted in either run, and the selected skill is the same on every step.

**E4 run 2: the real MetaMo governor** (`env/lunar_lander_metamo.py:77`), seeds 21–30.
- ε goes **0.100 → 0.011 → 0.000 from step 2 onward**: MetaMo's caution modulators saturate and pin ε at 0. Meanwhile the Safety weight rises from 0.63 to about 0.80.
- With ε = 0, the scaled tolerance is also 0, so fix on and fix off agree on **10/10 steps**.
- The fix is in place; with this governor it never gets a non-zero ε to act on. The saturation itself is a separate issue and out of scope here.

**E4 run 1c: a side finding.** Run 1c repeats run 1a with one pipeline for all steps. On a repeated context, the pipeline grows W_x from the weights it has already certified (`observe_certified_weight`), and the gate's worst case becomes min over W_x of w·Δn instead of min_i.
- With the fix **off**, `ppo_waste_side_15` is admitted at ε = 0 from step 1 onward: 200 + 0.5·(317.8 − 243) = +237.
- So same-context reuse loosens the gate independently of ε.

**E5: the Minecraft stub, m = 6.** Calibration on episodes 1–20 gives a scale of 0.050. Evaluation covers episodes 21–40: 800 pairs, with raw margins from −0.083 to −0.015.
- **Raw ε:** 800/800 pairs flip. ε acts as a switch: admissions go 0, then 560 at ε = 0.05, then 800 from ε = 0.1 upward.
- **With the fix:**
  - 200/800 pairs flip (25%), over 20 seeds and 2 candidates;
  - admissions are 0 up to ε = 0.2, then 20 at ε = 0.3 and 200 at ε = 0.5;
  - 0 monotonicity violations, and the guard fired 0 times.
- **Tests:** the Minecraft and bridge test files pass, 153/153.

**Full suite.** `python -m pytest tests/ -q -p no:cacheprovider` gives **1171 passed, 2 xfailed**, against 1144 passed and 2 xfailed before the change. MetaMo is untouched: `git -C external/metamo status --porcelain` is empty, and the diff to `certification/`, `library/`, `utils/mdn_runtime_pipeline.py` and `external/` is empty.

## 8. Side effects and limitations

- **What else the fix changes.** Only the ε stamped on records, and the `epsilon` and `admission_margin` that admitted records carry back. Δr and Δn are never touched, so selection is unchanged (E3: 0/60), and stored certificates keep environment units.
- **It cannot act while MetaMo pins ε at 0** (E4 run 2). The saturation has to be fixed separately before the risk channel works end to end with the real governor.
- **The task's own candidate set gives ε almost nothing to decide** (1/160 flips). With a crashing noop baseline, a landing scores Δr = +200 and nearly always clears any tolerance. The trade-off skills ε is meant for — landings that overspend an objective — only show up when such candidates are offered.
- **Team decisions needed:**
  - **The failure test (own payoff ≥ 0).** Timeouts (payoff 0) count as non-failures and account for 7 of the 34 I:C2 flips. The threshold is configurable.
  - **The baseline.** A noop baseline that crashes on every seed is what makes crash rejection accidental in raw units. A non-failing reference policy would change the picture.
  - **Minecraft.** The stub's hand-set `_REWARD_SCALE = 0.08` (`env/minecraft_stub.py:69-88`) already puts its margins in a raw-ε band, so enabling the fix there stacks two calibrations and makes the gate much stricter (0 admissions below ε = 0.3). The scale should be enabled per environment. If it becomes the standard, the stub's reward scale should be revisited, and its tests (`tests/test_minecraft_stub.py:260-285`) with it.
  - **Same-context reuse loosens the gate through W_x** (run 1c). This is pre-existing behaviour, but it interacts with any ε study that repeats contexts.
  - **The calibration** is a constant measured on one candidate population. A very different skill library would need re-measuring.

## 9. How to reproduce

Run everything from the repository root, with the PPO checkpoint and the MetaMo submodule present. The rollout caches are rebuilt with `--refresh`.

| Table | Command |
|---|---|
| E0 margins, band share, admission table | `python experiments/epsilon_scale/e0_measure_problem.py` |
| E1 option comparison | `python experiments/epsilon_scale/e1_option_comparison.py` |
| E2 scale stability | `python experiments/epsilon_scale/e2_scale_stability.py` |
| E3 selection impact | `python experiments/epsilon_scale/e3_selection_impact.py` |
| E4 end-to-end, before/after | `python experiments/epsilon_scale/e4_end_to_end.py` |
| E5 Minecraft | `python experiments/epsilon_scale/e5_minecraft_generality.py` |
| Tests for the fix | `python -m pytest tests/test_bridge_epsilon_scale.py tests/test_lunar_lander_integration.py -q -p no:cacheprovider` |
| Full suite | `python -m pytest tests/ -q -p no:cacheprovider` |

## Appendix: raw outputs

The printed output of each run is copied verbatim from `experiments/epsilon_scale/outputs/`. E1 appears twice:
- **A.2** is the first run (options A–H), which led to option I and the tradeoff candidates;
- **A.3** is the final run.

### A.1 E0 -- e0_measure_problem.py

```text

==========================================================================
E0 -- the epsilon-scale problem in raw units (seeds 1-40, baseline = noop)
==========================================================================
baseline (noop) outcomes over 40 seeds: landed=0 crashed=40 timeout=0
objectives: ['Safety', 'Fuel']; margin = delta_r + min_i delta_n_i; admit iff margin >= -eps; eps band = [-0.5, 0]

======================================
Per candidate-seed pair (extended set)
======================================
seed  candidate               dr       dn_Safety  dn_Fuel   margin     outcome  steps
----  ----------------------  -------  ---------  --------  ---------  -------  -----
   1  ppo_deterministic       200.000    326.026   -91.000    109.000  landed     205
   1  ppo_stochastic          200.000    321.747  -100.000    100.000  landed     195
   1  left_engine               0.000   -696.950   -79.000   -696.950  crashed     79
   1  main_engine               0.000   -771.728  -118.000   -771.728  crashed    118
   1  right_engine              0.000   -491.039   -78.000   -491.039  crashed     78
   1  random                    0.000    -47.576   -79.000    -79.000  crashed     99
   1  ppo_then_side_tradeoff    0.000   -274.301   -56.000   -274.301  crashed     81
   1  ppo_noisy_actions         0.000     67.873   -65.000    -65.000  crashed    106
   2  ppo_deterministic       200.000    208.491   -86.000    114.000  landed     193
   2  ppo_stochastic          200.000    206.780   -88.000    112.000  landed     185
   2  left_engine               0.000   -656.916   -73.000   -656.916  crashed     73
   2  main_engine               0.000  -2.81e+03  -289.000  -2.81e+03  crashed    289
   2  right_engine              0.000   -537.221   -72.000   -537.221  crashed     72
   2  random                    0.000   -228.601   -86.000   -228.601  crashed    106
   2  ppo_then_side_tradeoff    0.000   -236.866   -46.000   -236.866  crashed     74
   2  ppo_noisy_actions         0.000    -52.856   -53.000    -53.000  crashed     90
   3  ppo_deterministic       200.000    202.975   -83.000    117.000  landed     202
   3  ppo_stochastic          200.000    200.986   -91.000    109.000  landed     194
   3  left_engine               0.000   -485.873   -70.000   -485.873  crashed     70
   3  main_engine               0.000  -1.17e+03  -160.000  -1.17e+03  crashed    160
   3  right_engine              0.000   -451.573   -69.000   -451.573  crashed     69
   3  random                    0.000    -90.322   -75.000    -90.322  crashed     99
   3  ppo_then_side_tradeoff    0.000   -234.712   -43.000   -234.712  crashed     71
   3  ppo_noisy_actions         0.000     -0.005   -57.000    -57.000  crashed     90
   4  ppo_deterministic       200.000    206.683   -88.000    112.000  landed     196
   4  ppo_stochastic          200.000    201.341   -96.000    104.000  landed     205
   4  left_engine               0.000   -729.979   -76.000   -729.979  crashed     76
   4  main_engine               0.000  -1.06e+03  -127.000  -1.06e+03  crashed    127
   4  right_engine              0.000   -625.722   -78.000   -625.722  crashed     78
   4  random                    0.000   -172.219   -89.000   -172.219  crashed    114
   4  ppo_then_side_tradeoff    0.000   -344.796   -51.000   -344.796  crashed     79
   4  ppo_noisy_actions         0.000      3.400   -58.000    -58.000  crashed     98
   5  ppo_deterministic       200.000    201.555   -99.000    101.000  landed     216
   5  ppo_stochastic          200.000    204.894   -93.000    107.000  landed     194
   5  left_engine               0.000   -900.339   -86.000   -900.339  crashed     86
   5  main_engine               0.000  -3.47e+03  -322.000  -3.47e+03  crashed    322
   5  right_engine              0.000   -857.545   -86.000   -857.545  crashed     86
   5  random                    0.000   -116.180   -85.000   -116.180  crashed    117
   5  ppo_then_side_tradeoff    0.000   -392.825   -60.000   -392.825  crashed     88
   5  ppo_noisy_actions         0.000    -85.135   -58.000    -85.135  crashed    105
   6  ppo_deterministic       200.000    400.288  -143.000     57.000  landed     258
   6  ppo_stochastic          200.000    404.331  -144.000     56.000  landed     255
   6  left_engine               0.000   -878.827   -87.000   -878.827  crashed     87
   6  main_engine               0.000   -559.782   -74.000   -559.782  crashed     74
   6  right_engine              0.000   -710.745   -88.000   -710.745  crashed     88
   6  random                    0.000   -135.099   -80.000   -135.099  crashed     99
   6  ppo_then_side_tradeoff    0.000   -278.379   -70.000   -278.379  crashed     90
   6  ppo_noisy_actions         0.000    126.026   -87.000    -87.000  crashed    122
   7  ppo_deterministic       200.000    267.798   -87.000    113.000  landed     186
   7  ppo_stochastic          200.000    260.897  -105.000     95.000  landed     198
   7  left_engine               0.000   -372.604   -62.000   -372.604  crashed     62
   7  main_engine               0.000   -631.983  -108.000   -631.983  crashed    108
   7  right_engine              0.000   -308.473   -63.000   -308.473  crashed     63
   7  random                    0.000     -9.779   -62.000    -62.000  crashed     80
   7  ppo_then_side_tradeoff    0.000    -81.876   -42.000    -81.876  crashed     65
   7  ppo_noisy_actions         0.000     50.101   -62.000    -62.000  crashed     87
   8  ppo_deterministic       200.000    271.821  -174.000     26.000  landed     211
   8  ppo_stochastic          100.000     58.225  -441.000   -341.000  timeout   1000
   8  left_engine               0.000   -331.878   -56.000   -331.878  crashed     56
   8  main_engine               0.000   -472.251  -111.000   -472.251  crashed    111
   8  right_engine              0.000   -251.312   -55.000   -251.312  crashed     55
   8  random                    0.000    -90.358   -49.000    -90.358  crashed     67
   8  ppo_then_side_tradeoff    0.000    -68.696   -43.000    -68.696  crashed     60
   8  ppo_noisy_actions         0.000     13.553   -60.000    -60.000  crashed     75
   9  ppo_deterministic       200.000    425.979  -129.000     71.000  landed     250
   9  ppo_stochastic          200.000    433.446  -134.000     66.000  landed     246
   9  left_engine               0.000   -563.247   -81.000   -563.247  crashed     81
   9  main_engine               0.000   -468.784   -72.000   -468.784  crashed     72
   9  right_engine              0.000   -508.794   -80.000   -508.794  crashed     80
   9  random                    0.000    -57.130   -86.000    -86.000  crashed    113
   9  ppo_then_side_tradeoff    0.000    -98.831   -62.000    -98.831  crashed     81
   9  ppo_noisy_actions         0.000    123.328   -89.000    -89.000  crashed    118
  10  ppo_deterministic       200.000    326.447   -87.000    113.000  landed     217
  10  ppo_stochastic          200.000    324.361   -95.000    105.000  landed     206
  10  left_engine               0.000   -575.326   -83.000   -575.326  crashed     83
  10  main_engine               0.000  -2.73e+03  -300.000  -2.73e+03  crashed    300
  10  right_engine              0.000   -557.815   -76.000   -557.815  crashed     76
  10  random                    0.000    -78.381   -70.000    -78.381  crashed     96
  10  ppo_then_side_tradeoff    0.000   -189.253   -49.000   -189.253  crashed     77
  10  ppo_noisy_actions         0.000     36.228   -52.000    -52.000  crashed     96
  11  ppo_deterministic       200.000    259.566   -95.000    105.000  landed     188
  11  ppo_stochastic          200.000    241.239   -82.000    118.000  landed     178
  11  left_engine               0.000   -331.865   -58.000   -331.865  crashed     58
  11  main_engine               0.000   -643.902  -107.000   -643.902  crashed    107
  11  right_engine              0.000   -372.918   -56.000   -372.918  crashed     56
  11  random                    0.000   -137.230   -57.000   -137.230  crashed     76
  11  ppo_then_side_tradeoff    0.000   -164.497   -33.000   -164.497  crashed     59
  11  ppo_noisy_actions         0.000      2.525   -58.000    -58.000  crashed     86
  12  ppo_deterministic       200.000    295.603   -78.000    122.000  landed     176
  12  ppo_stochastic          200.000    278.015  -101.000     99.000  landed     183
  12  left_engine               0.000   -402.919   -58.000   -402.919  crashed     58
  12  main_engine               0.000   -529.951   -79.000   -529.951  crashed     79
  12  right_engine              0.000   -349.092   -57.000   -349.092  crashed     57
  12  random                    0.000   -277.472   -68.000   -277.472  crashed     89
  12  ppo_then_side_tradeoff    0.000      8.383   -82.000    -82.000  crashed    103
  12  ppo_noisy_actions       200.000    245.396   -94.000    106.000  landed     182
  13  ppo_deterministic       200.000    336.967  -122.000     78.000  landed     204
  13  ppo_stochastic          200.000    339.876  -121.000     79.000  landed     213
  13  left_engine               0.000   -357.503   -59.000   -357.503  crashed     59
  13  main_engine               0.000   -524.679   -80.000   -524.679  crashed     80
  13  right_engine              0.000   -338.223   -62.000   -338.223  crashed     62
  13  random                    0.000    -38.261   -58.000    -58.000  crashed     74
  13  ppo_then_side_tradeoff    0.000   -360.222   -77.000   -360.222  crashed     97
  13  ppo_noisy_actions         0.000     30.178   -75.000    -75.000  crashed     91
  14  ppo_deterministic       200.000    320.504   -93.000    107.000  landed     202
  14  ppo_stochastic          200.000    322.638   -92.000    108.000  landed     201
  14  left_engine               0.000   -458.053   -75.000   -458.053  crashed     75
  14  main_engine               0.000   -831.737  -133.000   -831.737  crashed    133
  14  right_engine              0.000   -571.576   -76.000   -571.576  crashed     76
  14  random                    0.000   -188.822   -77.000   -188.822  crashed    104
  14  ppo_then_side_tradeoff    0.000   -201.192   -50.000   -201.192  crashed     76
  14  ppo_noisy_actions         0.000     43.814   -62.000    -62.000  crashed     89
  15  ppo_deterministic       200.000    326.801   -94.000    106.000  landed     228
  15  ppo_stochastic          200.000    326.386   -86.000    114.000  landed     216
  15  left_engine               0.000   -649.335   -84.000   -649.335  crashed     84
  15  main_engine               0.000  -1.71e+03  -201.000  -1.71e+03  crashed    201
  15  right_engine              0.000   -772.537   -84.000   -772.537  crashed     84
  15  random                    0.000    -13.706   -84.000    -84.000  crashed    114
  15  ppo_then_side_tradeoff    0.000   -272.989   -58.000   -272.989  crashed     86
  15  ppo_noisy_actions         0.000     66.767   -62.000    -62.000  crashed    106
  16  ppo_deterministic       200.000    308.021  -130.000     70.000  landed     325
  16  ppo_stochastic          200.000    360.233  -415.000   -215.000  landed     704
  16  left_engine               0.000   -632.736   -76.000   -632.736  crashed     76
  16  main_engine               0.000   -537.806   -90.000   -537.806  crashed     90
  16  right_engine              0.000   -335.922   -77.000   -335.922  crashed     77
  16  random                    0.000     64.874   -80.000    -80.000  crashed    111
  16  ppo_then_side_tradeoff    0.000   -159.933   -59.000   -159.933  crashed     79
  16  ppo_noisy_actions         0.000   -122.794   -79.000   -122.794  crashed    117
  17  ppo_deterministic       200.000    612.801  -124.000     76.000  landed     253
  17  ppo_stochastic          200.000    643.748  -130.000     70.000  landed     236
  17  left_engine               0.000   -642.960   -87.000   -642.960  crashed     87
  17  main_engine               0.000   -298.436   -81.000   -298.436  crashed     81
  17  right_engine              0.000   -334.616   -86.000   -334.616  crashed     86
  17  random                    0.000    -51.776   -69.000    -69.000  crashed     95
  17  ppo_then_side_tradeoff    0.000    -18.785   -69.000    -69.000  crashed     90
  17  ppo_noisy_actions         0.000    319.836   -80.000    -80.000  crashed    122
  18  ppo_deterministic       100.000    302.749  -809.000   -709.000  timeout   1000
  18  ppo_stochastic          200.000    434.725  -122.000     78.000  landed     242
  18  left_engine               0.000   -546.754   -82.000   -546.754  crashed     82
  18  main_engine               0.000   -507.095   -82.000   -507.095  crashed     82
  18  right_engine              0.000   -815.985   -85.000   -815.985  crashed     85
  18  random                    0.000   -320.634   -75.000   -320.634  crashed     96
  18  ppo_then_side_tradeoff    0.000   -241.847   -63.000   -241.847  crashed     83
  18  ppo_noisy_actions         0.000     99.547   -80.000    -80.000  crashed    117
  19  ppo_deterministic       200.000    191.204   -93.000    107.000  landed     198
  19  ppo_stochastic          200.000    191.557   -96.000    104.000  landed     202
  19  left_engine               0.000   -511.600   -71.000   -511.600  crashed     71
  19  main_engine               0.000  -1.09e+03  -158.000  -1.09e+03  crashed    158
  19  right_engine              0.000   -467.879   -71.000   -467.879  crashed     71
  19  random                    0.000   -194.128   -88.000   -194.128  crashed    109
  19  ppo_then_side_tradeoff    0.000   -290.856   -45.000   -290.856  crashed     72
  19  ppo_noisy_actions         0.000    -74.758   -49.000    -74.758  crashed     85
  20  ppo_deterministic       200.000    311.894   -91.000    109.000  landed     176
  20  ppo_stochastic          200.000    288.444   -87.000    113.000  landed     172
  20  left_engine               0.000   -424.748   -52.000   -424.748  crashed     52
  20  main_engine               0.000  -2.69e+03  -324.000  -2.69e+03  crashed    324
  20  right_engine              0.000   -172.654   -52.000   -172.654  crashed     52
  20  random                    0.000    -78.276   -45.000    -78.276  crashed     62
  20  ppo_then_side_tradeoff    0.000    -84.421   -39.000    -84.421  crashed     57
  20  ppo_noisy_actions         0.000    102.648   -48.000    -48.000  crashed     65
  21  ppo_deterministic       200.000    328.936  -128.000     72.000  landed     218
  21  ppo_stochastic          200.000    328.961  -120.000     80.000  landed     216
  21  left_engine               0.000   -244.787   -54.000   -244.787  crashed     54
  21  main_engine               0.000   -516.234   -70.000   -516.234  crashed     70
  21  right_engine              0.000   -325.045   -51.000   -325.045  crashed     51
  21  random                    0.000   -214.676   -53.000   -214.676  crashed     68
  21  ppo_then_side_tradeoff    0.000    -71.746   -38.000    -71.746  crashed     56
  21  ppo_noisy_actions         0.000     -9.990   -71.000    -71.000  crashed     84
  22  ppo_deterministic       200.000    198.586   -98.000    102.000  landed     192
  22  ppo_stochastic          200.000    199.237   -95.000    105.000  landed     185
  22  left_engine               0.000   -403.870   -59.000   -403.870  crashed     59
  22  main_engine               0.000  -1.01e+03  -164.000  -1.01e+03  crashed    164
  22  right_engine              0.000   -410.131   -59.000   -410.131  crashed     59
  22  random                    0.000   -102.717   -57.000   -102.717  crashed     73
  22  ppo_then_side_tradeoff    0.000   -390.259   -41.000   -390.259  crashed     70
  22  ppo_noisy_actions         0.000     -0.339   -53.000    -53.000  crashed     73
  23  ppo_deterministic       200.000    282.146  -120.000     80.000  landed     229
  23  ppo_stochastic          200.000    274.736  -124.000     76.000  landed     221
  23  left_engine               0.000   -423.800   -66.000   -423.800  crashed     66
  23  main_engine               0.000   -530.773   -77.000   -530.773  crashed     77
  23  right_engine              0.000   -466.278   -65.000   -466.278  crashed     65
  23  random                    0.000   -239.518   -56.000   -239.518  crashed     82
  23  ppo_then_side_tradeoff    0.000   -155.882   -48.000   -155.882  crashed     68
  23  ppo_noisy_actions         0.000     34.597   -72.000    -72.000  crashed     95
  24  ppo_deterministic       200.000    307.118   -95.000    105.000  landed     212
  24  ppo_stochastic          200.000    294.920  -111.000     89.000  landed     218
  24  left_engine               0.000   -170.923   -50.000   -170.923  crashed     50
  24  main_engine               0.000   -433.778   -77.000   -433.778  crashed     77
  24  right_engine              0.000   -228.247   -48.000   -228.247  crashed     48
  24  random                    0.000     -6.716   -55.000    -55.000  crashed     65
  24  ppo_then_side_tradeoff    0.000     18.060   -69.000    -69.000  crashed     82
  24  ppo_noisy_actions         0.000     -9.816   -50.000    -50.000  crashed     61
  25  ppo_deterministic         0.000   -146.096  -166.000   -166.000  crashed    315
  25  ppo_stochastic          200.000    277.088   -95.000    105.000  landed     202
  25  left_engine               0.000   -739.983   -78.000   -739.983  crashed     78
  25  main_engine               0.000   -841.814  -122.000   -841.814  crashed    122
  25  right_engine              0.000   -689.444   -77.000   -689.444  crashed     77
  25  random                    0.000   -318.160   -84.000   -318.160  crashed    121
  25  ppo_then_side_tradeoff    0.000   -290.350   -55.000   -290.350  crashed     80
  25  ppo_noisy_actions         0.000    -24.047   -63.000    -63.000  crashed     97
  26  ppo_deterministic       200.000    447.291  -129.000     71.000  landed     225
  26  ppo_stochastic          200.000    445.486  -125.000     75.000  landed     215
  26  left_engine               0.000   -220.300   -66.000   -220.300  crashed     66
  26  main_engine               0.000   -379.073   -80.000   -379.073  crashed     80
  26  right_engine              0.000   -490.306   -67.000   -490.306  crashed     67
  26  random                    0.000   -256.023   -70.000   -256.023  crashed     92
  26  ppo_then_side_tradeoff    0.000    -70.777   -47.000    -70.777  crashed     67
  26  ppo_noisy_actions       100.000    277.436  -237.000   -137.000  timeout   1000
  27  ppo_deterministic       200.000    355.264  -118.000     82.000  landed     232
  27  ppo_stochastic          200.000    354.892  -122.000     78.000  landed     227
  27  left_engine               0.000   -646.140   -83.000   -646.140  crashed     83
  27  main_engine               0.000   -613.966   -85.000   -613.966  crashed     85
  27  right_engine              0.000   -690.169   -81.000   -690.169  crashed     81
  27  random                    0.000   -129.557   -76.000   -129.557  crashed    107
  27  ppo_then_side_tradeoff    0.000   -373.529   -63.000   -373.529  crashed     85
  27  ppo_noisy_actions         0.000    -88.725   -83.000    -88.725  crashed    134
  28  ppo_deterministic       200.000    331.565  -109.000     91.000  landed     201
  28  ppo_stochastic          200.000    318.439  -103.000     97.000  landed     214
  28  left_engine               0.000   -126.515   -52.000   -126.515  crashed     52
  28  main_engine               0.000   -421.653  -110.000   -421.653  crashed    110
  28  right_engine              0.000   -286.982   -51.000   -286.982  crashed     51
  28  random                    0.000   -116.506   -44.000   -116.506  crashed     58
  28  ppo_then_side_tradeoff    0.000   -109.963   -43.000   -109.963  crashed     59
  28  ppo_noisy_actions         0.000     35.136   -54.000    -54.000  crashed     68
  29  ppo_deterministic       200.000    340.599  -124.000     76.000  landed     280
  29  ppo_stochastic          200.000    338.297  -126.000     74.000  landed     280
  29  left_engine               0.000   -659.146   -85.000   -659.146  crashed     85
  29  main_engine               0.000   -497.241   -76.000   -497.241  crashed     76
  29  right_engine              0.000   -783.905   -83.000   -783.905  crashed     83
  29  random                    0.000   -254.622   -72.000   -254.622  crashed    101
  29  ppo_then_side_tradeoff    0.000   -428.763   -67.000   -428.763  crashed     86
  29  ppo_noisy_actions         0.000    107.775   -81.000    -81.000  crashed    110
  30  ppo_deterministic       100.000     82.334  -504.000   -404.000  timeout   1000
  30  ppo_stochastic          200.000    330.140  -117.000     83.000  landed     209
  30  left_engine               0.000   -341.591   -62.000   -341.591  crashed     62
  30  main_engine               0.000   -503.699   -82.000   -503.699  crashed     82
  30  right_engine              0.000   -393.450   -62.000   -393.450  crashed     62
  30  random                    0.000    -63.460   -56.000    -63.460  crashed     85
  30  ppo_then_side_tradeoff    0.000   -142.879   -43.000   -142.879  crashed     64
  30  ppo_noisy_actions         0.000     26.017   -72.000    -72.000  crashed     93
  31  ppo_deterministic       200.000    341.870   -94.000    106.000  landed     200
  31  ppo_stochastic          200.000    344.295   -73.000    127.000  landed     178
  31  left_engine               0.000   -192.195   -56.000   -192.195  crashed     56
  31  main_engine               0.000   -560.530  -126.000   -560.530  crashed    126
  31  right_engine              0.000   -296.976   -53.000   -296.976  crashed     53
  31  random                    0.000     54.170   -57.000    -57.000  crashed     74
  31  ppo_then_side_tradeoff    0.000    -10.512   -35.000    -35.000  crashed     58
  31  ppo_noisy_actions         0.000     56.096   -59.000    -59.000  crashed     77
  32  ppo_deterministic       200.000    210.020   -99.000    101.000  landed     215
  32  ppo_stochastic          200.000    196.340  -101.000     99.000  landed     210
  32  left_engine               0.000   -707.132   -83.000   -707.132  crashed     83
  32  main_engine               0.000  -1.15e+03  -144.000  -1.15e+03  crashed    144
  32  right_engine              0.000   -697.937  -100.000   -697.937  crashed    100
  32  random                    0.000    -54.927  -117.000   -117.000  crashed    149
  32  ppo_then_side_tradeoff    0.000   -404.981   -56.000   -404.981  crashed     84
  32  ppo_noisy_actions         0.000    -73.395   -62.000    -73.395  crashed    100
  33  ppo_deterministic       200.000    308.118  -112.000     88.000  landed     220
  33  ppo_stochastic          100.000    179.254  -449.000   -349.000  timeout   1000
  33  left_engine               0.000   -335.581   -56.000   -335.581  crashed     56
  33  main_engine               0.000   -499.785   -85.000   -499.785  crashed     85
  33  right_engine              0.000   -359.578   -59.000   -359.578  crashed     59
  33  random                    0.000    -54.724   -55.000    -55.000  crashed     73
  33  ppo_then_side_tradeoff    0.000     -9.438   -40.000    -40.000  crashed     59
  33  ppo_noisy_actions         0.000    -17.627   -58.000    -58.000  crashed     79
  34  ppo_deterministic       200.000    197.599   -94.000    106.000  landed     231
  34  ppo_stochastic          200.000    164.233   -91.000    109.000  landed     259
  34  left_engine               0.000   -297.818   -55.000   -297.818  crashed     55
  34  main_engine               0.000   -573.656   -99.000   -573.656  crashed     99
  34  right_engine              0.000   -397.009   -54.000   -397.009  crashed     54
  34  random                    0.000   -346.914   -52.000   -346.914  crashed     74
  34  ppo_then_side_tradeoff    0.000   -197.751   -41.000   -197.751  crashed     61
  34  ppo_noisy_actions         0.000    -35.072   -62.000    -62.000  crashed     80
  35  ppo_deterministic       200.000    300.975   -99.000    101.000  landed     402
  35  ppo_stochastic          200.000    299.783  -110.000     90.000  landed     364
  35  left_engine               0.000   -356.979   -59.000   -356.979  crashed     59
  35  main_engine               0.000   -471.320   -73.000   -471.320  crashed     73
  35  right_engine              0.000   -312.240   -55.000   -312.240  crashed     55
  35  random                    0.000   -304.506   -59.000   -304.506  crashed     81
  35  ppo_then_side_tradeoff    0.000   -402.112   -71.000   -402.112  crashed     92
  35  ppo_noisy_actions         0.000    -42.241   -59.000    -59.000  crashed     74
  36  ppo_deterministic       200.000    335.240   -99.000    101.000  landed     205
  36  ppo_stochastic          200.000    332.725  -102.000     98.000  landed     213
  36  left_engine               0.000   -625.110   -83.000   -625.110  crashed     83
  36  main_engine               0.000   -897.423  -124.000   -897.423  crashed    124
  36  right_engine              0.000   -898.268   -86.000   -898.268  crashed     86
  36  random                    0.000   -292.828   -90.000   -292.828  crashed    120
  36  ppo_then_side_tradeoff    0.000   -287.659   -59.000   -287.659  crashed     85
  36  ppo_noisy_actions         0.000     69.915   -62.000    -62.000  crashed    102
  37  ppo_deterministic       200.000    339.693  -149.000     51.000  landed     196
  37  ppo_stochastic            0.000    -53.282  -164.000   -164.000  crashed    265
  37  left_engine               0.000   -372.191   -55.000   -372.191  crashed     55
  37  main_engine               0.000   -500.430  -139.000   -500.430  crashed    139
  37  right_engine              0.000   -130.431   -53.000   -130.431  crashed     53
  37  random                    0.000     77.571   -69.000    -69.000  crashed     92
  37  ppo_then_side_tradeoff    0.000    -42.693   -41.000    -42.693  crashed     59
  37  ppo_noisy_actions         0.000    107.274   -59.000    -59.000  crashed     76
  38  ppo_deterministic       200.000    244.718   -97.000    103.000  landed     206
  38  ppo_stochastic          200.000    262.081  -105.000     95.000  landed     338
  38  left_engine               0.000   -497.158   -72.000   -497.158  crashed     72
  38  main_engine               0.000   -644.461  -107.000   -644.461  crashed    107
  38  right_engine              0.000   -684.321   -81.000   -684.321  crashed     81
  38  random                    0.000   -318.730   -82.000   -318.730  crashed    112
  38  ppo_then_side_tradeoff    0.000   -225.828   -49.000   -225.828  crashed     73
  38  ppo_noisy_actions         0.000    -26.005   -65.000    -65.000  crashed     92
  39  ppo_deterministic       200.000    357.722   -97.000    103.000  landed     245
  39  ppo_stochastic          200.000    357.394   -93.000    107.000  landed     232
  39  left_engine               0.000   -383.378   -58.000   -383.378  crashed     58
  39  main_engine               0.000   -422.064   -96.000   -422.064  crashed     96
  39  right_engine              0.000   -160.570   -57.000   -160.570  crashed     57
  39  random                    0.000      6.173   -55.000    -55.000  crashed     72
  39  ppo_then_side_tradeoff    0.000    -54.259   -42.000    -54.259  crashed     61
  39  ppo_noisy_actions         0.000     89.297   -53.000    -53.000  crashed     71
  40  ppo_deterministic       200.000    221.259   -90.000    110.000  landed     183
  40  ppo_stochastic          200.000    233.223   -89.000    111.000  landed     177
  40  left_engine               0.000   -292.904   -51.000   -292.904  crashed     51
  40  main_engine               0.000   -585.891  -118.000   -585.891  crashed    118
  40  right_engine              0.000   -373.549   -50.000   -373.549  crashed     50
  40  random                    0.000     23.869   -49.000    -49.000  crashed     66
  40  ppo_then_side_tradeoff    0.000   -115.405   -38.000   -115.405  crashed     57
  40  ppo_noisy_actions         0.000    -29.852   -54.000    -54.000  crashed     70

==========================
Per candidate (seeds 1-40)
==========================
candidate               landed  crashed  timeout  mean dr  mean dn_Saf  mean dn_Fuel  margin min  median    max     
----------------------  ------  -------  -------  -------  -----------  ------------  ----------  --------  --------
ppo_deterministic        92.5%     2.5%     5.0%  190.000      287.228      -135.425    -709.000   101.000   122.000
ppo_stochastic           92.5%     2.5%     5.0%  190.000      288.053      -130.850    -349.000    97.500   127.000
left_engine               0.0%   100.0%     0.0%    0.000     -479.698       -68.450    -900.339  -441.400  -126.515
main_engine               0.0%   100.0%     0.0%    0.000     -889.527      -126.750   -3.47e+03  -567.093  -298.436
right_engine              0.0%   100.0%     0.0%    0.000     -472.662       -68.575    -898.268  -430.852  -130.431
random                    0.0%   100.0%     0.0%    0.000     -129.347       -69.250    -346.914  -116.343   -49.000
ppo_then_side_tradeoff    0.0%   100.0%     0.0%    0.000     -193.841       -52.075    -428.763  -193.502   -35.000
ppo_noisy_actions         2.5%    95.0%     2.5%    7.500       36.053       -68.925    -137.000   -62.000   106.000

===========================================================
Margin distribution and share inside the eps band [-0.5, 0]
===========================================================
set       pairs  min        median    max      in band  share
--------  -----  ---------  --------  -------  -------  -----
default     240  -3.47e+03  -282.227  127.000  0/240     0.0%
extended    320  -3.47e+03  -164.248  127.000  0/320     0.0%
smallest |margin| over all extended pairs: 26.000; pairs with |margin| < 5: 0; < 50: 6

=========================================
Admission table, raw units -- default set
=========================================
candidate          eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5  flips 0->0.5
-----------------  ------  --------  -------  -------  -------  -------  ------------
ppo_deterministic  37/40   37/40     37/40    37/40    37/40    37/40               0
ppo_stochastic     36/40   36/40     36/40    36/40    36/40    36/40               0
left_engine        0/40    0/40      0/40     0/40     0/40     0/40                0
main_engine        0/40    0/40      0/40     0/40     0/40     0/40                0
right_engine       0/40    0/40      0/40     0/40     0/40     0/40                0
random             0/40    0/40      0/40     0/40     0/40     0/40                0
ALL                73/240  73/240    73/240   73/240   73/240   73/240              0
pairs that change admission between eps=0 and eps=0.5: 0/240 (0.0%)

==========================================
Admission table, raw units -- extended set
==========================================
candidate               eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5  flips 0->0.5
----------------------  ------  --------  -------  -------  -------  -------  ------------
ppo_deterministic       37/40   37/40     37/40    37/40    37/40    37/40               0
ppo_stochastic          36/40   36/40     36/40    36/40    36/40    36/40               0
left_engine             0/40    0/40      0/40     0/40     0/40     0/40                0
main_engine             0/40    0/40      0/40     0/40     0/40     0/40                0
right_engine            0/40    0/40      0/40     0/40     0/40     0/40                0
random                  0/40    0/40      0/40     0/40     0/40     0/40                0
ppo_then_side_tradeoff  0/40    0/40      0/40     0/40     0/40     0/40                0
ppo_noisy_actions       1/40    1/40      1/40     1/40     1/40     1/40                0
ALL                     74/320  74/320    74/320   74/320   74/320   74/320              0
pairs that change admission between eps=0 and eps=0.5: 0/320 (0.0%)
```

### A.2 E1, first run (options A-H) -- e1_option_comparison.py

```text

===============================================
E1 -- option comparison on held-out seeds 21-40
===============================================
fixed scales (seeds 1-20, extended set, pooled): s_r=50.00 s_n=250.12 s_i=[404.5, 95.8]
admit iff mu >= -eps; flips = pairs whose admission differs between eps=0 and eps=0.5
crit1: flips >= 10% of pairs, >= 5 seeds, >= 2 candidates | crit2: no crash admitted at any eps
crit3: >= 95% of ppo_deterministic landings admitted at eps=0.1 | crit4: 0 monotonicity violations

==============================
Summary -- task's extended set
==============================
option       flips   share  seeds  cands  crash adm  det@0.1  mono viol  c1    c2    c3    c4  
-----------  ------  -----  -----  -----  ---------  -------  ---------  ----  ----  ----  ----
F            0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
A-fixed      39/160  24.4%     20      4  38/121     18/18            0  PASS  fail  PASS  PASS
A-per-step   36/160  22.5%     20      5  34/121     18/18            0  PASS  fail  PASS  PASS
B-fixed      12/160   7.5%     10      3  11/121     18/18            0  fail  fail  PASS  PASS
B-per-step   5/160    3.1%      4      2  5/121      18/18            0  fail  fail  PASS  PASS
C1-fixed     38/160  23.8%     20      3  38/121     18/18            0  PASS  fail  PASS  PASS
C1-per-step  34/160  21.2%     20      3  34/121     18/18            0  PASS  fail  PASS  PASS
C2-fixed     43/160  26.9%     20      5  42/121     18/18            0  PASS  fail  PASS  PASS
C2-per-step  37/160  23.1%     20      3  36/121     18/18            0  PASS  fail  PASS  PASS
G-fixed      42/160  26.2%     20      5  41/121     18/18            0  PASS  fail  PASS  PASS
G-per-step   36/160  22.5%     20      3  36/121     18/18            0  PASS  fail  PASS  PASS
D            16/160  10.0%      9      3  16/121     18/18            0  PASS  fail  PASS  PASS
E            60/160  37.5%     20      5  43/121     10/18            0  PASS  fail  fail  PASS
H            5/160    3.1%      5      3  5/121      18/18            0  fail  fail  PASS  PASS

===================================================
Admissions over the eps grid -- task's extended set
===================================================
option       eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5
-----------  ------  --------  -------  -------  -------  -------
F            36/160  36/160    36/160   36/160   36/160   36/160 
A-fixed      38/160  39/160    39/160   44/160   70/160   77/160 
A-per-step   37/160  38/160    38/160   39/160   47/160   73/160 
B-fixed      36/160  36/160    36/160   36/160   36/160   48/160 
B-per-step   36/160  36/160    36/160   36/160   36/160   41/160 
C1-fixed     36/160  36/160    36/160   41/160   67/160   74/160 
C1-per-step  36/160  36/160    36/160   36/160   44/160   70/160 
C2-fixed     36/160  36/160    36/160   54/160   69/160   79/160 
C2-per-step  36/160  36/160    36/160   39/160   61/160   73/160 
G-fixed      36/160  36/160    36/160   41/160   67/160   78/160 
G-per-step   36/160  36/160    36/160   36/160   45/160   72/160 
D            36/160  36/160    36/160   36/160   37/160   52/160 
E            20/160  20/160    20/160   40/160   60/160   80/160 
H            36/160  36/160    36/160   36/160   36/160   41/160 

================================================================================================
Which candidates flip (count by candidate; outcome of the flipping pairs) -- task's extended set
================================================================================================
option       flipping candidates                                                                                flip outcomes                   
-----------  -------------------------------------------------------------------------------------------------  --------------------------------
F            -                                                                                                  -                               
A-fixed      ppo_deterministic:1, ppo_noisy_actions:19, ppo_then_side_tradeoff:9, random:10                     crashed:38, timeout:1           
A-per-step   ppo_deterministic:1, ppo_noisy_actions:19, ppo_stochastic:1, ppo_then_side_tradeoff:6, random:9    crashed:34, timeout:2           
B-fixed      ppo_noisy_actions:1, ppo_then_side_tradeoff:10, random:1                                           crashed:11, timeout:1           
B-per-step   ppo_then_side_tradeoff:3, random:2                                                                 crashed:5                       
C1-fixed     ppo_noisy_actions:19, ppo_then_side_tradeoff:9, random:10                                          crashed:38                      
C1-per-step  ppo_noisy_actions:19, ppo_then_side_tradeoff:6, random:9                                           crashed:34                      
C2-fixed     left_engine:1, ppo_noisy_actions:20, ppo_then_side_tradeoff:10, random:11, right_engine:1          crashed:42, timeout:1           
C2-per-step  ppo_noisy_actions:20, ppo_then_side_tradeoff:7, random:10                                          crashed:36, timeout:1           
G-fixed      left_engine:1, ppo_noisy_actions:20, ppo_then_side_tradeoff:9, random:11, right_engine:1           crashed:41, timeout:1           
G-per-step   ppo_noisy_actions:19, ppo_then_side_tradeoff:7, random:10                                          crashed:36                      
D            ppo_noisy_actions:7, ppo_then_side_tradeoff:5, random:4                                            crashed:16                      
E            ppo_deterministic:9, ppo_noisy_actions:19, ppo_stochastic:8, ppo_then_side_tradeoff:13, random:11  crashed:43, landed:16, timeout:1
H            ppo_noisy_actions:1, ppo_then_side_tradeoff:3, random:1                                            crashed:5                       

===========================================
Summary -- extended + borderline candidates
===========================================
option       flips    share  seeds  cands  crash adm  det@0.1  mono viol  c1    c2    c3    c4  
-----------  -------  -----  -----  -----  ---------  -------  ---------  ----  ----  ----  ----
F            0/280     0.0%      0      0  0/207      18/18            0  fail  PASS  PASS  PASS
A-fixed      123/280  43.9%     20     10  122/207    18/18            0  PASS  fail  PASS  PASS
A-per-step   88/280   31.4%     20      9  87/207     18/18            0  PASS  fail  PASS  PASS
B-fixed      35/280   12.5%     20      6  33/207     18/18            0  PASS  fail  PASS  PASS
B-per-step   26/280    9.3%     20      5  26/207     18/18            0  fail  fail  PASS  PASS
C1-fixed     127/280  45.4%     20      9  122/207    18/18            0  PASS  fail  PASS  PASS
C1-per-step  91/280   32.5%     20      9  87/207     18/18            0  PASS  fail  PASS  PASS
C2-fixed     133/280  47.5%     20     11  126/207    18/18            0  PASS  fail  PASS  PASS
C2-per-step  119/280  42.5%     20      9  114/207    18/18            0  PASS  fail  PASS  PASS
G-fixed      132/280  47.1%     20     11  125/207    18/18            0  PASS  fail  PASS  PASS
G-per-step   106/280  37.9%     20      9  101/207    18/18            0  PASS  fail  PASS  PASS
D            51/280   18.2%     19      7  50/207     18/18            0  PASS  fail  PASS  PASS
E            121/280  43.2%     20     11  74/207     9/18             0  PASS  fail  fail  PASS
H            24/280    8.6%     17      6  23/207     18/18            0  fail  fail  PASS  PASS

flips counted on the borderline candidates only (what the option does when such skills exist):
option       flips   share  seeds  cands  crash adm
-----------  ------  -----  -----  -----  ---------
F            0/120    0.0%      0      0  0/86     
A-fixed      84/120  70.0%     20      6  84/86    
A-per-step   59/120  49.2%     20      6  58/86    
B-fixed      23/120  19.2%     20      3  22/86    
B-per-step   21/120  17.5%     20      2  21/86    
C1-fixed     89/120  74.2%     20      6  84/86    
C1-per-step  62/120  51.7%     20      6  58/86    
C2-fixed     90/120  75.0%     20      6  84/86    
C2-per-step  86/120  71.7%     20      6  81/86    
G-fixed      90/120  75.0%     20      6  84/86    
G-per-step   74/120  61.7%     20      6  69/86    
D            35/120  29.2%     19      4  34/86    
E            73/120  60.8%     20      6  50/86    
H            19/120  15.8%     17      3  18/86    

================================================================
Admissions over the eps grid -- extended + borderline candidates
================================================================
option       eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5
-----------  ------  --------  -------  -------  -------  -------
F            63/280  63/280    63/280   63/280   63/280   63/280 
A-fixed      72/280  76/280    78/280   96/280   165/280  195/280
A-per-step   70/280  73/280    73/280   77/280   89/280   158/280
B-fixed      68/280  70/280    71/280   75/280   80/280   103/280
B-per-step   68/280  70/280    71/280   73/280   78/280   94/280 
C1-fixed     63/280  66/280    68/280   87/280   159/280  190/280
C1-per-step  63/280  66/280    66/280   70/280   83/280   154/280
C2-fixed     63/280  66/280    72/280   114/280  176/280  196/280
C2-per-step  63/280  66/280    67/280   79/280   117/280  182/280
G-fixed      63/280  66/280    68/280   87/280   160/280  195/280
G-per-step   63/280  66/280    66/280   70/280   84/280   169/280
D            63/280  65/280    66/280   68/280   76/280   114/280
E            19/280  21/280    35/280   62/280   80/280   140/280
H            63/280  65/280    66/280   67/280   72/280   87/280 

=============================================================================================================
Which candidates flip (count by candidate; outcome of the flipping pairs) -- extended + borderline candidates
=============================================================================================================
option       flipping candidates                                                                                                                                                                                                         flip outcomes                   
-----------  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  --------------------------------
F            -                                                                                                                                                                                                                           -                               
A-fixed      ppo_deterministic:1, ppo_greedy_p10:5, ppo_greedy_p20:15, ppo_greedy_p30:19, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:10                      crashed:122, timeout:1          
A-per-step   ppo_greedy_p10:1, ppo_greedy_p20:12, ppo_greedy_p30:15, ppo_noisy_actions:17, ppo_then_noop_100:10, ppo_then_noop_150:2, ppo_then_noop_50:19, ppo_then_side_tradeoff:6, random:6                                            crashed:87, timeout:1           
B-fixed      ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, random:1                                                                                                        crashed:33, timeout:2           
B-per-step   ppo_greedy_p30:1, ppo_noisy_actions:1, ppo_then_noop_50:20, ppo_then_side_tradeoff:3, random:1                                                                                                                              crashed:26                      
C1-fixed     ppo_greedy_p10:6, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:10                                           crashed:122, landed:5           
C1-per-step  ppo_greedy_p10:2, ppo_greedy_p20:13, ppo_greedy_p30:16, ppo_noisy_actions:17, ppo_then_noop_100:10, ppo_then_noop_150:2, ppo_then_noop_50:19, ppo_then_side_tradeoff:6, random:6                                            crashed:87, landed:4            
C2-fixed     left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, random:11, right_engine:1           crashed:126, landed:5, timeout:2
C2-per-step  ppo_greedy_p10:6, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:6, ppo_then_noop_50:20, ppo_then_side_tradeoff:7, random:7                                            crashed:114, landed:5           
G-fixed      left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:11, right_engine:1            crashed:125, landed:5, timeout:2
G-per-step   ppo_greedy_p10:4, ppo_greedy_p20:17, ppo_greedy_p30:18, ppo_noisy_actions:19, ppo_then_noop_100:10, ppo_then_noop_150:5, ppo_then_noop_50:20, ppo_then_side_tradeoff:6, random:7                                            crashed:101, landed:5           
D            ppo_greedy_p20:7, ppo_greedy_p30:5, ppo_noisy_actions:7, ppo_then_noop_100:5, ppo_then_noop_50:18, ppo_then_side_tradeoff:5, random:4                                                                                       crashed:50, landed:1            
E            ppo_deterministic:14, ppo_greedy_p10:11, ppo_greedy_p20:11, ppo_greedy_p30:11, ppo_noisy_actions:16, ppo_stochastic:10, ppo_then_noop_100:9, ppo_then_noop_150:11, ppo_then_noop_50:20, ppo_then_side_tradeoff:5, random:3  crashed:74, landed:47           
H            ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:16, ppo_then_side_tradeoff:3, random:1                                                                                                         crashed:23, landed:1
```

### A.3 E1, final run -- e1_option_comparison.py

```text

===============================================
E1 -- option comparison on held-out seeds 21-40
===============================================
fixed scales (seeds 1-20, extended set, pooled): s_r=50.00 s_n=250.12 s_i=[404.5, 95.8]
admit iff mu >= -eps; flips = pairs whose admission differs between eps=0 and eps=0.5
crit1: flips >= 10% of pairs, >= 5 seeds, >= 2 candidates | crit2: no crash admitted at any eps
crit3: >= 95% of ppo_deterministic landings admitted at eps=0.1 | crit4: 0 monotonicity violations

==============================
Summary -- task's extended set
==============================
option         flips   share  seeds  cands  crash adm  det@0.1  mono viol  c1    c2    c3    c4  
-------------  ------  -----  -----  -----  ---------  -------  ---------  ----  ----  ----  ----
F              0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
A-fixed        39/160  24.4%     20      4  38/121     18/18            0  PASS  fail  PASS  PASS
A-per-step     36/160  22.5%     20      5  34/121     18/18            0  PASS  fail  PASS  PASS
B-fixed        12/160   7.5%     10      3  11/121     18/18            0  fail  fail  PASS  PASS
B-per-step     5/160    3.1%      4      2  5/121      18/18            0  fail  fail  PASS  PASS
C1-fixed       38/160  23.8%     20      3  38/121     18/18            0  PASS  fail  PASS  PASS
C1-per-step    34/160  21.2%     20      3  34/121     18/18            0  PASS  fail  PASS  PASS
C2-fixed       43/160  26.9%     20      5  42/121     18/18            0  PASS  fail  PASS  PASS
C2-per-step    37/160  23.1%     20      3  36/121     18/18            0  PASS  fail  PASS  PASS
G-fixed        42/160  26.2%     20      5  41/121     18/18            0  PASS  fail  PASS  PASS
G-per-step     36/160  22.5%     20      3  36/121     18/18            0  PASS  fail  PASS  PASS
D              16/160  10.0%      9      3  16/121     18/18            0  PASS  fail  PASS  PASS
E              60/160  37.5%     20      5  43/121     10/18            0  PASS  fail  fail  PASS
H              5/160    3.1%      5      3  5/121      18/18            0  fail  fail  PASS  PASS
I:C1-fixed     0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
I:C1-per-step  0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
I:C2-fixed     1/160    0.6%      1      1  0/121      18/18            0  fail  PASS  PASS  PASS
I:C2-per-step  1/160    0.6%      1      1  0/121      18/18            0  fail  PASS  PASS  PASS
I:G-fixed      1/160    0.6%      1      1  0/121      18/18            0  fail  PASS  PASS  PASS
I:G-per-step   0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
I:D            0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS
I:H            0/160    0.0%      0      0  0/121      18/18            0  fail  PASS  PASS  PASS

===================================================
Admissions over the eps grid -- task's extended set
===================================================
option         eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5
-------------  ------  --------  -------  -------  -------  -------
F              36/160  36/160    36/160   36/160   36/160   36/160 
A-fixed        38/160  39/160    39/160   44/160   70/160   77/160 
A-per-step     37/160  38/160    38/160   39/160   47/160   73/160 
B-fixed        36/160  36/160    36/160   36/160   36/160   48/160 
B-per-step     36/160  36/160    36/160   36/160   36/160   41/160 
C1-fixed       36/160  36/160    36/160   41/160   67/160   74/160 
C1-per-step    36/160  36/160    36/160   36/160   44/160   70/160 
C2-fixed       36/160  36/160    36/160   54/160   69/160   79/160 
C2-per-step    36/160  36/160    36/160   39/160   61/160   73/160 
G-fixed        36/160  36/160    36/160   41/160   67/160   78/160 
G-per-step     36/160  36/160    36/160   36/160   45/160   72/160 
D              36/160  36/160    36/160   36/160   37/160   52/160 
E              20/160  20/160    20/160   40/160   60/160   80/160 
H              36/160  36/160    36/160   36/160   36/160   41/160 
I:C1-fixed     36/160  36/160    36/160   36/160   36/160   36/160 
I:C1-per-step  36/160  36/160    36/160   36/160   36/160   36/160 
I:C2-fixed     36/160  36/160    36/160   36/160   36/160   37/160 
I:C2-per-step  36/160  36/160    36/160   36/160   36/160   37/160 
I:G-fixed      36/160  36/160    36/160   36/160   36/160   37/160 
I:G-per-step   36/160  36/160    36/160   36/160   36/160   36/160 
I:D            36/160  36/160    36/160   36/160   36/160   36/160 
I:H            36/160  36/160    36/160   36/160   36/160   36/160 

================================================================================================
Which candidates flip (count by candidate; outcome of the flipping pairs) -- task's extended set
================================================================================================
option         flipping candidates                                                                                flip outcomes                   
-------------  -------------------------------------------------------------------------------------------------  --------------------------------
F              -                                                                                                  -                               
A-fixed        ppo_deterministic:1, ppo_noisy_actions:19, ppo_then_side_tradeoff:9, random:10                     crashed:38, timeout:1           
A-per-step     ppo_deterministic:1, ppo_noisy_actions:19, ppo_stochastic:1, ppo_then_side_tradeoff:6, random:9    crashed:34, timeout:2           
B-fixed        ppo_noisy_actions:1, ppo_then_side_tradeoff:10, random:1                                           crashed:11, timeout:1           
B-per-step     ppo_then_side_tradeoff:3, random:2                                                                 crashed:5                       
C1-fixed       ppo_noisy_actions:19, ppo_then_side_tradeoff:9, random:10                                          crashed:38                      
C1-per-step    ppo_noisy_actions:19, ppo_then_side_tradeoff:6, random:9                                           crashed:34                      
C2-fixed       left_engine:1, ppo_noisy_actions:20, ppo_then_side_tradeoff:10, random:11, right_engine:1          crashed:42, timeout:1           
C2-per-step    ppo_noisy_actions:20, ppo_then_side_tradeoff:7, random:10                                          crashed:36, timeout:1           
G-fixed        left_engine:1, ppo_noisy_actions:20, ppo_then_side_tradeoff:9, random:11, right_engine:1           crashed:41, timeout:1           
G-per-step     ppo_noisy_actions:19, ppo_then_side_tradeoff:7, random:10                                          crashed:36                      
D              ppo_noisy_actions:7, ppo_then_side_tradeoff:5, random:4                                            crashed:16                      
E              ppo_deterministic:9, ppo_noisy_actions:19, ppo_stochastic:8, ppo_then_side_tradeoff:13, random:11  crashed:43, landed:16, timeout:1
H              ppo_noisy_actions:1, ppo_then_side_tradeoff:3, random:1                                            crashed:5                       
I:C1-fixed     -                                                                                                  -                               
I:C1-per-step  -                                                                                                  -                               
I:C2-fixed     ppo_noisy_actions:1                                                                                timeout:1                       
I:C2-per-step  ppo_noisy_actions:1                                                                                timeout:1                       
I:G-fixed      ppo_noisy_actions:1                                                                                timeout:1                       
I:G-per-step   -                                                                                                  -                               
I:D            -                                                                                                  -                               
I:H            -                                                                                                  -                               

===========================================
Summary -- extended + borderline candidates
===========================================
option         flips    share  seeds  cands  crash adm  det@0.1  mono viol  c1    c2    c3    c4  
-------------  -------  -----  -----  -----  ---------  -------  ---------  ----  ----  ----  ----
F              0/280     0.0%      0      0  0/207      18/18            0  fail  PASS  PASS  PASS
A-fixed        123/280  43.9%     20     10  122/207    18/18            0  PASS  fail  PASS  PASS
A-per-step     88/280   31.4%     20      9  87/207     18/18            0  PASS  fail  PASS  PASS
B-fixed        35/280   12.5%     20      6  33/207     18/18            0  PASS  fail  PASS  PASS
B-per-step     26/280    9.3%     20      5  26/207     18/18            0  fail  fail  PASS  PASS
C1-fixed       127/280  45.4%     20      9  122/207    18/18            0  PASS  fail  PASS  PASS
C1-per-step    91/280   32.5%     20      9  87/207     18/18            0  PASS  fail  PASS  PASS
C2-fixed       133/280  47.5%     20     11  126/207    18/18            0  PASS  fail  PASS  PASS
C2-per-step    119/280  42.5%     20      9  114/207    18/18            0  PASS  fail  PASS  PASS
G-fixed        132/280  47.1%     20     11  125/207    18/18            0  PASS  fail  PASS  PASS
G-per-step     106/280  37.9%     20      9  101/207    18/18            0  PASS  fail  PASS  PASS
D              51/280   18.2%     19      7  50/207     18/18            0  PASS  fail  PASS  PASS
E              121/280  43.2%     20     11  74/207     9/18             0  PASS  fail  fail  PASS
H              24/280    8.6%     17      6  23/207     18/18            0  fail  fail  PASS  PASS
I:C1-fixed     5/280     1.8%      5      3  0/207      18/18            0  fail  PASS  PASS  PASS
I:C1-per-step  4/280     1.4%      4      3  0/207      18/18            0  fail  PASS  PASS  PASS
I:C2-fixed     7/280     2.5%      6      4  0/207      18/18            0  fail  PASS  PASS  PASS
I:C2-per-step  5/280     1.8%      5      3  0/207      18/18            0  fail  PASS  PASS  PASS
I:G-fixed      7/280     2.5%      6      4  0/207      18/18            0  fail  PASS  PASS  PASS
I:G-per-step   5/280     1.8%      5      3  0/207      18/18            0  fail  PASS  PASS  PASS
I:D            1/280     0.4%      1      1  0/207      18/18            0  fail  PASS  PASS  PASS
I:H            1/280     0.4%      1      1  0/207      18/18            0  fail  PASS  PASS  PASS

flips counted on the added candidates only (ppo_greedy_p10, ppo_greedy_p20, ppo_greedy_p30, ppo_then_noop_50, ppo_then_noop_100, ppo_then_noop_150):
option         flips   share  seeds  cands  crash adm
-------------  ------  -----  -----  -----  ---------
F              0/120    0.0%      0      0  0/86     
A-fixed        84/120  70.0%     20      6  84/86    
A-per-step     59/120  49.2%     20      6  58/86    
B-fixed        23/120  19.2%     20      3  22/86    
B-per-step     21/120  17.5%     20      2  21/86    
C1-fixed       89/120  74.2%     20      6  84/86    
C1-per-step    62/120  51.7%     20      6  58/86    
C2-fixed       90/120  75.0%     20      6  84/86    
C2-per-step    86/120  71.7%     20      6  81/86    
G-fixed        90/120  75.0%     20      6  84/86    
G-per-step     74/120  61.7%     20      6  69/86    
D              35/120  29.2%     19      4  34/86    
E              73/120  60.8%     20      6  50/86    
H              19/120  15.8%     17      3  18/86    
I:C1-fixed     5/120    4.2%      5      3  0/86     
I:C1-per-step  4/120    3.3%      4      3  0/86     
I:C2-fixed     6/120    5.0%      5      3  0/86     
I:C2-per-step  5/120    4.2%      5      3  0/86     
I:G-fixed      6/120    5.0%      5      3  0/86     
I:G-per-step   5/120    4.2%      5      3  0/86     
I:D            1/120    0.8%      1      1  0/86     
I:H            1/120    0.8%      1      1  0/86     

================================================================
Admissions over the eps grid -- extended + borderline candidates
================================================================
option         eps=0   eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5
-------------  ------  --------  -------  -------  -------  -------
F              63/280  63/280    63/280   63/280   63/280   63/280 
A-fixed        72/280  76/280    78/280   96/280   165/280  195/280
A-per-step     70/280  73/280    73/280   77/280   89/280   158/280
B-fixed        68/280  70/280    71/280   75/280   80/280   103/280
B-per-step     68/280  70/280    71/280   73/280   78/280   94/280 
C1-fixed       63/280  66/280    68/280   87/280   159/280  190/280
C1-per-step    63/280  66/280    66/280   70/280   83/280   154/280
C2-fixed       63/280  66/280    72/280   114/280  176/280  196/280
C2-per-step    63/280  66/280    67/280   79/280   117/280  182/280
G-fixed        63/280  66/280    68/280   87/280   160/280  195/280
G-per-step     63/280  66/280    66/280   70/280   84/280   169/280
D              63/280  65/280    66/280   68/280   76/280   114/280
E              19/280  21/280    35/280   62/280   80/280   140/280
H              63/280  65/280    66/280   67/280   72/280   87/280 
I:C1-fixed     63/280  63/280    63/280   64/280   67/280   68/280 
I:C1-per-step  63/280  63/280    63/280   63/280   64/280   67/280 
I:C2-fixed     63/280  63/280    63/280   64/280   68/280   70/280 
I:C2-per-step  63/280  63/280    63/280   64/280   67/280   68/280 
I:G-fixed      63/280  63/280    63/280   64/280   67/280   70/280 
I:G-per-step   63/280  63/280    63/280   63/280   64/280   68/280 
I:D            63/280  63/280    63/280   63/280   63/280   64/280 
I:H            63/280  63/280    63/280   63/280   63/280   64/280 

=============================================================================================================
Which candidates flip (count by candidate; outcome of the flipping pairs) -- extended + borderline candidates
=============================================================================================================
option         flipping candidates                                                                                                                                                                                                         flip outcomes                   
-------------  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  --------------------------------
F              -                                                                                                                                                                                                                           -                               
A-fixed        ppo_deterministic:1, ppo_greedy_p10:5, ppo_greedy_p20:15, ppo_greedy_p30:19, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:10                      crashed:122, timeout:1          
A-per-step     ppo_greedy_p10:1, ppo_greedy_p20:12, ppo_greedy_p30:15, ppo_noisy_actions:17, ppo_then_noop_100:10, ppo_then_noop_150:2, ppo_then_noop_50:19, ppo_then_side_tradeoff:6, random:6                                            crashed:87, timeout:1           
B-fixed        ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, random:1                                                                                                        crashed:33, timeout:2           
B-per-step     ppo_greedy_p30:1, ppo_noisy_actions:1, ppo_then_noop_50:20, ppo_then_side_tradeoff:3, random:1                                                                                                                              crashed:26                      
C1-fixed       ppo_greedy_p10:6, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:10                                           crashed:122, landed:5           
C1-per-step    ppo_greedy_p10:2, ppo_greedy_p20:13, ppo_greedy_p30:16, ppo_noisy_actions:17, ppo_then_noop_100:10, ppo_then_noop_150:2, ppo_then_noop_50:19, ppo_then_side_tradeoff:6, random:6                                            crashed:87, landed:4            
C2-fixed       left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, random:11, right_engine:1           crashed:126, landed:5, timeout:2
C2-per-step    ppo_greedy_p10:6, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:6, ppo_then_noop_50:20, ppo_then_side_tradeoff:7, random:7                                            crashed:114, landed:5           
G-fixed        left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, random:11, right_engine:1            crashed:125, landed:5, timeout:2
G-per-step     ppo_greedy_p10:4, ppo_greedy_p20:17, ppo_greedy_p30:18, ppo_noisy_actions:19, ppo_then_noop_100:10, ppo_then_noop_150:5, ppo_then_noop_50:20, ppo_then_side_tradeoff:6, random:7                                            crashed:101, landed:5           
D              ppo_greedy_p20:7, ppo_greedy_p30:5, ppo_noisy_actions:7, ppo_then_noop_100:5, ppo_then_noop_50:18, ppo_then_side_tradeoff:5, random:4                                                                                       crashed:50, landed:1            
E              ppo_deterministic:14, ppo_greedy_p10:11, ppo_greedy_p20:11, ppo_greedy_p30:11, ppo_noisy_actions:16, ppo_stochastic:10, ppo_then_noop_100:9, ppo_then_noop_150:11, ppo_then_noop_50:20, ppo_then_side_tradeoff:5, random:3  crashed:74, landed:47           
H              ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:16, ppo_then_side_tradeoff:3, random:1                                                                                                         crashed:23, landed:1            
I:C1-fixed     ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1                                                                                                                                                                        landed:5                        
I:C1-per-step  ppo_greedy_p10:1, ppo_greedy_p20:2, ppo_greedy_p30:1                                                                                                                                                                        landed:4                        
I:C2-fixed     ppo_greedy_p10:1, ppo_greedy_p20:4, ppo_greedy_p30:1, ppo_noisy_actions:1                                                                                                                                                   landed:5, timeout:2             
I:C2-per-step  ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1                                                                                                                                                                        landed:5                        
I:G-fixed      ppo_greedy_p10:1, ppo_greedy_p20:4, ppo_greedy_p30:1, ppo_noisy_actions:1                                                                                                                                                   landed:5, timeout:2             
I:G-per-step   ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1                                                                                                                                                                        landed:5                        
I:D            ppo_greedy_p20:1                                                                                                                                                                                                            landed:1                        
I:H            ppo_greedy_p20:1                                                                                                                                                                                                            landed:1                        

======================================================
Summary -- extended + borderline + tradeoff candidates
======================================================
option         flips    share  seeds  cands  crash adm  det@0.1  mono viol  c1    c2    c3    c4  
-------------  -------  -----  -----  -----  ---------  -------  ---------  ----  ----  ----  ----
F              0/400     0.0%      0      0  0/219      18/18            0  fail  PASS  PASS  PASS
A-fixed        124/400  31.0%     20     11  122/219    18/18            0  PASS  fail  PASS  PASS
A-per-step     134/400  33.5%     20     14  111/219    18/18            0  PASS  fail  PASS  PASS
B-fixed        37/400    9.2%     20      8  33/219     18/18            0  fail  fail  PASS  PASS
B-per-step     91/400   22.8%     20     13  73/219     18/18            0  PASS  fail  PASS  PASS
C1-fixed       146/400  36.5%     20     15  122/219    18/18            0  PASS  fail  PASS  PASS
C1-per-step    132/400  33.0%     20     15  111/219    18/18            0  PASS  fail  PASS  PASS
C2-fixed       160/400  40.0%     20     17  126/219    18/18            0  PASS  fail  PASS  PASS
C2-per-step    151/400  37.8%     20     16  123/219    18/18            0  PASS  fail  PASS  PASS
G-fixed        153/400  38.2%     20     17  125/219    18/18            0  PASS  fail  PASS  PASS
G-per-step     137/400  34.2%     20     15  114/219    18/18            0  PASS  fail  PASS  PASS
D              64/400   16.0%     19     13  50/219     18/18            0  PASS  fail  PASS  PASS
E              181/400  45.2%     20     17  87/219     14/18            0  PASS  fail  fail  PASS
H              35/400    8.8%     19     12  23/219     18/18            0  fail  fail  PASS  PASS
I:C1-fixed     24/400    6.0%     14      9  0/219      18/18            0  fail  PASS  PASS  PASS
I:C1-per-step  21/400    5.2%     12      9  0/219      18/18            0  fail  PASS  PASS  PASS
I:C2-fixed     34/400    8.5%     15     10  0/219      18/18            0  fail  PASS  PASS  PASS
I:C2-per-step  28/400    7.0%     14     10  0/219      18/18            0  fail  PASS  PASS  PASS
I:G-fixed      28/400    7.0%     14     10  0/219      18/18            0  fail  PASS  PASS  PASS
I:G-per-step   23/400    5.8%     13      9  0/219      18/18            0  fail  PASS  PASS  PASS
I:D            14/400    3.5%     10      7  0/219      18/18            0  fail  PASS  PASS  PASS
I:H            12/400    3.0%      9      7  0/219      18/18            0  fail  PASS  PASS  PASS

flips counted on the added candidates only (ppo_waste_main_12, ppo_waste_main_15, ppo_waste_main_18, ppo_waste_main_20, ppo_waste_side_15, ppo_waste_side_20):
option         flips   share  seeds  cands  crash adm
-------------  ------  -----  -----  -----  ---------
F              0/120    0.0%      0      0  0/12     
A-fixed        1/120    0.8%      1      1  0/12     
A-per-step     22/120  18.3%     15      5  0/12     
B-fixed        2/120    1.7%      2      2  0/12     
B-per-step     16/120  13.3%     11      5  0/12     
C1-fixed       19/120  15.8%     13      6  0/12     
C1-per-step    16/120  13.3%     11      6  0/12     
C2-fixed       27/120  22.5%     15      6  0/12     
C2-per-step    21/120  17.5%     13      6  0/12     
G-fixed        21/120  17.5%     13      6  0/12     
G-per-step     18/120  15.0%     12      6  0/12     
D              13/120  10.8%     10      6  0/12     
E              47/120  39.2%     19      6  0/12     
H              11/120   9.2%      9      6  0/12     
I:C1-fixed     19/120  15.8%     13      6  0/12     
I:C1-per-step  16/120  13.3%     11      6  0/12     
I:C2-fixed     27/120  22.5%     15      6  0/12     
I:C2-per-step  21/120  17.5%     13      6  0/12     
I:G-fixed      21/120  17.5%     13      6  0/12     
I:G-per-step   18/120  15.0%     12      6  0/12     
I:D            13/120  10.8%     10      6  0/12     
I:H            11/120   9.2%      9      6  0/12     

===========================================================================
Admissions over the eps grid -- extended + borderline + tradeoff candidates
===========================================================================
option         eps=0    eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5
-------------  -------  --------  -------  -------  -------  -------
F              95/400   95/400    95/400   95/400   95/400   95/400 
A-fixed        179/400  183/400   186/400  204/400  273/400  303/400
A-per-step     137/400  141/400   144/400  157/400  179/400  271/400
B-fixed        122/400  124/400   125/400  130/400  136/400  159/400
B-per-step     118/400  121/400   123/400  132/400  148/400  209/400
C1-fixed       95/400   100/400   106/400  130/400  204/400  241/400
C1-per-step    95/400   100/400   101/400  117/400  140/400  227/400
C2-fixed       95/400   100/400   111/400  159/400  225/400  255/400
C2-per-step    95/400   100/400   107/400  134/400  202/400  246/400
G-fixed        95/400   100/400   106/400  130/400  205/400  248/400
G-per-step     95/400   100/400   101/400  118/400  146/400  232/400
D              95/400   98/400    100/400  106/400  117/400  159/400
E              19/400   21/400    45/400   80/400   121/400  200/400
H              95/400   98/400    100/400  103/400  111/400  130/400
I:C1-fixed     95/400   97/400    101/400  107/400  112/400  119/400
I:C1-per-step  95/400   97/400    98/400   104/400  109/400  116/400
I:C2-fixed     95/400   97/400    102/400  109/400  117/400  129/400
I:C2-per-step  95/400   97/400    101/400  107/400  114/400  123/400
I:G-fixed      95/400   97/400    101/400  107/400  112/400  123/400
I:G-per-step   95/400   97/400    98/400   104/400  110/400  118/400
I:D            95/400   96/400    97/400   101/400  104/400  109/400
I:H            95/400   96/400    97/400   99/400   102/400  107/400

========================================================================================================================
Which candidates flip (count by candidate; outcome of the flipping pairs) -- extended + borderline + tradeoff candidates
========================================================================================================================
option         flipping candidates                                                                                                                                                                                                                                                                                                                                        flip outcomes                    
-------------  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  ---------------------------------
F              -                                                                                                                                                                                                                                                                                                                                                          -                                
A-fixed        ppo_deterministic:1, ppo_greedy_p10:5, ppo_greedy_p20:15, ppo_greedy_p30:19, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, ppo_waste_side_20:1, random:10                                                                                                                                crashed:122, timeout:2           
A-per-step     ppo_greedy_p10:4, ppo_greedy_p20:15, ppo_greedy_p30:19, ppo_noisy_actions:19, ppo_then_noop_100:15, ppo_then_noop_150:6, ppo_then_noop_50:20, ppo_then_side_tradeoff:7, ppo_waste_main_15:6, ppo_waste_main_18:5, ppo_waste_main_20:5, ppo_waste_side_15:1, ppo_waste_side_20:5, random:7                                                                  crashed:111, landed:2, timeout:21
B-fixed        ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, ppo_waste_main_12:1, ppo_waste_side_20:1, random:1                                                                                                                                                                                             crashed:33, landed:2, timeout:2  
B-per-step     ppo_greedy_p10:2, ppo_greedy_p20:7, ppo_greedy_p30:10, ppo_noisy_actions:14, ppo_then_noop_100:8, ppo_then_noop_50:20, ppo_then_side_tradeoff:8, ppo_waste_main_15:5, ppo_waste_main_18:4, ppo_waste_main_20:2, ppo_waste_side_15:2, ppo_waste_side_20:3, random:6                                                                                         crashed:73, landed:8, timeout:10 
C1-fixed       ppo_greedy_p10:6, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:4, ppo_waste_side_20:4, random:10                                            crashed:122, landed:24           
C1-per-step    ppo_greedy_p10:5, ppo_greedy_p20:17, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:15, ppo_then_noop_150:6, ppo_then_noop_50:20, ppo_then_side_tradeoff:7, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:3, ppo_waste_side_20:3, random:7                                             crashed:111, landed:21           
C2-fixed       left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:10, ppo_waste_main_12:2, ppo_waste_main_15:8, ppo_waste_main_18:3, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:5, random:11, right_engine:1            crashed:126, landed:27, timeout:7
C2-per-step    left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:4, random:10                             crashed:123, landed:26, timeout:2
G-fixed        left_engine:1, ppo_greedy_p10:6, ppo_greedy_p20:19, ppo_greedy_p30:20, ppo_noisy_actions:20, ppo_then_noop_100:16, ppo_then_noop_150:9, ppo_then_noop_50:20, ppo_then_side_tradeoff:9, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:4, random:11, right_engine:1             crashed:125, landed:26, timeout:2
G-per-step     ppo_greedy_p10:5, ppo_greedy_p20:18, ppo_greedy_p30:20, ppo_noisy_actions:19, ppo_then_noop_100:16, ppo_then_noop_150:7, ppo_then_noop_50:20, ppo_then_side_tradeoff:7, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:3, ppo_waste_side_20:4, random:7                                             crashed:114, landed:23           
D              ppo_greedy_p20:7, ppo_greedy_p30:5, ppo_noisy_actions:7, ppo_then_noop_100:5, ppo_then_noop_50:18, ppo_then_side_tradeoff:5, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:2, ppo_waste_main_20:1, ppo_waste_side_15:3, ppo_waste_side_20:2, random:4                                                                                        crashed:50, landed:14            
E              ppo_deterministic:15, ppo_greedy_p10:12, ppo_greedy_p20:14, ppo_greedy_p30:11, ppo_noisy_actions:16, ppo_stochastic:10, ppo_then_noop_100:9, ppo_then_noop_150:15, ppo_then_noop_50:19, ppo_then_side_tradeoff:7, ppo_waste_main_12:18, ppo_waste_main_15:9, ppo_waste_main_18:5, ppo_waste_main_20:3, ppo_waste_side_15:9, ppo_waste_side_20:3, random:6  crashed:87, landed:93, timeout:1 
H              ppo_greedy_p20:1, ppo_noisy_actions:1, ppo_then_noop_100:2, ppo_then_noop_50:16, ppo_then_side_tradeoff:3, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:1, ppo_waste_main_20:1, ppo_waste_side_15:2, ppo_waste_side_20:2, random:1                                                                                                          crashed:23, landed:12            
I:C1-fixed     ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:4, ppo_waste_side_20:4                                                                                                                                                                         landed:24                        
I:C1-per-step  ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:3, ppo_waste_side_20:3                                                                                                                                                                         landed:21                        
I:C2-fixed     ppo_greedy_p10:1, ppo_greedy_p20:4, ppo_greedy_p30:1, ppo_noisy_actions:1, ppo_waste_main_12:2, ppo_waste_main_15:8, ppo_waste_main_18:3, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:5                                                                                                                                                    landed:27, timeout:7             
I:C2-per-step  ppo_greedy_p10:1, ppo_greedy_p20:4, ppo_greedy_p30:1, ppo_noisy_actions:1, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:4                                                                                                                                                    landed:26, timeout:2             
I:G-fixed      ppo_greedy_p10:1, ppo_greedy_p20:4, ppo_greedy_p30:1, ppo_noisy_actions:1, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:6, ppo_waste_side_20:4                                                                                                                                                    landed:26, timeout:2             
I:G-per-step   ppo_greedy_p10:1, ppo_greedy_p20:3, ppo_greedy_p30:1, ppo_waste_main_12:2, ppo_waste_main_15:4, ppo_waste_main_18:2, ppo_waste_main_20:3, ppo_waste_side_15:3, ppo_waste_side_20:4                                                                                                                                                                         landed:23                        
I:D            ppo_greedy_p20:1, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:2, ppo_waste_main_20:1, ppo_waste_side_15:3, ppo_waste_side_20:2                                                                                                                                                                                                             landed:14                        
I:H            ppo_greedy_p20:1, ppo_waste_main_12:2, ppo_waste_main_15:3, ppo_waste_main_18:1, ppo_waste_main_20:1, ppo_waste_side_15:2, ppo_waste_side_20:2                                                                                                                                                                                                             landed:12
```

### A.4 E2 -- e2_scale_stability.py

```text

=====================
E2 -- scale stability
=====================
fixed scales (seeds 1-20, extended set, pooled): s_r=50.00 s_n=250.12 s_i=[404.5, 95.8]

=======================================================================
Per-seed scale estimates on the calibration seeds (task's extended set)
=======================================================================
seed  s_r     s_n      s_Safety  s_Fuel 
----  ------  -------  --------  -------
   1  50.000  228.952   374.655   83.250
   2  50.000  358.417   617.709   99.125
   3  50.000  217.943   354.887   81.000
   4  50.000  250.529   418.182   82.875
   5  50.000  444.943   778.761  111.125
   6  50.000  266.655   436.685   96.625
   7  50.000  160.907   247.939   73.875
   8  37.500  159.193   194.762  123.625
   9  50.000  213.284   334.942   91.625
  10  50.000  351.903   602.306  101.500
  11  50.000  168.734   269.218   68.250
  12  75.000  187.739   298.354   77.125
  13  50.000  186.244   290.739   81.750
  14  50.000  224.771   367.292   82.250
  15  50.000  305.808   517.492   94.125
  16  50.000  220.520   315.290  125.750
  17  50.000  228.060   365.370   90.750
  18  37.500  291.708   408.667  174.750
  19  50.000  230.453   377.030   83.875
  20  50.000  305.587   518.924   92.250

scale     mean     std      min      max      CV     CV <= 0.25
--------  -------  -------  -------  -------  -----  ----------
s_r        50.000    6.847   37.500   75.000  0.137  yes       
s_n       250.118   72.208  159.193  444.943  0.289  no        
s_r+s_n   300.118   71.879  196.693  494.943  0.240  yes       
s_Safety  404.460  138.309  194.762  778.761  0.342  no        
s_Fuel     95.775   23.342   68.250  174.750  0.244  yes       

=========================================================
Per-step scales on the evaluation seeds, by candidate set
=========================================================
candidate set  scale    mean     min      max      CV   
-------------  -------  -------  -------  -------  -----
task           s_n      194.128  126.536  277.760  0.235
task           s_r+s_n  241.003  172.035  327.760  0.194
task+bl        s_n      155.343  120.342  199.101  0.134
task+bl        s_r+s_n  206.057  168.434  247.279  0.115
all            s_n      183.868  138.256  230.955  0.126
all            s_r+s_n  260.618  173.256  296.805  0.118
(the same seeds give different per-step scales depending on which candidates are certified together)

===============================================================================
Leave-one-seed-out: fixed scales and eval admissions (all candidates, eps grid)
===============================================================================
s_n over 20 leave-one-out fits: min=239.86 max=254.90 (full fit 250.12)
s_r+s_n over 20 leave-one-out fits: min=289.86 max=305.56 (full fit 300.12)
option      max verdicts changed by dropping one cal seed  share
----------  ---------------------------------------------  -----
A-fixed     17/2400                                         0.7%
B-fixed     10/2400                                         0.4%
C1-fixed    14/2400                                         0.6%
C2-fixed    22/2400                                         0.9%
G-fixed     11/2400                                         0.5%
I:C1-fixed  1/2400                                          0.0%
I:C2-fixed  7/2400                                          0.3%
I:G-fixed   4/2400                                          0.2%

==================================================================================================
Fixed vs per-step admissions, and outcome-class agreement at eps=0.1 (all candidates, seeds 21-40)
==================================================================================================
option         outcome-class agree  share  >= 90%  fixed==per-step (grid)
-------------  -------------------  -----  ------  ----------------------
F              381/400              95.2%  yes     -                     
A-fixed        395/400              98.8%  yes                      87.5%
A-per-step     383/400              95.8%  yes     -                     
B-fixed        393/400              98.2%  yes                      94.5%
B-per-step     387/400              96.8%  yes     -                     
C1-fixed       378/400              94.5%  yes                      96.0%
C1-per-step    379/400              94.8%  yes     -                     
C2-fixed       375/400              93.8%  yes                      97.5%
C2-per-step    377/400              94.2%  yes     -                     
G-fixed        378/400              94.5%  yes                      96.2%
G-per-step     379/400              94.8%  yes     -                     
D              380/400              95.0%  yes     -                     
E              384/400              96.0%  yes     -                     
H              380/400              95.0%  yes     -                     
I:C1-fixed     383/400              95.8%  yes                      99.5%
I:C1-per-step  382/400              95.5%  yes     -                     
I:C2-fixed     384/400              96.0%  yes                      99.5%
I:C2-per-step  383/400              95.8%  yes     -                     
I:G-fixed      383/400              95.8%  yes                      99.5%
I:G-per-step   382/400              95.5%  yes     -                     
I:D            383/400              95.8%  yes     -                     
I:H            383/400              95.8%  yes     -
```

### A.5 E3 -- e3_selection_impact.py

```text

=================================================
E3 -- selection impact vs raw units (seeds 21-40)
=================================================
weights w = (w_Safety, 1 - w_Safety), w_Safety in (0.2, 0.5, 0.8); eps in (0.1, 0.5)

===================
Candidate set: task
===================
option         eps    selection changed  share  admission-driven  score-driven  rewrites records
-------------  -----  -----------------  -----  ----------------  ------------  ----------------
F              0.100  0/60                0.0%                 0             0  no              
F              0.500  0/60                0.0%                 0             0  no              
A-fixed        0.100  0/60                0.0%                 0             0  yes             
A-fixed        0.500  0/60                0.0%                 0             0  yes             
A-per-step     0.100  0/60                0.0%                 0             0  yes             
A-per-step     0.500  0/60                0.0%                 0             0  yes             
B-fixed        0.100  4/60                6.7%                 0             4  yes             
B-fixed        0.500  4/60                6.7%                 0             4  yes             
B-per-step     0.100  4/60                6.7%                 0             4  yes             
B-per-step     0.500  4/60                6.7%                 0             4  yes             
C1-fixed       0.100  0/60                0.0%                 0             0  no              
C1-fixed       0.500  0/60                0.0%                 0             0  no              
C1-per-step    0.100  0/60                0.0%                 0             0  no              
C1-per-step    0.500  0/60                0.0%                 0             0  no              
C2-fixed       0.100  0/60                0.0%                 0             0  no              
C2-fixed       0.500  0/60                0.0%                 0             0  no              
C2-per-step    0.100  0/60                0.0%                 0             0  no              
C2-per-step    0.500  0/60                0.0%                 0             0  no              
G-fixed        0.100  0/60                0.0%                 0             0  no              
G-fixed        0.500  0/60                0.0%                 0             0  no              
G-per-step     0.100  0/60                0.0%                 0             0  no              
G-per-step     0.500  0/60                0.0%                 0             0  no              
D              0.100  0/60                0.0%                 0             0  no              
D              0.500  0/60                0.0%                 0             0  no              
E              0.100  8/60               13.3%                 8             0  no              
E              0.500  0/60                0.0%                 0             0  no              
H              0.100  0/60                0.0%                 0             0  no              
H              0.500  0/60                0.0%                 0             0  no              
I:C1-fixed     0.100  0/60                0.0%                 0             0  no              
I:C1-fixed     0.500  0/60                0.0%                 0             0  no              
I:C1-per-step  0.100  0/60                0.0%                 0             0  no              
I:C1-per-step  0.500  0/60                0.0%                 0             0  no              
I:C2-fixed     0.100  0/60                0.0%                 0             0  no              
I:C2-fixed     0.500  0/60                0.0%                 0             0  no              
I:C2-per-step  0.100  0/60                0.0%                 0             0  no              
I:C2-per-step  0.500  0/60                0.0%                 0             0  no              
I:G-fixed      0.100  0/60                0.0%                 0             0  no              
I:G-fixed      0.500  0/60                0.0%                 0             0  no              
I:G-per-step   0.100  0/60                0.0%                 0             0  no              
I:G-per-step   0.500  0/60                0.0%                 0             0  no              
I:D            0.100  0/60                0.0%                 0             0  no              
I:D            0.500  0/60                0.0%                 0             0  no              
I:H            0.100  0/60                0.0%                 0             0  no              
I:H            0.500  0/60                0.0%                 0             0  no              

==================
Candidate set: all
==================
option         eps    selection changed  share  admission-driven  score-driven  rewrites records
-------------  -----  -----------------  -----  ----------------  ------------  ----------------
F              0.100  0/60                0.0%                 0             0  no              
F              0.500  0/60                0.0%                 0             0  no              
A-fixed        0.100  0/60                0.0%                 0             0  yes             
A-fixed        0.500  0/60                0.0%                 0             0  yes             
A-per-step     0.100  0/60                0.0%                 0             0  yes             
A-per-step     0.500  0/60                0.0%                 0             0  yes             
B-fixed        0.100  10/60              16.7%                 0            10  yes             
B-fixed        0.500  10/60              16.7%                 0            10  yes             
B-per-step     0.100  2/60                3.3%                 0             2  yes             
B-per-step     0.500  2/60                3.3%                 0             2  yes             
C1-fixed       0.100  0/60                0.0%                 0             0  no              
C1-fixed       0.500  0/60                0.0%                 0             0  no              
C1-per-step    0.100  0/60                0.0%                 0             0  no              
C1-per-step    0.500  0/60                0.0%                 0             0  no              
C2-fixed       0.100  0/60                0.0%                 0             0  no              
C2-fixed       0.500  0/60                0.0%                 0             0  no              
C2-per-step    0.100  0/60                0.0%                 0             0  no              
C2-per-step    0.500  0/60                0.0%                 0             0  no              
G-fixed        0.100  0/60                0.0%                 0             0  no              
G-fixed        0.500  0/60                0.0%                 0             0  no              
G-per-step     0.100  0/60                0.0%                 0             0  no              
G-per-step     0.500  0/60                0.0%                 0             0  no              
D              0.100  0/60                0.0%                 0             0  no              
D              0.500  0/60                0.0%                 0             0  no              
E              0.100  8/60               13.3%                 8             0  no              
E              0.500  0/60                0.0%                 0             0  no              
H              0.100  0/60                0.0%                 0             0  no              
H              0.500  0/60                0.0%                 0             0  no              
I:C1-fixed     0.100  0/60                0.0%                 0             0  no              
I:C1-fixed     0.500  0/60                0.0%                 0             0  no              
I:C1-per-step  0.100  0/60                0.0%                 0             0  no              
I:C1-per-step  0.500  0/60                0.0%                 0             0  no              
I:C2-fixed     0.100  0/60                0.0%                 0             0  no              
I:C2-fixed     0.500  0/60                0.0%                 0             0  no              
I:C2-per-step  0.100  0/60                0.0%                 0             0  no              
I:C2-per-step  0.500  0/60                0.0%                 0             0  no              
I:G-fixed      0.100  0/60                0.0%                 0             0  no              
I:G-fixed      0.500  0/60                0.0%                 0             0  no              
I:G-per-step   0.100  0/60                0.0%                 0             0  no              
I:G-per-step   0.500  0/60                0.0%                 0             0  no              
I:D            0.100  0/60                0.0%                 0             0  no              
I:D            0.500  0/60                0.0%                 0             0  no              
I:H            0.100  0/60                0.0%                 0             0  no              
I:H            0.500  0/60                0.0%                 0             0  no
```

### A.6 E4 -- e4_end_to_end.py

```text

======================================================================================
E4 run 1a -- scripted epsilon, same start every step (seed 8), fresh pipeline per step
======================================================================================

===============================
Run 1a -- fix OFF (raw epsilon)
===============================

step 0  seed 8  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.500      26.500  yes     
ppo_stochastic     timeout  -341.000              0.500    -340.500  no      
left_engine        crashed  -331.878              0.500    -331.378  no      
main_engine        crashed  -472.251              0.500    -471.751  no      
right_engine       crashed  -251.312              0.500    -250.812  no      
random             crashed   -90.358              0.500     -89.858  no      
ppo_waste_side_15  landed    -43.000              0.500     -42.500  no      
ppo_waste_main_15  timeout  -373.000              0.500    -372.500  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 1  seed 8  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.300      26.300  yes     
ppo_stochastic     timeout  -341.000              0.300    -340.700  no      
left_engine        crashed  -331.878              0.300    -331.578  no      
main_engine        crashed  -472.251              0.300    -471.951  no      
right_engine       crashed  -251.312              0.300    -251.012  no      
random             crashed   -90.358              0.300     -90.058  no      
ppo_waste_side_15  landed    -43.000              0.300     -42.700  no      
ppo_waste_main_15  timeout  -373.000              0.300    -372.700  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 2  seed 8  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.100      26.100  yes     
ppo_stochastic     timeout  -341.000              0.100    -340.900  no      
left_engine        crashed  -331.878              0.100    -331.778  no      
main_engine        crashed  -472.251              0.100    -472.151  no      
right_engine       crashed  -251.312              0.100    -251.212  no      
random             crashed   -90.358              0.100     -90.258  no      
ppo_waste_side_15  landed    -43.000              0.100     -42.900  no      
ppo_waste_main_15  timeout  -373.000              0.100    -372.900  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 8  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.000      26.000  yes     
ppo_stochastic     timeout  -341.000              0.000    -341.000  no      
left_engine        crashed  -331.878              0.000    -331.878  no      
main_engine        crashed  -472.251              0.000    -472.251  no      
right_engine       crashed  -251.312              0.000    -251.312  no      
random             crashed   -90.358              0.000     -90.358  no      
ppo_waste_side_15  landed    -43.000              0.000     -43.000  no      
ppo_waste_main_15  timeout  -373.000              0.000    -373.000  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted           selected           flown 
----  ----  -----  ----  -----------------  -----------------  ------
   0     8  0.500     1  ppo_deterministic  ppo_deterministic  landed
   1     8  0.300     1  ppo_deterministic  ppo_deterministic  landed
   2     8  0.100     1  ppo_deterministic  ppo_deterministic  landed
   3     8  0.000     1  ppo_deterministic  ppo_deterministic  landed

======================================
Run 1a -- fix ON (guarded scale 300.0)
======================================

step 0  seed 8  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000            150.000     176.000  yes     
ppo_stochastic     timeout  -341.000            150.000    -191.000  no      
left_engine        crashed  -331.878              0.500    -331.378  no      
main_engine        crashed  -472.251              0.500    -471.751  no      
right_engine       crashed  -251.312              0.500    -250.812  no      
random             crashed   -90.358              0.500     -89.858  no      
ppo_waste_side_15  landed    -43.000            150.000     107.000  yes     
ppo_waste_main_15  timeout  -373.000            150.000    -223.000  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 1  seed 8  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000             90.000     116.000  yes     
ppo_stochastic     timeout  -341.000             90.000    -251.000  no      
left_engine        crashed  -331.878              0.300    -331.578  no      
main_engine        crashed  -472.251              0.300    -471.951  no      
right_engine       crashed  -251.312              0.300    -251.012  no      
random             crashed   -90.358              0.300     -90.058  no      
ppo_waste_side_15  landed    -43.000             90.000      47.000  yes     
ppo_waste_main_15  timeout  -373.000             90.000    -283.000  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 2  seed 8  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000             30.000      56.000  yes     
ppo_stochastic     timeout  -341.000             30.000    -311.000  no      
left_engine        crashed  -331.878              0.100    -331.778  no      
main_engine        crashed  -472.251              0.100    -472.151  no      
right_engine       crashed  -251.312              0.100    -251.212  no      
random             crashed   -90.358              0.100     -90.258  no      
ppo_waste_side_15  landed    -43.000             30.000     -13.000  no      
ppo_waste_main_15  timeout  -373.000             30.000    -343.000  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 8  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.000      26.000  yes     
ppo_stochastic     timeout  -341.000              0.000    -341.000  no      
left_engine        crashed  -331.878              0.000    -331.878  no      
main_engine        crashed  -472.251              0.000    -472.251  no      
right_engine       crashed  -251.312              0.000    -251.312  no      
random             crashed   -90.358              0.000     -90.358  no      
ppo_waste_side_15  landed    -43.000              0.000     -43.000  no      
ppo_waste_main_15  timeout  -373.000              0.000    -373.000  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                              selected           flown 
----  ----  -----  ----  ------------------------------------  -----------------  ------
   0     8  0.500     2  ppo_deterministic, ppo_waste_side_15  ppo_deterministic  landed
   1     8  0.300     2  ppo_deterministic, ppo_waste_side_15  ppo_deterministic  landed
   2     8  0.100     1  ppo_deterministic                     ppo_deterministic  landed
   3     8  0.000     1  ppo_deterministic                     ppo_deterministic  landed

=============================================================================================
E4 run 1b -- scripted epsilon cycling 0.5/0.3/0.1/0.0, a new held-out seed every step (21-40)
=============================================================================================

===============================
Run 1b -- fix OFF (raw epsilon)
===============================

step 0  seed 21  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     72.000              0.500      72.500  yes     
ppo_stochastic     landed     80.000              0.500      80.500  yes     
left_engine        crashed  -244.787              0.500    -244.287  no      
main_engine        crashed  -516.234              0.500    -515.734  no      
right_engine       crashed  -325.045              0.500    -324.545  no      
random             crashed  -214.676              0.500    -214.176  no      
ppo_waste_side_15  timeout  -271.000              0.500    -270.500  no      
ppo_waste_main_15  timeout  -177.000              0.500    -176.500  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 1  seed 22  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     102.000              0.300     102.300  yes     
ppo_stochastic     landed     105.000              0.300     105.300  yes     
left_engine        crashed   -403.870              0.300    -403.570  no      
main_engine        crashed  -1.01e+03              0.300   -1.01e+03  no      
right_engine       crashed   -410.131              0.300    -409.831  no      
random             crashed   -102.717              0.300    -102.417  no      
ppo_waste_side_15  landed      77.000              0.300      77.300  yes     
ppo_waste_main_15  landed      13.000              0.300      13.300  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 2  seed 23  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     80.000              0.100      80.100  yes     
ppo_stochastic     landed     76.000              0.100      76.100  yes     
left_engine        crashed  -423.800              0.100    -423.700  no      
main_engine        crashed  -530.773              0.100    -530.673  no      
right_engine       crashed  -466.278              0.100    -466.178  no      
random             crashed  -239.518              0.100    -239.418  no      
ppo_waste_side_15  landed     18.000              0.100      18.100  yes     
ppo_waste_main_15  landed      1.000              0.100       1.100  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 24  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    105.000              0.000     105.000  yes     
ppo_stochastic     landed     89.000              0.000      89.000  yes     
left_engine        crashed  -170.923              0.000    -170.923  no      
main_engine        crashed  -433.778              0.000    -433.778  no      
right_engine       crashed  -228.247              0.000    -228.247  no      
random             crashed   -55.000              0.000     -55.000  no      
ppo_waste_side_15  timeout  -236.000              0.000    -236.000  no      
ppo_waste_main_15  timeout  -149.000              0.000    -149.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 4  seed 25  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  crashed  -166.000              0.500    -165.500  no      
ppo_stochastic     landed    105.000              0.500     105.500  yes     
left_engine        crashed  -739.983              0.500    -739.483  no      
main_engine        crashed  -841.814              0.500    -841.314  no      
right_engine       crashed  -689.444              0.500    -688.944  no      
random             crashed  -318.160              0.500    -317.660  no      
ppo_waste_side_15  crashed  -202.000              0.500    -201.500  no      
ppo_waste_main_15  crashed  -198.000              0.500    -197.500  no      
admitted: ['ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 5  seed 26  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     71.000              0.300      71.300  yes     
ppo_stochastic     landed     75.000              0.300      75.300  yes     
left_engine        crashed  -220.300              0.300    -220.000  no      
main_engine        crashed  -379.073              0.300    -378.773  no      
right_engine       crashed  -490.306              0.300    -490.006  no      
random             crashed  -256.023              0.300    -255.723  no      
ppo_waste_side_15  landed   -132.000              0.300    -131.700  no      
ppo_waste_main_15  timeout  -176.000              0.300    -175.700  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 6  seed 27  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     82.000              0.100      82.100  yes     
ppo_stochastic     landed     78.000              0.100      78.100  yes     
left_engine        crashed  -646.140              0.100    -646.040  no      
main_engine        crashed  -613.966              0.100    -613.866  no      
right_engine       crashed  -690.169              0.100    -690.069  no      
random             crashed  -129.557              0.100    -129.457  no      
ppo_waste_side_15  landed    -53.000              0.100     -52.900  no      
ppo_waste_main_15  landed     18.000              0.100      18.100  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 7  seed 28  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     91.000              0.000      91.000  yes     
ppo_stochastic     landed     97.000              0.000      97.000  yes     
left_engine        crashed  -126.515              0.000    -126.515  no      
main_engine        crashed  -421.653              0.000    -421.653  no      
right_engine       crashed  -286.982              0.000    -286.982  no      
random             crashed  -116.506              0.000    -116.506  no      
ppo_waste_side_15  landed     19.000              0.000      19.000  yes     
ppo_waste_main_15  landed     -3.000              0.000      -3.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 8  seed 29  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     76.000              0.500      76.500  yes     
ppo_stochastic     landed     74.000              0.500      74.500  yes     
left_engine        crashed  -659.146              0.500    -658.646  no      
main_engine        crashed  -497.241              0.500    -496.741  no      
right_engine       crashed  -783.905              0.500    -783.405  no      
random             crashed  -254.622              0.500    -254.122  no      
ppo_waste_side_15  timeout  -253.000              0.500    -252.500  no      
ppo_waste_main_15  timeout  -276.000              0.500    -275.500  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 9  seed 30  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  timeout  -404.000              0.300    -403.700  no      
ppo_stochastic     landed     83.000              0.300      83.300  yes     
left_engine        crashed  -341.591              0.300    -341.291  no      
main_engine        crashed  -503.699              0.300    -503.399  no      
right_engine       crashed  -393.450              0.300    -393.150  no      
random             crashed   -63.460              0.300     -63.160  no      
ppo_waste_side_15  landed   -318.000              0.300    -317.700  no      
ppo_waste_main_15  landed      4.000              0.300       4.300  yes     
admitted: ['ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_stochastic   flown outcome: landed

step 10  seed 31  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    106.000              0.100     106.100  yes     
ppo_stochastic     landed    127.000              0.100     127.100  yes     
left_engine        crashed  -192.195              0.100    -192.095  no      
main_engine        crashed  -560.530              0.100    -560.430  no      
right_engine       crashed  -296.976              0.100    -296.876  no      
random             crashed   -57.000              0.100     -56.900  no      
ppo_waste_side_15  landed      8.000              0.100       8.100  yes     
ppo_waste_main_15  timeout  -147.000              0.100    -146.900  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 11  seed 32  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     101.000              0.000     101.000  yes     
ppo_stochastic     landed      99.000              0.000      99.000  yes     
left_engine        crashed   -707.132              0.000    -707.132  no      
main_engine        crashed  -1.15e+03              0.000   -1.15e+03  no      
right_engine       crashed   -697.937              0.000    -697.937  no      
random             crashed   -117.000              0.000    -117.000  no      
ppo_waste_side_15  landed     -38.000              0.000     -38.000  no      
ppo_waste_main_15  landed      32.000              0.000      32.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 12  seed 33  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     88.000              0.500      88.500  yes     
ppo_stochastic     timeout  -349.000              0.500    -348.500  no      
left_engine        crashed  -335.581              0.500    -335.081  no      
main_engine        crashed  -499.785              0.500    -499.285  no      
right_engine       crashed  -359.578              0.500    -359.078  no      
random             crashed   -55.000              0.500     -54.500  no      
ppo_waste_side_15  crashed  -180.000              0.500    -179.500  no      
ppo_waste_main_15  landed    -25.000              0.500     -24.500  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 13  seed 34  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    106.000              0.300     106.300  yes     
ppo_stochastic     landed    109.000              0.300     109.300  yes     
left_engine        crashed  -297.818              0.300    -297.518  no      
main_engine        crashed  -573.656              0.300    -573.356  no      
right_engine       crashed  -397.009              0.300    -396.709  no      
random             crashed  -346.914              0.300    -346.614  no      
ppo_waste_side_15  landed    -22.000              0.300     -21.700  no      
ppo_waste_main_15  landed    -88.000              0.300     -87.700  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 14  seed 35  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    101.000              0.100     101.100  yes     
ppo_stochastic     landed     90.000              0.100      90.100  yes     
left_engine        crashed  -356.979              0.100    -356.879  no      
main_engine        crashed  -471.320              0.100    -471.220  no      
right_engine       crashed  -312.240              0.100    -312.140  no      
random             crashed  -304.506              0.100    -304.406  no      
ppo_waste_side_15  landed   -111.000              0.100    -110.900  no      
ppo_waste_main_15  timeout  -140.000              0.100    -139.900  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 15  seed 36  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    101.000              0.000     101.000  yes     
ppo_stochastic     landed     98.000              0.000      98.000  yes     
left_engine        crashed  -625.110              0.000    -625.110  no      
main_engine        crashed  -897.423              0.000    -897.423  no      
right_engine       crashed  -898.268              0.000    -898.268  no      
random             crashed  -292.828              0.000    -292.828  no      
ppo_waste_side_15  timeout  -233.000              0.000    -233.000  no      
ppo_waste_main_15  timeout  -148.000              0.000    -148.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 16  seed 37  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     51.000              0.500      51.500  yes     
ppo_stochastic     crashed  -164.000              0.500    -163.500  no      
left_engine        crashed  -372.191              0.500    -371.691  no      
main_engine        crashed  -500.430              0.500    -499.930  no      
right_engine       crashed  -130.431              0.500    -129.931  no      
random             crashed   -69.000              0.500     -68.500  no      
ppo_waste_side_15  crashed  -169.000              0.500    -168.500  no      
ppo_waste_main_15  crashed  -160.000              0.500    -159.500  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 17  seed 38  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    103.000              0.300     103.300  yes     
ppo_stochastic     landed     95.000              0.300      95.300  yes     
left_engine        crashed  -497.158              0.300    -496.858  no      
main_engine        crashed  -644.461              0.300    -644.161  no      
right_engine       crashed  -684.321              0.300    -684.021  no      
random             crashed  -318.730              0.300    -318.430  no      
ppo_waste_side_15  timeout  -255.000              0.300    -254.700  no      
ppo_waste_main_15  timeout  -160.000              0.300    -159.700  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 18  seed 39  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    103.000              0.100     103.100  yes     
ppo_stochastic     landed    107.000              0.100     107.100  yes     
left_engine        crashed  -383.378              0.100    -383.278  no      
main_engine        crashed  -422.064              0.100    -421.964  no      
right_engine       crashed  -160.570              0.100    -160.470  no      
random             crashed   -55.000              0.100     -54.900  no      
ppo_waste_side_15  landed   -130.000              0.100    -129.900  no      
ppo_waste_main_15  landed    -20.000              0.100     -19.900  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 19  seed 40  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    110.000              0.000     110.000  yes     
ppo_stochastic     landed    111.000              0.000     111.000  yes     
left_engine        crashed  -292.904              0.000    -292.904  no      
main_engine        crashed  -585.891              0.000    -585.891  no      
right_engine       crashed  -373.549              0.000    -373.549  no      
random             crashed   -49.000              0.000     -49.000  no      
ppo_waste_side_15  landed     29.000              0.000      29.000  yes     
ppo_waste_main_15  landed     69.000              0.000      69.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                                                                 selected           flown 
----  ----  -----  ----  -----------------------------------------------------------------------  -----------------  ------
   0    21  0.500     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   1    22  0.300     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed
   2    23  0.100     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_deterministic  landed
   3    24  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   4    25  0.500     1  ppo_stochastic                                                           ppo_stochastic     landed
   5    26  0.300     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   6    27  0.100     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
   7    28  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_deterministic  landed
   8    29  0.500     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   9    30  0.300     2  ppo_stochastic, ppo_waste_main_15                                        ppo_stochastic     landed
  10    31  0.100     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_stochastic     landed
  11    32  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
  12    33  0.500     1  ppo_deterministic                                                        ppo_deterministic  landed
  13    34  0.300     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
  14    35  0.100     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
  15    36  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
  16    37  0.500     1  ppo_deterministic                                                        ppo_deterministic  landed
  17    38  0.300     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
  18    39  0.100     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
  19    40  0.000     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed

======================================
Run 1b -- fix ON (guarded scale 300.0)
======================================

step 0  seed 21  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     72.000            150.000     222.000  yes     
ppo_stochastic     landed     80.000            150.000     230.000  yes     
left_engine        crashed  -244.787              0.500    -244.287  no      
main_engine        crashed  -516.234              0.500    -515.734  no      
right_engine       crashed  -325.045              0.500    -324.545  no      
random             crashed  -214.676              0.500    -214.176  no      
ppo_waste_side_15  timeout  -271.000            150.000    -121.000  no      
ppo_waste_main_15  timeout  -177.000            150.000     -27.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 1  seed 22  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     102.000             90.000     192.000  yes     
ppo_stochastic     landed     105.000             90.000     195.000  yes     
left_engine        crashed   -403.870              0.300    -403.570  no      
main_engine        crashed  -1.01e+03              0.300   -1.01e+03  no      
right_engine       crashed   -410.131              0.300    -409.831  no      
random             crashed   -102.717              0.300    -102.417  no      
ppo_waste_side_15  landed      77.000             90.000     167.000  yes     
ppo_waste_main_15  landed      13.000             90.000     103.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 2  seed 23  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     80.000             30.000     110.000  yes     
ppo_stochastic     landed     76.000             30.000     106.000  yes     
left_engine        crashed  -423.800              0.100    -423.700  no      
main_engine        crashed  -530.773              0.100    -530.673  no      
right_engine       crashed  -466.278              0.100    -466.178  no      
random             crashed  -239.518              0.100    -239.418  no      
ppo_waste_side_15  landed     18.000             30.000      48.000  yes     
ppo_waste_main_15  landed      1.000             30.000      31.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 24  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    105.000              0.000     105.000  yes     
ppo_stochastic     landed     89.000              0.000      89.000  yes     
left_engine        crashed  -170.923              0.000    -170.923  no      
main_engine        crashed  -433.778              0.000    -433.778  no      
right_engine       crashed  -228.247              0.000    -228.247  no      
random             crashed   -55.000              0.000     -55.000  no      
ppo_waste_side_15  timeout  -236.000              0.000    -236.000  no      
ppo_waste_main_15  timeout  -149.000              0.000    -149.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 4  seed 25  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  crashed  -166.000              0.500    -165.500  no      
ppo_stochastic     landed    105.000            150.000     255.000  yes     
left_engine        crashed  -739.983              0.500    -739.483  no      
main_engine        crashed  -841.814              0.500    -841.314  no      
right_engine       crashed  -689.444              0.500    -688.944  no      
random             crashed  -318.160              0.500    -317.660  no      
ppo_waste_side_15  crashed  -202.000              0.500    -201.500  no      
ppo_waste_main_15  crashed  -198.000              0.500    -197.500  no      
admitted: ['ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 5  seed 26  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     71.000             90.000     161.000  yes     
ppo_stochastic     landed     75.000             90.000     165.000  yes     
left_engine        crashed  -220.300              0.300    -220.000  no      
main_engine        crashed  -379.073              0.300    -378.773  no      
right_engine       crashed  -490.306              0.300    -490.006  no      
random             crashed  -256.023              0.300    -255.723  no      
ppo_waste_side_15  landed   -132.000             90.000     -42.000  no      
ppo_waste_main_15  timeout  -176.000             90.000     -86.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 6  seed 27  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     82.000             30.000     112.000  yes     
ppo_stochastic     landed     78.000             30.000     108.000  yes     
left_engine        crashed  -646.140              0.100    -646.040  no      
main_engine        crashed  -613.966              0.100    -613.866  no      
right_engine       crashed  -690.169              0.100    -690.069  no      
random             crashed  -129.557              0.100    -129.457  no      
ppo_waste_side_15  landed    -53.000             30.000     -23.000  no      
ppo_waste_main_15  landed     18.000             30.000      48.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 7  seed 28  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     91.000              0.000      91.000  yes     
ppo_stochastic     landed     97.000              0.000      97.000  yes     
left_engine        crashed  -126.515              0.000    -126.515  no      
main_engine        crashed  -421.653              0.000    -421.653  no      
right_engine       crashed  -286.982              0.000    -286.982  no      
random             crashed  -116.506              0.000    -116.506  no      
ppo_waste_side_15  landed     19.000              0.000      19.000  yes     
ppo_waste_main_15  landed     -3.000              0.000      -3.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 8  seed 29  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     76.000            150.000     226.000  yes     
ppo_stochastic     landed     74.000            150.000     224.000  yes     
left_engine        crashed  -659.146              0.500    -658.646  no      
main_engine        crashed  -497.241              0.500    -496.741  no      
right_engine       crashed  -783.905              0.500    -783.405  no      
random             crashed  -254.622              0.500    -254.122  no      
ppo_waste_side_15  timeout  -253.000            150.000    -103.000  no      
ppo_waste_main_15  timeout  -276.000            150.000    -126.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 9  seed 30  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  timeout  -404.000             90.000    -314.000  no      
ppo_stochastic     landed     83.000             90.000     173.000  yes     
left_engine        crashed  -341.591              0.300    -341.291  no      
main_engine        crashed  -503.699              0.300    -503.399  no      
right_engine       crashed  -393.450              0.300    -393.150  no      
random             crashed   -63.460              0.300     -63.160  no      
ppo_waste_side_15  landed   -318.000             90.000    -228.000  no      
ppo_waste_main_15  landed      4.000             90.000      94.000  yes     
admitted: ['ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_stochastic   flown outcome: landed

step 10  seed 31  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    106.000             30.000     136.000  yes     
ppo_stochastic     landed    127.000             30.000     157.000  yes     
left_engine        crashed  -192.195              0.100    -192.095  no      
main_engine        crashed  -560.530              0.100    -560.430  no      
right_engine       crashed  -296.976              0.100    -296.876  no      
random             crashed   -57.000              0.100     -56.900  no      
ppo_waste_side_15  landed      8.000             30.000      38.000  yes     
ppo_waste_main_15  timeout  -147.000             30.000    -117.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 11  seed 32  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     101.000              0.000     101.000  yes     
ppo_stochastic     landed      99.000              0.000      99.000  yes     
left_engine        crashed   -707.132              0.000    -707.132  no      
main_engine        crashed  -1.15e+03              0.000   -1.15e+03  no      
right_engine       crashed   -697.937              0.000    -697.937  no      
random             crashed   -117.000              0.000    -117.000  no      
ppo_waste_side_15  landed     -38.000              0.000     -38.000  no      
ppo_waste_main_15  landed      32.000              0.000      32.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 12  seed 33  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     88.000            150.000     238.000  yes     
ppo_stochastic     timeout  -349.000            150.000    -199.000  no      
left_engine        crashed  -335.581              0.500    -335.081  no      
main_engine        crashed  -499.785              0.500    -499.285  no      
right_engine       crashed  -359.578              0.500    -359.078  no      
random             crashed   -55.000              0.500     -54.500  no      
ppo_waste_side_15  crashed  -180.000              0.500    -179.500  no      
ppo_waste_main_15  landed    -25.000            150.000     125.000  yes     
admitted: ['ppo_deterministic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 13  seed 34  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    106.000             90.000     196.000  yes     
ppo_stochastic     landed    109.000             90.000     199.000  yes     
left_engine        crashed  -297.818              0.300    -297.518  no      
main_engine        crashed  -573.656              0.300    -573.356  no      
right_engine       crashed  -397.009              0.300    -396.709  no      
random             crashed  -346.914              0.300    -346.614  no      
ppo_waste_side_15  landed    -22.000             90.000      68.000  yes     
ppo_waste_main_15  landed    -88.000             90.000       2.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 14  seed 35  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    101.000             30.000     131.000  yes     
ppo_stochastic     landed     90.000             30.000     120.000  yes     
left_engine        crashed  -356.979              0.100    -356.879  no      
main_engine        crashed  -471.320              0.100    -471.220  no      
right_engine       crashed  -312.240              0.100    -312.140  no      
random             crashed  -304.506              0.100    -304.406  no      
ppo_waste_side_15  landed   -111.000             30.000     -81.000  no      
ppo_waste_main_15  timeout  -140.000             30.000    -110.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 15  seed 36  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    101.000              0.000     101.000  yes     
ppo_stochastic     landed     98.000              0.000      98.000  yes     
left_engine        crashed  -625.110              0.000    -625.110  no      
main_engine        crashed  -897.423              0.000    -897.423  no      
right_engine       crashed  -898.268              0.000    -898.268  no      
random             crashed  -292.828              0.000    -292.828  no      
ppo_waste_side_15  timeout  -233.000              0.000    -233.000  no      
ppo_waste_main_15  timeout  -148.000              0.000    -148.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 16  seed 37  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     51.000            150.000     201.000  yes     
ppo_stochastic     crashed  -164.000              0.500    -163.500  no      
left_engine        crashed  -372.191              0.500    -371.691  no      
main_engine        crashed  -500.430              0.500    -499.930  no      
right_engine       crashed  -130.431              0.500    -129.931  no      
random             crashed   -69.000              0.500     -68.500  no      
ppo_waste_side_15  crashed  -169.000              0.500    -168.500  no      
ppo_waste_main_15  crashed  -160.000              0.500    -159.500  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 17  seed 38  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    103.000             90.000     193.000  yes     
ppo_stochastic     landed     95.000             90.000     185.000  yes     
left_engine        crashed  -497.158              0.300    -496.858  no      
main_engine        crashed  -644.461              0.300    -644.161  no      
right_engine       crashed  -684.321              0.300    -684.021  no      
random             crashed  -318.730              0.300    -318.430  no      
ppo_waste_side_15  timeout  -255.000             90.000    -165.000  no      
ppo_waste_main_15  timeout  -160.000             90.000     -70.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 18  seed 39  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    103.000             30.000     133.000  yes     
ppo_stochastic     landed    107.000             30.000     137.000  yes     
left_engine        crashed  -383.378              0.100    -383.278  no      
main_engine        crashed  -422.064              0.100    -421.964  no      
right_engine       crashed  -160.570              0.100    -160.470  no      
random             crashed   -55.000              0.100     -54.900  no      
ppo_waste_side_15  landed   -130.000             30.000    -100.000  no      
ppo_waste_main_15  landed    -20.000             30.000      10.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_stochastic   flown outcome: landed

step 19  seed 40  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    110.000              0.000     110.000  yes     
ppo_stochastic     landed    111.000              0.000     111.000  yes     
left_engine        crashed  -292.904              0.000    -292.904  no      
main_engine        crashed  -585.891              0.000    -585.891  no      
right_engine       crashed  -373.549              0.000    -373.549  no      
random             crashed   -49.000              0.000     -49.000  no      
ppo_waste_side_15  landed     29.000              0.000      29.000  yes     
ppo_waste_main_15  landed     69.000              0.000      69.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                                                                 selected           flown 
----  ----  -----  ----  -----------------------------------------------------------------------  -----------------  ------
   0    21  0.500     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   1    22  0.300     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed
   2    23  0.100     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_deterministic  landed
   3    24  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   4    25  0.500     1  ppo_stochastic                                                           ppo_stochastic     landed
   5    26  0.300     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   6    27  0.100     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
   7    28  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_deterministic  landed
   8    29  0.500     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   9    30  0.300     2  ppo_stochastic, ppo_waste_main_15                                        ppo_stochastic     landed
  10    31  0.100     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_stochastic     landed
  11    32  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
  12    33  0.500     2  ppo_deterministic, ppo_waste_main_15                                     ppo_deterministic  landed
  13    34  0.300     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_deterministic  landed
  14    35  0.100     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
  15    36  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
  16    37  0.500     1  ppo_deterministic                                                        ppo_deterministic  landed
  17    38  0.300     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
  18    39  0.100     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_stochastic     landed
  19    40  0.000     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed

=====================================================================================
E4 run 1c -- as 1a but ONE pipeline for all steps (W_x grows on the repeated context)
=====================================================================================

===============================
Run 1c -- fix OFF (raw epsilon)
===============================

step 0  seed 8  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.500      26.500  yes     
ppo_stochastic     timeout  -341.000              0.500    -340.500  no      
left_engine        crashed  -331.878              0.500    -331.378  no      
main_engine        crashed  -472.251              0.500    -471.751  no      
right_engine       crashed  -251.312              0.500    -250.812  no      
random             crashed   -90.358              0.500     -89.858  no      
ppo_waste_side_15  landed    -43.000              0.500     -42.500  no      
ppo_waste_main_15  timeout  -373.000              0.500    -372.500  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

step 1  seed 8  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.300      26.300  yes     
ppo_stochastic     timeout  -341.000              0.300    -340.700  no      
left_engine        crashed  -331.878              0.300    -331.578  no      
main_engine        crashed  -472.251              0.300    -471.951  no      
right_engine       crashed  -251.312              0.300    -251.012  no      
random             crashed   -90.358              0.300     -90.058  no      
ppo_waste_side_15  landed    -43.000              0.300     -42.700  yes     
ppo_waste_main_15  timeout  -373.000              0.300    -372.700  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 2  seed 8  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.100      26.100  yes     
ppo_stochastic     timeout  -341.000              0.100    -340.900  no      
left_engine        crashed  -331.878              0.100    -331.778  no      
main_engine        crashed  -472.251              0.100    -472.151  no      
right_engine       crashed  -251.312              0.100    -251.212  no      
random             crashed   -90.358              0.100     -90.258  no      
ppo_waste_side_15  landed    -43.000              0.100     -42.900  yes     
ppo_waste_main_15  timeout  -373.000              0.100    -372.900  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 8  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.000      26.000  yes     
ppo_stochastic     timeout  -341.000              0.000    -341.000  no      
left_engine        crashed  -331.878              0.000    -331.878  no      
main_engine        crashed  -472.251              0.000    -472.251  no      
right_engine       crashed  -251.312              0.000    -251.312  no      
random             crashed   -90.358              0.000     -90.358  no      
ppo_waste_side_15  landed    -43.000              0.000     -43.000  yes     
ppo_waste_main_15  timeout  -373.000              0.000    -373.000  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                              selected           flown 
----  ----  -----  ----  ------------------------------------  -----------------  ------
   0     8  0.500     1  ppo_deterministic                     ppo_deterministic  landed
   1     8  0.300     2  ppo_deterministic, ppo_waste_side_15  ppo_deterministic  landed
   2     8  0.100     2  ppo_deterministic, ppo_waste_side_15  ppo_deterministic  landed
   3     8  0.000     2  ppo_deterministic, ppo_waste_side_15  ppo_deterministic  landed

======================================
Run 1c -- fix ON (guarded scale 300.0)
======================================

step 0  seed 8  eps=0.5000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000            150.000     176.000  yes     
ppo_stochastic     timeout  -341.000            150.000    -191.000  no      
left_engine        crashed  -331.878              0.500    -331.378  no      
main_engine        crashed  -472.251              0.500    -471.751  no      
right_engine       crashed  -251.312              0.500    -250.812  no      
random             crashed   -90.358              0.500     -89.858  no      
ppo_waste_side_15  landed    -43.000            150.000     107.000  yes     
ppo_waste_main_15  timeout  -373.000            150.000    -223.000  no      
admitted: ['ppo_deterministic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 1  seed 8  eps=0.3000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000             90.000     116.000  yes     
ppo_stochastic     timeout  -341.000             90.000    -251.000  no      
left_engine        crashed  -331.878              0.300    -331.578  no      
main_engine        crashed  -472.251              0.300    -471.951  no      
right_engine       crashed  -251.312              0.300    -251.012  no      
random             crashed   -90.358              0.300     -90.058  no      
ppo_waste_side_15  landed    -43.000             90.000      47.000  yes     
ppo_waste_main_15  timeout  -373.000             90.000    -283.000  yes     
admitted: ['ppo_deterministic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 2  seed 8  eps=0.1000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000             30.000      56.000  yes     
ppo_stochastic     timeout  -341.000             30.000    -311.000  no      
left_engine        crashed  -331.878              0.100    -331.778  no      
main_engine        crashed  -472.251              0.100    -472.151  no      
right_engine       crashed  -251.312              0.100    -251.212  no      
random             crashed   -90.358              0.100     -90.258  no      
ppo_waste_side_15  landed    -43.000             30.000     -13.000  no      
ppo_waste_main_15  timeout  -373.000             30.000    -343.000  yes     
admitted: ['ppo_deterministic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 3  seed 8  eps=0.0000  weights=[0.500, 0.500]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     26.000              0.000      26.000  yes     
ppo_stochastic     timeout  -341.000              0.000    -341.000  no      
left_engine        crashed  -331.878              0.000    -331.878  no      
main_engine        crashed  -472.251              0.000    -472.251  no      
right_engine       crashed  -251.312              0.000    -251.312  no      
random             crashed   -90.358              0.000     -90.358  no      
ppo_waste_side_15  landed    -43.000              0.000     -43.000  no      
ppo_waste_main_15  timeout  -373.000              0.000    -373.000  no      
admitted: ['ppo_deterministic']   selected: ppo_deterministic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                                                 selected           flown 
----  ----  -----  ----  -------------------------------------------------------  -----------------  ------
   0     8  0.500     2  ppo_deterministic, ppo_waste_side_15                     ppo_deterministic  landed
   1     8  0.300     3  ppo_deterministic, ppo_waste_main_15, ppo_waste_side_15  ppo_deterministic  landed
   2     8  0.100     2  ppo_deterministic, ppo_waste_main_15                     ppo_deterministic  landed
   3     8  0.000     1  ppo_deterministic                                        ppo_deterministic  landed

=================================================
E4 run 2 -- the real MetaMo governor, seeds 21-30
=================================================
real MetaMo governor: appraisal scales from seed 21: payoff_scale=75.0 motive_scale=220.9

==============================
Run 2 -- fix OFF (raw epsilon)
==============================

step 0  seed 21  eps=0.1000  weights=[0.632, 0.368]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     72.000              0.100      72.100  yes     
ppo_stochastic     landed     80.000              0.100      80.100  yes     
left_engine        crashed  -244.787              0.100    -244.687  no      
main_engine        crashed  -516.234              0.100    -516.134  no      
right_engine       crashed  -325.045              0.100    -324.945  no      
random             crashed  -214.676              0.100    -214.576  no      
ppo_waste_side_15  timeout  -271.000              0.100    -270.900  no      
ppo_waste_main_15  timeout  -177.000              0.100    -176.900  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 1  seed 22  eps=0.0107  weights=[0.737, 0.263]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     102.000              0.011     102.011  yes     
ppo_stochastic     landed     105.000              0.011     105.011  yes     
left_engine        crashed   -403.870              0.011    -403.859  no      
main_engine        crashed  -1.01e+03              0.011   -1.01e+03  no      
right_engine       crashed   -410.131              0.011    -410.120  no      
random             crashed   -102.717              0.011    -102.707  no      
ppo_waste_side_15  landed      77.000              0.011      77.011  yes     
ppo_waste_main_15  landed      13.000              0.011      13.011  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 2  seed 23  eps=0.0000  weights=[0.791, 0.209]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     80.000              0.000      80.000  yes     
ppo_stochastic     landed     76.000              0.000      76.000  yes     
left_engine        crashed  -423.800              0.000    -423.800  no      
main_engine        crashed  -530.773              0.000    -530.773  no      
right_engine       crashed  -466.278              0.000    -466.278  no      
random             crashed  -239.518              0.000    -239.518  no      
ppo_waste_side_15  landed     18.000              0.000      18.000  yes     
ppo_waste_main_15  landed      1.000              0.000       1.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_waste_main_15   flown outcome: landed

step 3  seed 24  eps=0.0000  weights=[0.801, 0.199]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    105.000              0.000     105.000  yes     
ppo_stochastic     landed     89.000              0.000      89.000  yes     
left_engine        crashed  -170.923              0.000    -170.923  no      
main_engine        crashed  -433.778              0.000    -433.778  no      
right_engine       crashed  -228.247              0.000    -228.247  no      
random             crashed   -55.000              0.000     -55.000  no      
ppo_waste_side_15  timeout  -236.000              0.000    -236.000  no      
ppo_waste_main_15  timeout  -149.000              0.000    -149.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 4  seed 25  eps=0.0000  weights=[0.781, 0.219]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  crashed  -166.000              0.000    -166.000  no      
ppo_stochastic     landed    105.000              0.000     105.000  yes     
left_engine        crashed  -739.983              0.000    -739.983  no      
main_engine        crashed  -841.814              0.000    -841.814  no      
right_engine       crashed  -689.444              0.000    -689.444  no      
random             crashed  -318.160              0.000    -318.160  no      
ppo_waste_side_15  crashed  -202.000              0.000    -202.000  no      
ppo_waste_main_15  crashed  -198.000              0.000    -198.000  no      
admitted: ['ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 5  seed 26  eps=0.0000  weights=[0.767, 0.233]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     71.000              0.000      71.000  yes     
ppo_stochastic     landed     75.000              0.000      75.000  yes     
left_engine        crashed  -220.300              0.000    -220.300  no      
main_engine        crashed  -379.073              0.000    -379.073  no      
right_engine       crashed  -490.306              0.000    -490.306  no      
random             crashed  -256.023              0.000    -256.023  no      
ppo_waste_side_15  landed   -132.000              0.000    -132.000  no      
ppo_waste_main_15  timeout  -176.000              0.000    -176.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 6  seed 27  eps=0.0000  weights=[0.759, 0.241]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     82.000              0.000      82.000  yes     
ppo_stochastic     landed     78.000              0.000      78.000  yes     
left_engine        crashed  -646.140              0.000    -646.140  no      
main_engine        crashed  -613.966              0.000    -613.966  no      
right_engine       crashed  -690.169              0.000    -690.169  no      
random             crashed  -129.557              0.000    -129.557  no      
ppo_waste_side_15  landed    -53.000              0.000     -53.000  no      
ppo_waste_main_15  landed     18.000              0.000      18.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 7  seed 28  eps=0.0000  weights=[0.753, 0.247]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     91.000              0.000      91.000  yes     
ppo_stochastic     landed     97.000              0.000      97.000  yes     
left_engine        crashed  -126.515              0.000    -126.515  no      
main_engine        crashed  -421.653              0.000    -421.653  no      
right_engine       crashed  -286.982              0.000    -286.982  no      
random             crashed  -116.506              0.000    -116.506  no      
ppo_waste_side_15  landed     19.000              0.000      19.000  yes     
ppo_waste_main_15  landed     -3.000              0.000      -3.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 8  seed 29  eps=0.0000  weights=[0.748, 0.252]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     76.000              0.000      76.000  yes     
ppo_stochastic     landed     74.000              0.000      74.000  yes     
left_engine        crashed  -659.146              0.000    -659.146  no      
main_engine        crashed  -497.241              0.000    -497.241  no      
right_engine       crashed  -783.905              0.000    -783.905  no      
random             crashed  -254.622              0.000    -254.622  no      
ppo_waste_side_15  timeout  -253.000              0.000    -253.000  no      
ppo_waste_main_15  timeout  -276.000              0.000    -276.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 9  seed 30  eps=0.0000  weights=[0.745, 0.255]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  timeout  -404.000              0.000    -404.000  no      
ppo_stochastic     landed     83.000              0.000      83.000  yes     
left_engine        crashed  -341.591              0.000    -341.591  no      
main_engine        crashed  -503.699              0.000    -503.699  no      
right_engine       crashed  -393.450              0.000    -393.450  no      
random             crashed   -63.460              0.000     -63.460  no      
ppo_waste_side_15  landed   -318.000              0.000    -318.000  no      
ppo_waste_main_15  landed      4.000              0.000       4.000  yes     
admitted: ['ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_stochastic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                                                                 selected           flown 
----  ----  -----  ----  -----------------------------------------------------------------------  -----------------  ------
   0    21  0.100     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   1    22  0.011     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed
   2    23  0.000     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_waste_main_15  landed
   3    24  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   4    25  0.000     1  ppo_stochastic                                                           ppo_stochastic     landed
   5    26  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   6    27  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
   7    28  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_deterministic  landed
   8    29  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   9    30  0.000     2  ppo_stochastic, ppo_waste_main_15                                        ppo_stochastic     landed
real MetaMo governor: appraisal scales from seed 21: payoff_scale=75.0 motive_scale=220.9

=====================================
Run 2 -- fix ON (guarded scale 300.0)
=====================================

step 0  seed 21  eps=0.1000  weights=[0.632, 0.368]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     72.000             30.000     102.000  yes     
ppo_stochastic     landed     80.000             30.000     110.000  yes     
left_engine        crashed  -244.787              0.100    -244.687  no      
main_engine        crashed  -516.234              0.100    -516.134  no      
right_engine       crashed  -325.045              0.100    -324.945  no      
random             crashed  -214.676              0.100    -214.576  no      
ppo_waste_side_15  timeout  -271.000             30.000    -241.000  no      
ppo_waste_main_15  timeout  -177.000             30.000    -147.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 1  seed 22  eps=0.0107  weights=[0.737, 0.263]  baseline payoff=-100
candidate          outcome  margin     tolerance eps_env  margin+tol  admitted
-----------------  -------  ---------  -----------------  ----------  --------
ppo_deterministic  landed     102.000              3.206     105.206  yes     
ppo_stochastic     landed     105.000              3.206     108.206  yes     
left_engine        crashed   -403.870              0.011    -403.859  no      
main_engine        crashed  -1.01e+03              0.011   -1.01e+03  no      
right_engine       crashed   -410.131              0.011    -410.120  no      
random             crashed   -102.717              0.011    -102.707  no      
ppo_waste_side_15  landed      77.000              3.206      80.206  yes     
ppo_waste_main_15  landed      13.000              3.206      16.206  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_stochastic   flown outcome: landed

step 2  seed 23  eps=0.0000  weights=[0.791, 0.209]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     80.000              0.000      80.000  yes     
ppo_stochastic     landed     76.000              0.000      76.000  yes     
left_engine        crashed  -423.800              0.000    -423.800  no      
main_engine        crashed  -530.773              0.000    -530.773  no      
right_engine       crashed  -466.278              0.000    -466.278  no      
random             crashed  -239.518              0.000    -239.518  no      
ppo_waste_side_15  landed     18.000              0.000      18.000  yes     
ppo_waste_main_15  landed      1.000              0.000       1.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15', 'ppo_waste_side_15']   selected: ppo_waste_main_15   flown outcome: landed

step 3  seed 24  eps=0.0000  weights=[0.801, 0.199]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed    105.000              0.000     105.000  yes     
ppo_stochastic     landed     89.000              0.000      89.000  yes     
left_engine        crashed  -170.923              0.000    -170.923  no      
main_engine        crashed  -433.778              0.000    -433.778  no      
right_engine       crashed  -228.247              0.000    -228.247  no      
random             crashed   -55.000              0.000     -55.000  no      
ppo_waste_side_15  timeout  -236.000              0.000    -236.000  no      
ppo_waste_main_15  timeout  -149.000              0.000    -149.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 4  seed 25  eps=0.0000  weights=[0.781, 0.219]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  crashed  -166.000              0.000    -166.000  no      
ppo_stochastic     landed    105.000              0.000     105.000  yes     
left_engine        crashed  -739.983              0.000    -739.983  no      
main_engine        crashed  -841.814              0.000    -841.814  no      
right_engine       crashed  -689.444              0.000    -689.444  no      
random             crashed  -318.160              0.000    -318.160  no      
ppo_waste_side_15  crashed  -202.000              0.000    -202.000  no      
ppo_waste_main_15  crashed  -198.000              0.000    -198.000  no      
admitted: ['ppo_stochastic']   selected: ppo_stochastic   flown outcome: landed

step 5  seed 26  eps=0.0000  weights=[0.767, 0.233]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     71.000              0.000      71.000  yes     
ppo_stochastic     landed     75.000              0.000      75.000  yes     
left_engine        crashed  -220.300              0.000    -220.300  no      
main_engine        crashed  -379.073              0.000    -379.073  no      
right_engine       crashed  -490.306              0.000    -490.306  no      
random             crashed  -256.023              0.000    -256.023  no      
ppo_waste_side_15  landed   -132.000              0.000    -132.000  no      
ppo_waste_main_15  timeout  -176.000              0.000    -176.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 6  seed 27  eps=0.0000  weights=[0.759, 0.241]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     82.000              0.000      82.000  yes     
ppo_stochastic     landed     78.000              0.000      78.000  yes     
left_engine        crashed  -646.140              0.000    -646.140  no      
main_engine        crashed  -613.966              0.000    -613.966  no      
right_engine       crashed  -690.169              0.000    -690.169  no      
random             crashed  -129.557              0.000    -129.557  no      
ppo_waste_side_15  landed    -53.000              0.000     -53.000  no      
ppo_waste_main_15  landed     18.000              0.000      18.000  yes     
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_deterministic   flown outcome: landed

step 7  seed 28  eps=0.0000  weights=[0.753, 0.247]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     91.000              0.000      91.000  yes     
ppo_stochastic     landed     97.000              0.000      97.000  yes     
left_engine        crashed  -126.515              0.000    -126.515  no      
main_engine        crashed  -421.653              0.000    -421.653  no      
right_engine       crashed  -286.982              0.000    -286.982  no      
random             crashed  -116.506              0.000    -116.506  no      
ppo_waste_side_15  landed     19.000              0.000      19.000  yes     
ppo_waste_main_15  landed     -3.000              0.000      -3.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic', 'ppo_waste_side_15']   selected: ppo_deterministic   flown outcome: landed

step 8  seed 29  eps=0.0000  weights=[0.748, 0.252]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  landed     76.000              0.000      76.000  yes     
ppo_stochastic     landed     74.000              0.000      74.000  yes     
left_engine        crashed  -659.146              0.000    -659.146  no      
main_engine        crashed  -497.241              0.000    -497.241  no      
right_engine       crashed  -783.905              0.000    -783.905  no      
random             crashed  -254.622              0.000    -254.622  no      
ppo_waste_side_15  timeout  -253.000              0.000    -253.000  no      
ppo_waste_main_15  timeout  -276.000              0.000    -276.000  no      
admitted: ['ppo_deterministic', 'ppo_stochastic']   selected: ppo_deterministic   flown outcome: landed

step 9  seed 30  eps=0.0000  weights=[0.745, 0.255]  baseline payoff=-100
candidate          outcome  margin    tolerance eps_env  margin+tol  admitted
-----------------  -------  --------  -----------------  ----------  --------
ppo_deterministic  timeout  -404.000              0.000    -404.000  no      
ppo_stochastic     landed     83.000              0.000      83.000  yes     
left_engine        crashed  -341.591              0.000    -341.591  no      
main_engine        crashed  -503.699              0.000    -503.699  no      
right_engine       crashed  -393.450              0.000    -393.450  no      
random             crashed   -63.460              0.000     -63.460  no      
ppo_waste_side_15  landed   -318.000              0.000    -318.000  no      
ppo_waste_main_15  landed      4.000              0.000       4.000  yes     
admitted: ['ppo_stochastic', 'ppo_waste_main_15']   selected: ppo_stochastic   flown outcome: landed

summary:
step  seed  eps    #adm  admitted                                                                 selected           flown 
----  ----  -----  ----  -----------------------------------------------------------------------  -----------------  ------
   0    21  0.100     2  ppo_deterministic, ppo_stochastic                                        ppo_stochastic     landed
   1    22  0.011     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_stochastic     landed
   2    23  0.000     4  ppo_deterministic, ppo_stochastic, ppo_waste_main_15, ppo_waste_side_15  ppo_waste_main_15  landed
   3    24  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   4    25  0.000     1  ppo_stochastic                                                           ppo_stochastic     landed
   5    26  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   6    27  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_main_15                     ppo_deterministic  landed
   7    28  0.000     3  ppo_deterministic, ppo_stochastic, ppo_waste_side_15                     ppo_deterministic  landed
   8    29  0.000     2  ppo_deterministic, ppo_stochastic                                        ppo_deterministic  landed
   9    30  0.000     2  ppo_stochastic, ppo_waste_main_15                                        ppo_stochastic     landed

========================================================
E4 summary -- does the admitted set change with epsilon?
========================================================
run  fix  eps range     admitted-set changes between steps  step pairs  total admissions
---  ---  ------------  ----------------------------------  ----------  ----------------
1a   off  0.000..0.500                                   0           3                 4
1a   on   0.000..0.500                                   1           3                 6
1b   off  0.000..0.500                                  15          19                47
1b   on   0.000..0.500                                  17          19                51
1c   off  0.000..0.500                                   1           3                 7
1c   on   0.000..0.500                                   3           3                 8
  2  off  0.000..0.100                                   8           9                25
  2  on   0.000..0.100                                   8           9                25

fix on vs fix off, step by step (same seeds, same governor script):
run  steps whose admitted set differs  steps     
---  --------------------------------  ----------
1a   2/4                               0, 1      
1b   3/20                              12, 13, 18
1c   4/4                               0, 1, 2, 3
  2  0/10                              -         
run 1a (only epsilon varies): fix off -> 1 distinct admitted set(s); fix on -> 2 distinct admitted sets over eps [0.5, 0.3, 0.1, 0.0]
```

### A.7 E5 -- e5_minecraft_generality.py

```text

=============================================================
E5 -- the guarded epsilon scale on the Minecraft stub (m = 6)
=============================================================
calibration: 160 decisions from episodes 1-20; mean|dr|=0.0020, mean|dn|=0.0400, floor=0.01; scale = max(s_r,floor)+max(s_n,floor) = 0.0500
evaluation: 160 decisions x 5 candidates from episodes 21-40; raw margins min=-0.0826 median=-0.0354 max=-0.0146
with the fix, eps in [0, 0.5] maps to a tolerance in [0, 0.0250]
guard fired (candidate's own task payoff < 0): 0 times

gate                    eps=0  eps=0.05  eps=0.1  eps=0.2  eps=0.3  eps=0.5  flips 0->0.5  seeds  cands  mono viol
----------------------  -----  --------  -------  -------  -------  -------  ------------  -----  -----  ---------
raw eps (fix off)       0/800  560/800   800/800  800/800  800/800  800/800  800 (100.0%)     20      5          0
guarded scale (fix on)  0/800  0/800     0/800    0/800    20/800   200/800  200 (25.0%)      20      2          0
margins epsilon can decide: raw band [-0.5, 0): 800/800; fixed band [-0.0250, 0): 200/800

===============================
Minecraft and bridge test files
===============================
command: python -m pytest tests/test_minecraft_stub.py tests/test_minecraft_rollout.py tests/test_run_metamo_pipeline.py tests/test_bridge_budget.py tests/test_bridge_controller.py tests/test_bridge_e2e.py tests/test_bridge_governor.py tests/test_bridge_weights.py tests/test_bridge_epsilon_scale.py -q -p no:cacheprovider
153 passed in 5.06s
```

### A.8 Full suite

```text
$ python -m pytest tests/ -q -p no:cacheprovider
1171 passed, 2 xfailed, 361 warnings in 88.30s (0:01:28)
```
