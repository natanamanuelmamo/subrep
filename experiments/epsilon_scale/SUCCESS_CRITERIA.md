# Success criteria for the ε-scale options

These criteria were written **before** the option comparison (E1–E3) was run. If they are refined later, the change goes under *Amendments* with a date, and the text below stays as originally written.

An option **fixes the ε-scale problem** only if all six criteria hold on the **evaluation seeds 21–40**. Seeds 1–20 are used only to choose scales.

- **ε grid:** {0, 0.05, 0.1, 0.2, 0.3, 0.5}.
- **Pair:** one (candidate, seed) combination.
- **Admission rule:** every option is expressed as a normalised margin μ, and a candidate is admitted iff μ ≥ −ε (`options.py`).

| # | Criterion | Original wording | Threshold used here |
|---|---|---|---|
| 1 | ε matters | A meaningful share of candidate-seed pairs change between admitted and rejected as ε moves from 0 to 0.5. A handful of flips that all come from one seed doesn't count. | **≥ 10 %** of pairs flip between ε = 0 and ε = 0.5, **and** the flips come from **≥ 5 distinct seeds** and **≥ 2 distinct candidates**. Reported separately for the task's candidate sets (default, extended) and for the borderline set defined in `candidates.py`. |
| 2 | Safety is kept | Crash-only candidates are rejected at every ε in [0, 0.5]. | **100 %** of pairs where the candidate crashed are rejected at **every** ε in the grid, including ε = 0.5. |
| 3 | Good skills still pass | Clear landings (PPO deterministic, when it lands) are admitted at the neutral ε = 0.1. | **≥ 95 %** of `ppo_deterministic` pairs that landed are admitted at ε = 0.1. |
| 4 | Monotonic | Lowering ε never admits more candidates. | **0 violations**: for every seed, admitted(ε_a) ⊆ admitted(ε_b) whenever ε_a < ε_b, over the grid and over a 51-point sweep of [0, 0.5]. |
| 5 | Stable | Admissions don't swing wildly between seeds that have similar candidates. | **Fixed scales:** the coefficient of variation across the per-seed estimates of each scale (seeds 1–20) is **≤ 0.25**. **Both scale sources:** among pairs of the same candidate with the same outcome class (landed / crashed / timed out), the admission at ε = 0.1 agrees with that group's majority in **≥ 90 %** of pairs. Admission agreement between per-step and fixed scales is also reported. |
| 6 | Generic | The same code works for m = 2 (LunarLander) and m = 6 (Minecraft stub), and doesn't break the Minecraft tests. | The same option code runs unchanged at m = 2 and m = 6, with no hard-coded 2. All Minecraft and bridge tests pass. On the Minecraft stub with the fix on, ε still changes **≥ 1** admission over a threat cycle. |

## Ranking rule if several options pass
1. Options that leave the records unchanged, because they have no selection side effect (E3).
2. Fixed scales over per-step scales.
3. A higher flip share on the task's candidate sets.
4. Simpler code.

## If no option reaches criterion 1 on the task's candidate sets
This is a sign that the set is bimodal: clear crashes and clear landings, with nothing in between. In that case:
- criterion 1 is judged on the borderline set;
- the report says explicitly which part of the result comes from the option and which from the candidate set.

## Amendments
_None._
