# Known open defects

Every item below was identified in the Phase 1 audit (`AUDIT_REPORT.md`) and is
**not fixed**. They are recorded here so that no reader has to rediscover them,
and so that no claim in `README.md` silently depends on one being absent.

Fixed defects are not listed here; see the git log (`git log --oneline`), where
each fix is one commit naming its defect.

---

## Open — affects what the numbers mean

### M1 — the baseline performs zero modifications, not one
`levels.py`, the `self_mod and level >= 2` branch

Tětek et al. Theorem 7 compares `π_t` against `π_1`: the policy after *t*
self-modifications against the policy after **one**. The L2/L3 baseline arm here
never modifies at all, so `D_k` compares against `π_0`. There is no `elif`
covering the L2/L3 baseline (only L1 has one). The module docstring previously
claimed "self-mod disabled after step 1", which was wrong; it now states what
the code does.

Consequence: `D_k` is not the quantity Theorem 7 bounds. This is one of the
reasons the Theorem 7 comparison was removed rather than repaired.

### M8 — the discount horizon is shorter than the modification period
`levels.py` (`n_steps`, `steps_per_mod`), `metrics.py` (`value_from_step_vec`)

γ is applied **per integration step**, not per second. The effective horizon
`1/(1−γ)` is 10 / 20 / 100 steps = **0.1 s / 0.2 s / 1.0 s** at `dt = 0.01`,
while `T_mod` is 100 steps = **1.0 s**.

At γ = 0.9 and γ = 0.95, `V_k` is ~90% determined within the first 10–20 steps
after modification *k* and has decayed to nothing before modification *k+1*
exists. Compounding across modifications — the entire mechanism Theorem 7
describes — cannot appear inside the measured window at two of the three γ
values. Only γ = 0.99 gives a value horizon comparable to the modification
period.

### M10 — per-seed results are not reproducible across seed counts
`plants.py`, the `rng.standard_normal(size=ds.shape)` calls

Process noise is drawn as one `(S, n)` matrix per RK4 stage, so seed 0's
realisation depends on the batch size `S`. Same config, `base_seed = 10042`:

```
n_seeds=2   -> V(seed 0, k=1) = 9.820660
n_seeds=3   -> V(seed 0, k=1) = 9.819467
n_seeds=200 -> V(seed 0, k=1) = 9.820444
```

Runs are byte-identical at fixed `S` (`tests/test_determinism.py` verifies
this), but `--seeds 100` and `--seeds 200` do not share a single per-seed
number. Any statement of the form "seed 47 diverged" is meaningful only
relative to the `--seeds` value it was produced with.

### M12 — CSTR linearisation is fixed, mis-sited, and has a sign error
`plants.py` (`linearise`, `loop_tf`), `metrics.py` (`plant.linearise(0.8)`)

Three separate problems:

- **Fixed, not tracked.** `linearise(0.8)` is hardcoded; the operating point is
  never recomputed as the reactor moves.
- **Sign error.** `A[0,1] = +drate_dth`. Since `dc/dt = (c_in − c) − rate`, the
  correct entry is `−drate_dth`. Finite-difference check: analytic `+2.530e-9`
  vs FD `−2.530e-9`.
- **Wrong operating point.** `steady_state(0.8)` returns c = 1.000, θ = 0.867 —
  essentially zero conversion, since `exp(−20/0.867) ≈ 1e-10`. The simulation
  runs at θ ∈ [1.4, 1.6]. The linearisation describes a reactor state the
  experiment never visits, which is also why the sign error is currently
  numerically invisible.

Consequence: every CSTR margin is computed for the wrong plant. The README's
surviving claims are Rohrs-only for this reason.

### M14 — H and σ_n do nothing for L0 and L1
`experiment.py` (`run_sweep` combination loop), `levels.py` (the
`level >= 2` gate)

`H` and `sigma_n` enter only through `epsilon_optimize`, which is gated on
`level >= 2`. Verified: for L1,
`max |D(H=5, σ=0.0) − D(H=100, σ=0.1)|` at matched seeds is exactly **0.0**.

Half the sweep's configurations therefore vary two dimensions that provably
have no effect, producing replicates rather than distinct conditions. Any
config count quoted for L0/L1 overstates the number of distinct conditions by
9×.

---

### The divergence finding is confounded by γ_a being both swept and modifiable
`experiment.py` (γ_a sweep dimension), `controllers.py` (`gamma_up`/`gamma_down`)

γ_a is a configuration dimension **and** a parameter L3 can modify. A
verification run (README §3.1c) established a hard stability boundary at
γ_a ≈ 14–15 (ω = 16.1) and 16–18 (ω = 5.0) for a controller that never
self-modifies. L3 reaches γ_a = 20–80.

The headline divergence result therefore does **not** show that
self-modification is destabilising per se. It shows that a margin-blind
objective will walk a free parameter across a stability boundary. The README
states this; it is recorded here because the confound is structural and has not
been designed out.

A cleaner experiment would fix γ_a outside the agent's reach and let it modify
only parameters with no independent stability boundary, or add a margin term to
the objective and test whether the walk still happens. Neither was done.

## Open — statistical

### M7 (residual) — no uncertainty on any headline number
`analysis.py`

The exponential fit and the heavy-tail claim were removed. What remains is
descriptive, but still reports point estimates without confidence intervals,
and pools rows that are not independent: each configuration contributes 9
correlated modification steps × 200 seeds. No clustering correction is applied
anywhere. Reported medians and percentiles should be read as descriptive
summaries of this particular sweep, not as population estimates.

### Effective sample size is far below the nominal seed count
`levels.py` (`make_ref_fn`), `plants.py` (noise injection)

Seeds differ **only** by additive derivative noise. The reference signal is
identical across all seeds, and there is no disturbance signal. On the CSTR the
decision process is fully degenerate: candidate-selection counts come out as
exact multiples of the seed count, meaning every seed makes the identical choice
at every modification step. On Rohrs seeds do diverge, but several
configurations still show all seeds agreeing on the same candidate.

### Noise injection is not a consistent SDE discretisation
`plants.py` (`dynamics`), `plants.py` (`rk4_step`)

Noise is added to the **derivative** and re-drawn independently at each of the
four RK4 stages. This is not a consistent discretisation of any stochastic
differential equation, so `sigma_n` and `ep_noise` do not have the magnitudes
their names suggest. Treat them as relative perturbation knobs, not as
physically calibrated noise intensities.

---

## Open — measurement scope

### ε_emp is not the ε of Theorem 7
`agent.py` (`epsilon_optimize`)

The theorem's ε satisfies `Q(π) ≥ sup_π' Q(π') − ε` over **all** policies on the
infinite-horizon Q. `eps_emp` is a max over the agent's 8 (L2) or 14 (L3)
hand-designed candidates, on an **H-step** exact rollout. It is a lower bound on
the true ε of unknown tightness. It is reported descriptively and is not
compared against any bound.

### C4 (residual) — ε and D remain measured over different horizons
`agent.py` vs `levels.py`

ε spans H steps; `D_k` spans the episode remainder (900 − 100k steps). Since
`r ∈ [0,1]`, at γ = 0.99 and H = 5 the maximum attainable ε is 4.90 while the
maximum attainable D is ~99.99 — a 20× mismatch. The bound comparison that this
invalidated has been removed; the two quantities are still reported side by side
and must not be differenced.

### The optimizer's-curse gap conflates two effects, and δ_m was never swept
`levels.py` (`self_deception` column), `experiment.py` (`--delta-m`)

`self_deception = in_sample − true_sel` differences a noisy-model rollout
against an exact-plant rollout, so it mixes **selection bias** (the optimizer's
curse proper) with **model bias** (δ_m). No second independent noisy draw exists
to separate them.

The believed-vs-realised gap reported in README §3.3 has the same two
explanations, and this design cannot distinguish them. Measured from the sweep,
believed minus realised (Rohrs L2/L3, pooled over γ, H, ω):

| σ_n | γ_a = 1 | γ_a = 10 | γ_a = 50 |
|-----|---------|----------|----------|
| 0.0 | 0.138 | 0.156 | 0.066 |
| 0.01 | 0.140 | 0.257 | 0.422 |
| 0.1 | 0.130 | 0.256 | 0.421 |

The gap does not vanish at σ_n = 0, and at γ_a = 1 is flat in σ_n. That is
suggestive but **not** diagnostic: σ_n = 0 removes additive scoring noise while
leaving model error intact, and selecting the argmax of a deterministically
biased score is itself selection bias.

**δ_m is fixed at 0.5 in all 972 configurations.** The model-bias axis was never
swept, so it cannot be separated from the curse on this data. Separating them
requires sweeping δ_m — including δ_m = 0, where the agent's model is the true
plant — with σ_n = 0. That experiment has not been run, and no mechanism claim
should be made until it is.

### `margin_down` conflates margin loss with margin reallocation
`levels.py` (`margin_down`, `MARGIN_REL_TOL`)

The flag fires when **either** gain margin or phase margin falls by more than
1%, so a modification that trades one against the other is counted as a
degradation. Across all 224,768 flagged Rohrs steps, only **54.9%** have both
margins down; 15.9% are GM-down/PM-up and 29.3% are PM-down/GM-up. At γ_a = 50
the flag is dominated by trades — 66.9% PM-down/GM-up, 33.1% both.

A single scalar robustness measure (a sensitivity peak, a disc margin, or a
distance-to-instability) would not have this ambiguity. The flag was left as-is
because changing it would alter every quadrant number after the fact; the
decomposition is reported in README §3.3 instead.

### Per-candidate margin attribution is not recoverable from the results file
`levels.py` (row construction), `results/sweep_results.csv`

`chosen`, the index of the selected candidate, is computed at every modification
step but never written to the output. Margin change per step is recorded
(`gain_margin_pre` / `gain_margin`), so *when* margin is lost is answerable, but
*which modification* lost it is not, without re-running the sweep. This blocks
the natural follow-up to the margin-consumption finding: ranking candidates by
total margin consumed.

---

## Open — minor

| ID | Location | Issue |
|----|----------|-------|
| m4 | `tests/test_rohrs_instability.py` | `assert max_y > 10.0` passes on a **bounded** response: max \|y\| = 24.80 with the final-10% max at 24.70 — a saturation-limited limit cycle. The quantity that actually diverges (θ₂ → −5.7×10⁵) is not asserted. The test's own name overstates what it checks. |
| m5 | `tests/test_epsilon_zero.py` | With `sigma_n = 0, delta_m = 0` the noisy and exact rollouts call bit-identical plant objects, so `eps = 0` holds **by construction** rather than by good selection. The test does confirm modifications are proposed and accepted (75% non-`no_change`), but cannot distinguish a correct ε-optimizer from any argmax. All 10 seeds also produce one trajectory (`ep_noise = 0`). |
| m6 | `tests/test_value_bounds.py` | Tautological: `exp(−(w_e·e² + w_u·du²))` with non-negative weights lies in (0,1] identically, so the assertion cannot fail for any input. It is never run on the sweep's own trajectories. |
| m7 | `experiment.py` | `python experiment.py > log.txt` crashes on Windows with `UnicodeEncodeError` on the `γ`/`σ`/`ω` characters in the progress line. Workaround: `PYTHONIOENCODING=utf-8`. |
| m8 | `agent.py` | `big_ref_fn` uses `np.repeat` while the states are built with `np.tile` — inconsistent interleaving. Inert **only** because `make_ref_fn` returns the same reference for every seed. It will silently corrupt candidate scoring the moment per-seed references are introduced. |
| m9 | `plants.py` (`Constraints.apply`) | `sat_viol` is evaluated **after** rate-limiting, so a command far outside bounds that gets rate-limited into range is never counted as saturation. |
| m11 | `levels.py` | Dead code: `p_eff = params` followed by `if frozen: p_eff = params`. |
| m12 | — | `no_change` is selected 0/1800 times at γ_a = 1. There is no accept/reject path; the agent modifies at every opportunity. "Accepted modification" in any write-up means "selected", not "passed a test". |
| C5 (residual) | `controllers.py` (`Candidate.apply`) | `deadzone_up` / `deadzone_down` remain **structural no-ops**: `dead_zone` initialises to 0.0, and both `×2.0` and `max(/2.0, 0.0)` leave it at 0. Two of L3's six exclusive candidates can never do anything. |
| C5 (residual) | `plants.py` (noise draw shape) | L2 and L3 draw agent-scoring noise as `(n_cand·S, 3)` matrices of **different sizes** (1600 vs 2800), so candidate 0 / seed 0 sees different noise in L2 than in L3. The two ladder levels are not paired in the agent's noise, which confounds any direct L2-vs-L3 difference. |

---

## Verified clean

Recorded so that later work does not re-litigate them. Evidence is in
`AUDIT_REPORT.md`.

- **Plant transfer function.** `dx1 = −x1 + 2u`, `dx2 = x3`,
  `dx3 = 229x1 − 229x2 − 30x3`, `y = x2` gives exactly
  `2/(s+1) · 229/(s²+30s+229)`, matching Rohrs et al. (1985).
- **Integrator step.** Fastest eigenvalue |λ| = 15.13 (poles −15 ± 2j), so
  |λ|·dt = 0.151 at dt = 0.01, well inside RK4's ~2.785 limit. Halving and
  quartering dt leaves k\* unchanged. No numerical artefact.
- **Pairing.** Both arms consume `plant_rng` at an identical rate;
  `max |Δreward|` over steps before the first modification is exactly 0.0.
- **Theorem 7 transcription.** `min(ε/γ^(t−1), 1/(1−γ))` with γ^(t−1) in the
  denominator and `u: (A×E)* → [0,1]`, verified symbol-by-symbol against the
  arXiv TeX source (`policy1.tex`). The transcription in the README was correct;
  the *comparison built on it* was not.
- **Symmetric termination.** No level terminates early; all episodes run the
  full `n_steps`; constraints are applied identically. The divergence warm-up
  window is the same for every level and both arms.
- **Determinism at fixed S.** No unseeded RNG, no set/dict iteration-order
  dependence, no parallel reduction. See M10 for the `S`-dependence caveat.
