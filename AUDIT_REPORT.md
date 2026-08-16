# Adversarial Audit — Phase 1 Report

Audit performed against commit `pre-audit baseline`. No repository file was
modified during the audit. Verification performed: `pytest` (10/10 pass), full
216-config × 200-seed sweep re-run to a scratchpad copy and diffed against
`results/sweep_results.csv`, both cited papers checked against their arXiv TeX
sources, all DOIs resolved via Crossref.

Environment: Python 3.13.5, numpy 2.3.5, scipy 1.16.3, control 0.10.2,
pandas 3.0.5, Windows 11.

---

## CRITICAL

### C1 — The margin computation evaluates a *reversed* transfer function. Q3 is void.

`metrics.py:75-83`

```python
num = np.polyval(num_coeffs[::-1], s)
den = np.polyval(den_coeffs[::-1], s)
```

`np.polyval` expects **decreasing** power order. `_tf_coeffs_rohrs` already
returns decreasing order (`den = [1, 31, 259, 229]` = s³+31s²+259s+229), and
`ct.tf(...).den[0][0]` for the CSTR does too. The `[::-1]` therefore evaluates
**229s³ + 259s² + 31s + 1** — a different plant.

```
code    G(j1) = -1.11720 + 0.85739j
correct G(j1) =  0.85739 - 1.11720j   # exactly the reversed polynomial
```

Every `gain_margin` / `phase_margin` value in `results/sweep_results.csv` is
fiction. The sweep reports GM = 0.149, PM = −32.3° for the baseline L2
controller — i.e. the linear analysis says the loop is unstable before any
self-modification, while the simulation never leaves |y| ≤ 2. True margins:

| Controller | code reports | truth (`ct.margin`) |
|---|---|---|
| MRAC + Ki/s (as simulated) | GM 0.149, PM −32.3° | GM 26.0, PM 60.0° |
| + toggle_lead | GM 0.428, PM −7.1° | GM 11.2, PM 77.7° |
| + toggle_filter | GM 0.141, PM −34.8° | GM 12.5, PM 56.0° |

**Why the tests missed it:** `tests/test_nominal_stability.py:42` is the only
test of this path and uses the nominal plant, whose denominator `[1, 1]` is a
**palindrome**. The bug is invisible in the one case that is tested.

**Fix:** delete both `[::-1]` slices; add a regression test on a
non-palindromic denominator.

### C2 — The margin model is not the controller that is simulated.

`metrics.py:110-135`, mirrored in `controllers.py:200-224` and `metrics.py:240-250`

1. **The integrator is dropped whenever MRAC is on.** `C = np.where(mrac_on,
   C_mrac, C_pi)` (metrics.py:125). But `controllers.py:136` adds `Ki·∫e` on
   top of the MRAC law regardless of `mrac_on`, and L1–L3 all have
   `mrac_on=True` (`controllers.py:80,91`).
2. **`theta2_init` is used, not the adapted `theta2`.** `metrics.py:115` reads
   the constant −0.5 that `apply_choices` never touches (`agent.py:234-236`),
   while the real feedback gain `ctrl_state[:,TH2]` adapts every step.

Consequence: of the 14 L3 candidates, **10 produce a bit-identical margin**.
Only `toggle_lead` and `toggle_filter` move it. The margin takes exactly
**4 distinct values** across all 54,000 L2-Rohrs rows.

### C3 — The instability detector cannot fire.

`levels.py:207` — `np.abs(y[:, 0]) > 1e6`

The Rohrs plant is open-loop stable (poles −1, −15±2j) and the actuator
saturates at |u| ≤ 10 (`levels.py:52`). Measured step-response peak is 2.0, so
**|y| ≤ 20 for all time, for any controller**. The threshold is 50,000× outside
the reachable set. CSTR: max |θ| = 2.33 under any admissible u ∈ [0.3, 3.0].

README §3 Q3 calls this "a clean negative for the destabilisation question". It
is not a negative result — the experiment has no positive branch. k\* = −1 in
432,000/432,000 rows by construction. This is also failure mode #10: a |y|
threshold alone, with no distinction between divergence, saturation, and a
bounded transient.

The thing that actually diverges in the genuine Rohrs phenomenon is the adapted
parameter: measured θ₂ → −5.7×10⁵ while |y| stayed at 24.8.

### C4 — ε and D are measured over different horizons.

`agent.py:179-191` vs `levels.py:225-227`

ε is the gap between two length-**H** rollouts (H ∈ {5, 20, 100} steps). D_k is
the gap between two **episode-remainder** values (1000 − 100k steps). Since
r ∈ [0,1]:

| γ | H | max attainable ε | max attainable D | ratio |
|---|---|---|---|---|
| 0.99 | 5 | 4.90 | 99.99 | 20.4× |
| 0.99 | 20 | 18.21 | 99.99 | 5.5× |
| 0.90 | 100 | 10.00 | 10.00 | 1.0× |

Flag rate tracks the mismatch exactly: 0.817 at (γ=0.99, H=5); 0.429 at
(γ=0.90, H=100). The record "violation" (D = 12.87) has ε_emp = 0.121 — an ε
that cannot exceed 4.9 compared against a D that can reach 100.

README lines 130-136 blame "trajectory divergence" and call it "a known
limitation of the paired-trajectory design, not an implementation error". That
diagnosis is not supported.

Compounding: **Theorem 7's ε is a sup over all policies** (verified in the
paper's `definitions.tex`); ε_emp is a max over 8 or 14 hand-picked candidates —
a lower bound of unknown tightness. And a same-state comparison
(`decision['true_values'][0]`, `agent.py:183`) is already computed and
discarded.

### C5 — L3 has collapsed into L2 on the primary plant.

`controllers.py:262-265, 286-293`

Instrumented selection over 1800 decisions (200 seeds × 9 steps) at Rohrs,
γ=0.95, H=20, σ=0.01:

```
L3 rohrs:  toggle_lead 42.1%   toggle_filter 4.4%   Ki_up 53.4%
           gamma_up/down, sigma_up/down, deadzone_up/down:  0 / 1800  (0.0%)
L3 cstr:   only gamma_down ever chosen (11.1%); deadzone: 0 / 1800
```

- **`deadzone_up`/`deadzone_down` are structural no-ops.** `dead_zone`
  initialises to `0.0` (`controllers.py:94`); `×2.0` and `max(/2.0, 0.0)` are
  both 0.
- **The comparison is confounded.** `rollout` draws
  `rng.standard_normal(size=(n_cand·S, 3))` (`plants.py:126,137`). L2 draws
  (1600,3), L3 draws (2800,3), so candidate 0 / seed 0 sees different noise in
  L2 than in L3. The observed max |D_L2 − D_L3| = 3.74 comes entirely from
  RNG-stream divergence.

### C6 — The committed `results/` do not match the current code.

Full re-run (216 configs, 200 seeds, 451.9 s), joined on
`(level, plant, gamma, H, sigma_n, seed, mod_step)`:

```
V_selfmod    maxdiff = 0.0        (bit-identical)
V_baseline   maxdiff = 0.0
D            maxdiff = 0.0
gain_margin  maxdiff = 0.0
eps_emp      NaN new = 237,600    NaN old = 0        <-- committed file has 0.0
bound_violated   new = 111,529    old = 176,047      disagree = 64,518
```

The committed CSV stores `eps_emp = 0.0` where the current code writes `NaN` —
for all 216,000 L0/L1 rows and all 21,600 `mod_step == 10` rows. Those phantom
zeros create a bound of 0 that any D > 1e-8 exceeds.

| README §3 | committed CSV | current code |
|---|---|---|
| flagged rows | 132,847 / 216,000 | 111,529 / 194,400 |
| ε = 0 flagged | 103,969 (78%) | 82,651 (74%) |
| ε mean, L2 Rohrs | 0.0254 | 0.0282 |
| ε mean, L3 Rohrs | 0.0185 | 0.0205 |
| ε p95, L2 CSTR | 0.00001 | 0.000027 |
| ε p95, L3 Rohrs | 0.0782 | 0.0861 |
| val-up-margin-down L2 | 0.206 (22291/108000) | 0.229 (22291/97200) |
| val-up-margin-down L3 | 0.241 (25980/108000) | 0.267 (25980/97200) |

---

## MAJOR

- **M1** `levels.py:150-180` — The L2/L3 baseline is *zero* modifications, not
  π₁. Theorem 7 compares π_t against π₁. README §2 says "disabled after step 1";
  the code implements π₀.
- **M2** `levels.py:112,133,137,225-226` — Only 9 of the advertised 10
  modification steps occur (`t % 100 == 0 and t > 0` over `range(1000)` →
  k = 1…9). Row k=10 has `eps = NaN`, `chosen = -1`, and `mod_steps` clipped to
  999 so `V` is a **single reward sample** (0.9966 vs ~9.8 at k=1). Also the
  value horizon shrinks with k: V_1 spans 900 steps, V_9 spans 100.
- **M3** `levels.py:141-182` — `val_up_margin_down` is off by one step
  (margins computed *before* modification k, `prev_*` updated after), and
  `prev_gm = prev_pm = inf` at init makes `margin_down` unconditionally True at
  k=1. Measured: 0.238 at k=1, exactly 0.000 at k=2.
- **M4** `levels.py:164-168` — `val_up` uses the **true** value
  (`true_sel` vs `true_values[0]`), not the in-sample value, despite the dead
  first assignment and the comment both saying "in-sample". README's
  "the agent believes it is improving" is not what was measured.
- **M5** `levels.py:150` — `not np.any(unstable)` couples all seeds: one
  unstable seed halts self-modification for the whole batch, and S varies with
  `--seeds`. Latent only because C3 means `unstable` is never set.
- **M6** `analysis.py:43,58` — Q1 averages D across all three γ (whose value
  scales differ 10×) and draws the bound at `sub['gamma'].iloc[0]`, an
  arbitrary γ. `q1_deterioration.png` is not interpretable.
- **M7** `README:140-151`, `analysis.py:93-103,253-261` — Q2's median/p95/max
  table is from one config (spread 0.7267→0.7298, a 0.4% range over 200 seeds);
  the "max D up to 12.87" is from all configs at a different γ. The 10 medians
  being fitted are sign-alternating and non-monotone
  (−0.22, −0.57, −0.17, −1.18, +0.73, −0.83, −0.42, +0.08, +2.70, +0.05);
  `A·exp(B·k)` is strictly positive and cannot represent them. No clustering is
  accounted for anywhere.
- **M8** `levels.py:272-273`, `metrics.py:52` — γ is applied per integration
  step, not per second. Effective horizon 1/(1−γ) = 0.1 / 0.2 / 1.0 s at
  dt = 0.01, vs `T_mod` = 1.0 s. At γ = 0.9 and 0.95 the value window closes
  before the next modification exists, suppressing the compounding that
  Theorem 7 describes.
- **M9** `controllers.py:92`, `levels.py:39` — The Rohrs regime is never
  entered. `omega = 5.0  # rad/s, excites unmodeled dynamics (poles ~15 rad/s)`
  is self-refuting. Measured at the sweep's own `gamma_a = 1.0`:
  max|y| = 1.43 at 5 rad/s, 1.23 at 16.1 rad/s. The repo's own instability test
  needs γ_a = 50.
- **M10** `plants.py:126,137` — Noise is drawn as one `(S, n)` matrix per RK4
  stage, so seed 0's realisation depends on **S**. Same config, `base_seed=10042`:
  `n_seeds=2 → 9.820660`, `n_seeds=3 → 9.819467`, `n_seeds=200 → 9.820444`.
  `test_determinism` only compares at fixed S.
- **M11** `agent.py:81-83`, `plants.py:119-127` — With `delta_m > 0` the agent
  model is `RohrsPlant(unmodeled=False)` (`n_states=1`) but is seeded with the
  3-state real plant state; `rk4_step` computes `(S,3) + (dt/6)*(S,1)`, which
  NumPy **broadcasts**. Separately the nominal model's `output` returns `x1`
  while the real plant returns `x2`.
- **M12** `plants.py:247-269`, `metrics.py:103` — CSTR linearisation is
  hardcoded at `linearise(0.8)`; `A[0,1] = +drate_dth` has the wrong sign
  (should be `−drate_dth`; FD check: analytic +2.530e-9 vs FD −2.530e-9); and
  `steady_state(0.8)` returns c = 1.000, θ = 0.867 — essentially zero conversion,
  an operating point the simulation (θ ∈ [1.4, 1.6]) never visits.
- **M13** `README:309` — `doi:10.1109/TAC.1985.1104058` resolves to *"Adaptive
  stabilization of a discrete linear system with an unknown high-frequency
  gain"*, IEEE TAC 30(8), 798–799. The correct DOI for Rohrs et al. is
  **10.1109/TAC.1985.1104070**.
- **M14** `experiment.py:47-60` — H and σ_n enter only via `epsilon_optimize`,
  gated on `level >= 2`. Verified: for L1, `max |D(H=5,σ=0.0) − D(H=100,σ=0.1)|`
  at matched seeds = **0.0**. 108 of 216 "configurations" are 9× replicates.

---

## MINOR

| # | Location | Finding |
|---|---|---|
| m1 | `README:16` | "future work (§6)" — the paper's Future Work is §5; §6 is "Derivation of the results". |
| m2 | `README:201` | "7 candidates for L2, 13 for L3" — actual counts are 8 and 14. |
| m3 | `README:91` | "10 modification steps" — 9 occur. |
| m4 | `tests/test_rohrs_instability.py:62` | `assert max_y > 10.0` passes on a bounded response: max\|y\| = 24.80, final-10% \|y\| = 24.70 — a saturation-limited limit cycle. θ₂ → −5.7×10⁵ is not asserted. |
| m5 | `tests/test_epsilon_zero.py:45` | ε=0 test: modifications *are* accepted (75% non-`no_change`), so not vacuous on that axis. But ε=0 holds **by construction** (`sigma_n=0, delta_m=0` ⇒ noisy and exact rollouts call bit-identical plant objects); all 10 seeds give one trajectory; and `max_D < 2e-2` exists solely to accommodate the phantom `mod_step=5` row (real steps 1–4 give D ∈ [−3.92, −2.50]; step 5 gives +0.014). |
| m6 | `tests/test_value_bounds.py` | Tautological: `exp(−(w_e·e² + w_u·du²))` with w ≥ 0 is in (0,1] identically. Never run on the sweep's own trajectories. |
| m7 | `experiment.py:63` | `python experiment.py > log.txt` crashes on Windows: `UnicodeEncodeError: 'charmap' codec can't encode 'γ'`. |
| m8 | `agent.py:168-169` | `np.repeat` against states built with `np.tile` — inconsistent interleaving. Inert only because the reference is seed-independent. |
| m9 | `plants.py:50-59` | `sat_viol` evaluated *after* rate-limiting. |
| m10 | `README:295` | "(2021)" — arXiv:2011.06275 submitted Nov 2020 (v2 Jan 2021). |
| m11 | `levels.py:189-193` | Dead code: `p_eff = params; if frozen: p_eff = params`. |
| m12 | — | `no_change` selected **0/1800** times on both plants. There is no accept/reject path. |

---

## Verdicts on the 20 requested checks

| # | Item | Verdict |
|---|---|---|
| 1 | ε=0 test vacuous? | **Partly clean.** Mods are accepted; arms are distinct objects. But ε=0 by construction, and the D tolerance covers a phantom row. (m5) |
| 2 | Paired? | **CLEAN.** Both arms consume `plant_rng` identically; `max\|Δreward\|` for t<100 = 0.0. |
| 3 | ε measured or assumed? | **Measured, not a stub.** Real exact rollout on the noise-free plant, called every step. But over horizon H, maxed over 8/14 candidates. (C4) |
| 4 | Curse gap independent? | **Partly.** Different rollouts, so not the same draw reused; but conflates model bias with selection bias. |
| 5 | r ∈ [0,1]; same horizon? | **Half.** r ∈ [0,1] holds identically. Horizons are NOT the same. (C4) |
| 6 | Theorem 7 transcribed? | **CLEAN.** Verified against arXiv TeX (`policy1.tex`): `min(ε/γ^(t−1), 1/(1−γ))`, γ^(t−1) in the denominator, u → [0,1]. `metrics.py:68` matches. Corollary 15 and Everitt Thm 16 both check out. |
| 7 | Margins include unmodeled block? | **CLEAN on this axis** (`episode_plant.unmodeled=True`, 3rd-order loop) — but voided by C1 and C2. |
| 8 | Plant TF exact? | **CLEAN.** Exactly 2/(s+1)·229/(s²+30s+229), y = x2. Matches the published counterexample. |
| 9 | dt too large? | **CLEAN.** \|λ\| = 15.13, \|λ\|·dt = 0.151 ≪ RK4's ~2.8. k\* = −1 unchanged at dt = 0.01/0.005/0.0025. No numerical artefact. |
| 10 | Detector distinguishes divergence? | **FAIL — CRITICAL.** (C3) |
| 11 | CSTR linearisation recomputed? | **FAIL.** Fixed, wrong point, sign error. (M12) |
| 12 | Symmetric termination? | **CLEAN.** No level terminates early. Caveat: M5 is an asymmetric rule, currently inert. |
| 13 | Seed count honest? | **CLEAN on count** (200 declared/used, 2000 rows/config). Not clean on independence. (M14) |
| 14 | Fit method reported? | **Partly.** Method, R², param error printed. Residuals computed, never reported; model class invalid; no other headline number carries uncertainty. (M7) |
| 15 | results/ consistent? | **FAIL.** 64,518 rows disagree. (C6) |
| 16 | Determinism? | **Clean within a run.** No unseeded RNG, no set/dict order dependence, no parallelism. **Fails across seed counts.** (M10) |
| 17 | L3 ≠ L2? | **FAIL — CRITICAL.** 0/1800 L3-exclusive selections on Rohrs. (C5) |
| 18 | README follows from results/? | **FAIL.** 17 overstating sentences listed below. |
| 19 | Citations verified? | Tětek ✓, Everitt ✓, Rohrs bibliographic data ✓ but **DOI wrong** (M13). Bequette 1998 real but unverifiable at the given granularity. §6 vs §5 (m1). |
| 20 | Limitations honest? | **Mixed — partly decorative.** (d) actively conceals C1/C2; (g) promotes the C4 excuse to a limitation; (b) has wrong counts. Ten load-bearing items missing. |

---

## Item 18 — README sentences that overstate

1. **L96-136** (Q1 heading "does deterioration respect Theorem 7?") — the
   experiment cannot answer this. (C4)
2. **L130-136** "Root cause: … trajectory divergence … not an implementation
   error" — unsupported and, on the evidence, wrong. Hedged overstatement.
3. **L135-136** "the ε = 0 test confirms the epsilon-optimizer is correctly
   implemented" — that test cannot distinguish a correct ε-optimizer from any
   argmax. (m5)
4. **L109-110** "L2 and L3 show positive mean deterioration" — per plant,
   Rohrs L2 = −0.1305, L3 = −0.1316 (self-modification *improves* value on the
   primary plant); the positive aggregate is entirely CSTR.
5. **L107-109** "L1's negative mean … (the Rohrs phenomenon hurts some seeds)" —
   the negative mean is CSTR (−0.504); L1 on Rohrs is **+0.0099**.
6. **L121-122** "The Rohrs plant induces larger ε … because the unmodeled
   second-order dynamics cause more model mismatch" — the two mismatches are
   structurally different; the stated mechanism is not the implemented one. (M11)
7. **L148-151** "confirming a heavy-tailed distribution" — median from one
   config, max from all configs at a different γ. (M7)
8. **L152-156** "consistent with their observation that the worst case is
   attainable but not typical" — the paper makes no such empirical observation.
9. **L163-166** "no destabilisation was observed … This is a clean negative" —
   a tautology, not a negative result. (C3)
10. **L175-178** "21–24% of accepted self-modifications trade stability for
    performance … the agent believes it is improving" — fictional margins (C1,
    C2), off-by-one flag (M3), true-value not in-sample (M4), wrong denominator
    (C6).
11. **L178** "the safety margin is being consumed" — no evidence. Correct
    margins are GM 11–26, PM 47–78° throughout.
12. **L186-191** "L3 does not make deterioration meaningfully worse than L2" —
    L3's candidates are never selected on Rohrs (C5). On CSTR the paired
    difference is −0.0092, t = −11.4 (L3 significantly *better*) — and that
    t-statistic is itself invalid.
13. **L188-190** "the larger candidate set provides more good options" — the
    larger set is never exercised on Rohrs.
14. **L243** "All five tests pass" — two cannot fail, one passes on a bounded
    transient, one is scoped to a palindromic denominator hiding C1, one is
    scoped to fixed S.
15. **L245-247** "phase margin … matches … to < 0.01°" — true, but validates the
    PI path, which no L1–L3 run uses. Docstring claims 1e-6; test asserts 1e-2.
16. **L275** "432,000 rows" — correct count, stale contents. (C6)
17. **L234-239** "Reducing seeds to 100 … with minimal loss of statistical
    power" — scaling is roughly linear, and changing S changes every per-seed
    realisation. (M10)

---

## Item 20 — limitations that were missing entirely

1. The instability experiment has no reachable positive outcome. (C3)
2. ε and D are measured over different horizons. (C4)
3. ε_emp is a max over a finite hand-picked set, not the sup over policies the
   theorem requires.
4. The baseline is π₀, not the π₁ the theorem compares against. (M1)
5. γ is per-step, so the value horizon is shorter than the modification period
   at 2 of 3 γ values. (M8)
6. The Rohrs plant is run outside the regime where it is a counterexample. (M9)
7. Seeds differ only by additive derivative noise; the reference is identical
   across all seeds; there is no disturbance signal.
8. Noise is re-drawn at each RK4 stage and added to the derivative — not a
   consistent SDE discretisation, so σ_n does not have its documented magnitude.
9. Per-seed results are not reproducible across different `--seeds`. (M10)
10. No headline number carries a confidence interval; the 54,000-row
    t-statistics ignore clustering by seed, config, and mod-step.

---

## What a control theorist would ask in the first five minutes

**1. "Show me the Bode plot of the loop you're actually simulating, and tell me
why your gain margin is 0.15 when the simulation is manifestly stable."**
**Does the repo survive? No.** GM = 0.149 and PM = −32.3° assert an unstable
loop; the trajectories are bounded. That contradiction is visible in the CSV
without running anything and traces to the `[::-1]` at `metrics.py:81`.

**2. "Your actuator saturates at ±10 on a plant with DC gain 2 and no RHP poles.
What trajectory were you expecting to reach 10⁶?"**
**No.** |y| ≤ 20 for all admissible inputs. There is no controller — in or out
of the candidate set — that could trip the detector.

**3. "You picked the Rohrs counterexample. Why is your adaptation gain 1.0 and
your excitation 5 rad/s?"**
**No.** At γ_a = 1.0 the loop is quiescent at both 5 and 16.1 rad/s. The Rohrs
plant is present as scenery.

**A fourth, by minute six:** *"Your L3 arm never once chose an L3 action on your
primary plant. What is the L3-vs-L2 comparison comparing?"* — 0/1800.

---

## Bottom line

The physics core is sound: the plant transfer function is exact, the integrator
step is comfortably fine, pairing is genuine, the Theorem 7 transcription is
correct symbol-for-symbol, and the sweep reproduces V/D/margins bit-identically.
What sits on top of it — Q3 in its entirety, the Theorem 7 comparison, the
L3-vs-L2 headline, and the tail claim in Q2 — does not support the claims made.
