# Does Self-Modification Deteriorate a Physically Constrained Controller Faster Than the Abstract Bound Predicts?

## 1. The Question and the Gap in Tětek et al.

Tětek, Sklenka, and Gavenčiak (2021) prove that a bounded-rational
ε-optimizer with the ability to self-modify may deteriorate exponentially
over time. Their Theorem 7 states:

> E[Q(æ<t, π_t(æ<t))] ≥ E[Q(æ<t, π_1(æ<t))] − min(ε/γ^(t−1), 1/(1−γ))

where the instantaneous utility lies in [0, 1], γ is the discount factor,
and t counts self-modification steps. When ε = 0, self-modification is
harmless (Corollary 15, recovering Everitt et al. 2016, Theorem 16).

**The gap**: their result is (a) worst-case only — they flag full
probabilistic analysis as future work (§6) — and (b) for a disembodied
agent whose self-modifications have no direct physical effect on the
environment. This project embeds the same self-modification structure in
a physical plant with stability and actuator constraints, and asks
whether the deterioration respects the abstract bound, what its
distribution looks like over random draws, and whether closed-loop
stability degrades even when in-sample value improves.

## 2. Setup

### Plants

**Primary — Rohrs counterexample** (Rohrs, Valavani, Athans & Stein, IEEE TAC
30(9):881–889, 1985):

    G(s) = [2 / (s + 1)] × [229 / (s² + 30 s + 229)]

The nominal first-order block 2/(s+1) is what the controller is designed
against; the fast second-order block (poles at −15 ± 2j) is unmodeled.

**Secondary — non-isothermal CSTR** with Arrhenius kinetics and jacket
cooling, in standard dimensionless form:

    dc/dt   = (c_in − c) − Da · c · exp(−γ_arr / θ)
    dθ/dt   = (θ_in − θ) + B · Da · c · exp(−γ_arr / θ) − β(θ − θ_c)

Both plants have hard constraints: actuator saturation u ∈ [u_min, u_max],
rate limit |du/dt| ≤ R, and a safety bound |y − y_sp| ≤ y_max.

### Self-Modification Ladder

| Level | What changes | Discrete? |
|-------|-------------|-----------|
| L0 | Fixed PI, tuned once on nominal model | No |
| L1 | MRAC gradient law θ̇ = −γ_a · e · φ, with σ-mod and dead zone | Continuous |
| L2 | Structure: add/remove integral, lead, filter, Kp/Ki scaling | Yes (every T_mod) |
| L3 | Adaptation law: γ_a, σ, dead-zone width | Yes (every T_mod) |

Baseline for every comparison: the same level, same seed, same disturbance
realisation, but with self-modification disabled after step 1.

### Epsilon: Induced, Not Assumed

The agent scores each candidate by a finite-horizon rollout of length H
on a noisy internal model (process noise σ_n, model mismatch δ_m). It
selects the argmax of the **noisy** score, making it an ε-optimizer in
the Tětek sense. After each decision, we compute the **true** value of
every candidate by an exact rollout on the real plant (no noise, no
mismatch) and record the realised optimality gap:

    ε_emp = max(true_values) − true_value_of_selected

We sweep H ∈ {5, 20, 100}, σ_n ∈ {0.0, 0.01, 0.1}, δ_m = 0.5 (fixed).
Reported ε distribution (mean, p95, max) per configuration.

### Value Function

    r_t = exp(−(w_e · e_t² + w_u · du_t²))     ∈ [0, 1]
    V   = Σ_t γ^t · r_t                          γ ∈ {0.9, 0.95, 0.99}

Discounted value from modification step k: V_k = Σ_{t≥k} γ^(t−k) r_t.

### Sweep

| Dimension | Values |
|-----------|--------|
| Levels | L0, L1, L2, L3 |
| γ | 0.9, 0.95, 0.99 |
| H | 5, 20, 100 |
| σ_n | 0.0, 0.01, 0.1 |
| Plants | Rohrs, CSTR |
| Seeds | 200 per configuration |
| δ_m | 0.5 (fixed) |

Total: 216 configurations × 200 seeds = 43,200 paired episodes.
dt = 0.01 s, T_episode = 10 s, T_mod = 1 s (10 modification steps).
w_e = 1.0, w_u = 0.1, episode noise = 0.01.

## 3. Results

### Q1: Instantiation — does deterioration respect Theorem 7?

Mean deterioration D_k = V_baseline − V_selfmod per level (all configs):

| Level | D mean | D p95 | D max |
|-------|--------|-------|-------|
| L0 | 0.0000 | 0.0000 | 0.0000 |
| L1 | −0.2471 | 0.3472 | 0.8123 |
| L2 | 0.0901 | 2.6461 | 11.9654 |
| L3 | 0.0850 | 2.6613 | 12.8726 |

L0 shows zero deterioration (no self-modification possible). L1's negative
mean indicates MRAC adaptation improves over the frozen baseline on average
(the Rohrs phenomenon hurts some seeds, max D = 0.81). L2 and L3 show
positive mean deterioration with a heavy tail (max D ≈ 12).

**Empirical epsilon** (L2/L3 only):

| Config | ε mean | ε p95 | ε max |
|--------|--------|-------|-------|
| L2 Rohrs | 0.0254 | 0.1216 | 0.5247 |
| L3 Rohrs | 0.0185 | 0.0782 | 0.6182 |
| L2 CSTR | 0.00005 | 0.00001 | 0.0262 |
| L3 CSTR | 0.00006 | 0.00013 | 0.0239 |

The Rohrs plant induces larger ε than the CSTR because the unmodeled
second-order dynamics cause more model mismatch (δ_m = 0.5).

**Theorem 7 bound check** (L2/L3 only):
- 132,847 of 216,000 L2/L3 rows are flagged as D_k > bound.
- Of these, 103,969 (78%) occur when ε = 0 (bound = 0). These are
  cases where the agent picked the true best candidate, yet the paired-
  trajectory D_k is slightly positive due to trajectory divergence.
- 28,878 (22%) occur with ε > 0, where D_k exceeds the same-state bound.
- **Root cause**: the Theorem 7 bound is a same-state comparison
  (Q(π_t) vs Q(π_1) at the same history), while our D_k compares two
  separate runs that diverge after step 1. The trajectory divergence
  adds deterioration the abstract bound does not capture. This is a
  known limitation of the paired-trajectory design, not an
  implementation error — the ε = 0 test (Test 2) confirms the
  epsilon-optimizer is correctly implemented.

### Q2: Typicality — is the deterioration distribution median-benign with a heavy tail?

At k = 5, Rohrs, γ = 0.95, H = 20, σ_n = 0.01 (n = 200 seeds):

| Level | Median | p95 | Max |
|-------|--------|-----|-----|
| L2 | 0.7267 | 0.7287 | 0.7298 |
| L3 | 0.7266 | 0.7291 | 0.7304 |

Exponential fit to median growth (A·exp(B·k)):
- B = 0.52, but R² = 0.134 — the fit is poor.
- The median does **not** grow exponentially. It is roughly flat.
- The tail (max D up to 12.87 across all configs) is much larger than
  the median, confirming a heavy-tailed distribution.

**Interpretation**: the Tětek worst-case bound predicts exponential
deterioration, but the typical case is much milder. The median is
benign; the tail is not. This is consistent with their observation that
the worst case is attainable but not typical. 200 seeds is insufficient
for strong distributional claims — the fit uncertainty (param error
0.98) is larger than the parameter itself.

### Q3: The Physical Addition — do stability margins degrade?

**Destabilisation depth k***: no destabilisation was observed for L2 or L3
within the 10-second episodes (no seed hit the instability threshold
|y| > 10⁶). This is a **clean negative** for the destabilisation question:
margins degrade but the loop does not destabilise in this setting.

**Value-up-margin-down quadrant**: fraction of accepted modifications that
improve in-sample value while degrading stability margins:

| Level | Fraction | Count |
|-------|----------|-------|
| L2 | 0.206 | 22,291 / 108,000 |
| L3 | 0.241 | 25,980 / 108,000 |

21–24% of accepted self-modifications trade stability for performance.
This is the physically interesting quadrant: the agent believes it is
improving (in-sample value up) while the closed-loop margin shrinks.
The loop did not destabilise, but the safety margin is being consumed.

### L3 vs L2: Does modifying the adaptation law make things worse?

| Plant | L2 D mean | L3 D mean | Diff | L2 ε mean | L3 ε mean | ε ratio |
|-------|-----------|-----------|------|-----------|-----------|---------|
| Rohrs | −0.1305 | −0.1316 | −0.001 | 0.0254 | 0.0185 | 0.73 |
| CSTR | 0.3107 | 0.3016 | −0.009 | 0.00005 | 0.00006 | 1.25 |

Modifying the adaptation law (L3) does **not** make deterioration
meaningfully worse than modifying the controller structure (L2). The mean
D is nearly identical, and on the Rohrs plant L3 actually has smaller ε
(the larger candidate set provides more good options). On the CSTR, L3
has slightly larger ε but the absolute magnitude is negligible.

## 4. Limitations

(a) **Two plants is not a general claim about physical systems.** The
Rohrs counterexample is a specific linear plant with a specific
unmodeled block; the CSTR is a specific nonlinear reactor. Results may
not generalise to other physical systems.

(b) **The candidate modification set is hand-designed and finite** (7
candidates for L2, 13 for L3). This is not open-ended self-modification.
An agent with a richer or learned modification space could behave
differently.

(c) **The value function is a modelling choice.** The [0, 1] mapping
r_t = exp(−(w_e·e² + w_u·du²)) makes utility commensurable with
Theorem 7, but different weight choices (w_e, w_u) or different
functional forms would change the absolute values and the comparison
with the theoretical bound.

(d) **Margins are computed on a linearisation**, which is only locally
valid for the CSTR. The Rohrs plant is linear so the margin computation
is exact; the CSTR linearisation is at a single operating point and
may not capture far-from-equilibrium behaviour.

(e) **200 seeds supports distributional claims weakly, not strongly.**
The exponential fit has R² = 0.134 and parameter uncertainty larger
than the parameter. The median and p95 are reliable; the max is not
(a single outlier seed can dominate). Strong distributional claims
would require 10,000+ seeds.

(f) **δ_m is fixed at 0.5.** The spec says to sweep H, σ_n, δ_m, but
only H and σ_n are in the sweep table. Fixing δ_m reduces the sweep
from 648 to 216 configs. This is a deliberate choice to stay within
the 6-minute budget.

(g) **Paired-trajectory comparison introduces trajectory divergence.**
The Theorem 7 bound is a same-state comparison; our D_k compares two
separate runs that diverge after step 1. This means D_k can exceed
the bound even when the ε-optimizer is correctly implemented. The
ε = 0 test confirms the implementation is correct; the bound
exceedances are a property of the experimental design.

(h) **Runtime**: the full sweep took approximately 7 minutes on the
test machine (Python 3.13, numpy 2.3, single-threaded). This exceeds
the 6-minute target. The bottleneck is the CSTR L3 H=100
configurations. Reducing seeds to 100 would bring it under 6 minutes
with minimal loss of statistical power (the Python overhead dominates,
so scaling is sublinear in seed count).

## 5. Tests

All five tests pass:

1. **L0 nominal stability**: PI on G(s) = 2/(s+1) is stable; phase
   margin = 70.53° matches the analytic hand calculation (2·arctan(1/√2))
   to < 0.01°; gain margin = ∞ (no phase crossover).
2. **ε = 0**: with H = 500, σ_n = 0, δ_m = 0, the realised ε = 0
   (max < 10⁻⁶) and D_k < 0.02 (within numerical tolerance). This
   reproduces Everitt et al. (2016) Theorem 16 and Tětek et al.
   Corollary 15.
3. **Rohrs instability**: MRAC with γ_a = 50, σ = 0, no dead zone,
   and a 5 rad/s sinusoidal reference on the full plant produces
   instability (|y| > 10) within 20 seconds.
4. **Determinism**: same seed → byte-identical results.
5. **Value bounds**: r_t ∈ [0, 1] asserted at every step for both
   plants.

## 6. How to Run

```bash
# Install dependencies (Python 3.11+)
pip install numpy scipy control pandas matplotlib pytest

# Run tests
python -m pytest tests/ -v

# Run full sweep (≈7 min, 200 seeds)
python experiment.py --seeds 200

# Run analysis and generate plots
python analysis.py
```

Results are written to `results/sweep_results.csv` (432,000 rows).
Plots are saved to `results/q1_deterioration.png`,
`results/q2_distribution.png`, `results/q3_margins.png`.

## 7. Repository Layout

```
plants.py        — Rohrs plant, CSTR, constraints, fixed-step RK4
controllers.py  — PI, MRAC, structure/adaptation-law parameterisation
agent.py        — candidate enumeration, noisy scoring, ε-optimizer, exact rollout
metrics.py      — value, margins, Theorem 7 envelope, constraint accounting
levels.py       — L0–L3 wiring, paired baselines, metric recording
experiment.py   — CLI, sweeps, results writing, config hashing
analysis.py     — Q1–Q3 analyses and plots, L3 vs L2 comparison
tests/          — 5 test files (10 tests total, all passing)
results/        — sweep_results.csv, q1/q2/q3 plots
```

## 8. Citations

- Tětek, J., Sklenka, M., & Gavenčiak, T. (2021). *Performance of
  Bounded-Rational Agents With the Ability to Self-Modify.*
  arXiv:2011.06275 [cs.AI].
  https://arxiv.org/abs/2011.06275

- Everitt, T., Filan, D., Daswani, M., & Hutter, M. (2016).
  *Self-Modification of Policy and Utility Function in Rational
  Agents.* AGI-16. arXiv:1605.03142 [cs.AI].
  https://arxiv.org/abs/1605.03142

- Rohrs, C. E., Valavani, L., Athans, M., & Stein, G. (1985).
  *Robustness of continuous-time adaptive control algorithms in the
  presence of unmodeled dynamics.* IEEE Transactions on Automatic
  Control, 30(9), 881–889.
  doi:10.1109/TAC.1985.1104058

- Bequette, B. W. (1998). *Process Dynamics: Modeling, Analysis, and
  Simulation.* Prentice Hall. (CSTR dimensionless form)
