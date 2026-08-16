# Does self-modification consume stability margin, and does it drive parameter divergence?

A physically constrained instantiation of the self-modification structure from
Tětek, Sklenka & Gavenčiak (2020), run on the Rohrs adaptive-control
counterexample.

**This README describes a repository that has been through an adversarial
audit.** Three of its original headline claims were withdrawn rather than
repaired; see [Removed claims](#5-removed-claims). The defects that remain
unfixed are named in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). The audit itself is
[AUDIT_REPORT.md](AUDIT_REPORT.md). Read all three before citing anything here.

---

## 1. The question

Tětek et al. prove that a bounded-rational ε-optimizer with the ability to
self-modify may deteriorate exponentially. Their Theorem 7 states:

> E[Q(æ<t, π_t(æ<t))] ≥ E[Q(æ<t, π_1(æ<t))] − min(ε/γ^(t−1), 1/(1−γ))

with instantaneous utility in [0, 1], γ the discount factor, and t counting
self-modification steps. Setting ε = 0 recovers Everitt et al. (2016)
Theorem 16 (their Corollary 15).

Their result is worst-case, and it is for a disembodied agent whose
self-modifications have no direct physical effect. The original intent of this
repository was to test the bound in a physical setting. **That test does not
survive the audit** (see [Removed claims](#5-removed-claims)). What survives is
a narrower, purely empirical question that the corrected code can actually
answer:

> With correct stability margins, and the Rohrs plant operated in the regime
> where it is genuinely a counterexample, does self-modification consume
> stability margin, and does it ever drive parameter divergence that the frozen
> baseline does not show?

No claim is made here about Theorem 7 being respected or violated.

---

## 2. Setup

### Plant

Rohrs, Valavani, Athans & Stein (1985) counterexample:

    G(s) = [2 / (s + 1)] × [229 / (s² + 30 s + 229)]

The first-order block is the design model; the second-order block (poles at
−15 ± 2j, natural frequency 15.13 rad/s) is unmodeled. Verified against the
published plant, and the implementation verified against `tf_full()` to
5.7 × 10⁻⁷ on a step response.

A non-isothermal CSTR is also simulated. **No margin-based claim is made for
it**: its linearisation is fixed at an operating point the simulation never
visits and carries a sign error (KNOWN_ISSUES M12). Its divergence and
deterioration results are reported, since both are time-domain quantities that
do not touch the linearisation.

Constraints: `u ∈ [−10, 10]`, rate limit `|du/dt| ≤ 100`, safety bound
`|y − y_sp| ≤ 5`.

### Self-modification ladder

| Level | What changes | Candidates |
|-------|-------------|-----------|
| L0 | Fixed PI, tuned once on the nominal model | 1 (`no_change`) |
| L1 | MRAC gradient law with σ-mod and dead zone, continuous | 1 (`no_change`) |
| L2 | Structure: add/remove integral, lead, filter; scale Kp, Ki | **8** |
| L3 | L2 plus the adaptation law: γ_a, σ, dead-zone width | **14** |

Candidate counts include `no_change`, which is scored like any other candidate.
Two of L3's six exclusive candidates (`deadzone_up`, `deadzone_down`) are
structural no-ops because `dead_zone` initialises to 0 — an unfixed defect
(KNOWN_ISSUES, C5 residual).

Baseline for every comparison: same level, same seed, same disturbance
realisation, self-modification never applied.

> **The contrast is modify-vs-never-modify, not π_t vs π_1.** Tětek et al.
> Theorem 7 compares the policy after *t* self-modifications against the policy
> after **one**. The baseline here performs **zero** modifications, so every
> D_k and every divergence comparison in this repository contrasts a
> self-modifying agent against a *never*-modifying one. It is π_t vs π₀. This
> is an unfixed defect (KNOWN_ISSUES M1), not a modelling choice, and it is one
> of the reasons the Theorem 7 comparison was withdrawn (§5.1). Read every
> "baseline" in §3 as "never modifies".

### Operating regime

The audit found the original sweep ran at γ_a = 1.0 with a 5 rad/s reference —
a quiescent regime where max |y| = 1.43 and nothing adaptive happens. γ_a and ω
are now swept:

| Dimension | Values |
|-----------|--------|
| Adaptation gain γ_a | 1, 10, 50 |
| Excitation ω (Rohrs) | 5.0, 16.1 rad/s |

16.1 rad/s is the canonical Rohrs excitation for the −15 ± 2j block.

### Value and epsilon

    r_t = exp(−(w_e · e_t² + w_u · du_t²))   ∈ (0, 1]
    V_k = Σ_{t≥k} γ^(t−k) r_t

The agent scores candidates by an H-step rollout on a noisy internal model
(process noise σ_n, model mismatch δ_m) and selects the argmax of the **noisy**
score. After each decision, an exact rollout of every candidate on the real
plant gives

    ε_emp = max(true_values) − true_value_of_selected

**ε_emp is not the ε of Theorem 7.** The theorem's ε is a sup over all policies
on the infinite-horizon Q; ε_emp is a max over 8 or 14 hand-designed candidates
on an H-step rollout. It is reported descriptively and is not compared against
any bound.

### Sweep

| Dimension | Values |
|-----------|--------|
| Levels | L0, L1, L2, L3 |
| γ | 0.9, 0.95, 0.99 |
| H | 5, 20, 100 |
| σ_n | 0.0, 0.01, 0.1 |
| γ_a | 1, 10, 50 |
| ω | 5.0, 16.1 (Rohrs only) |
| Plants | Rohrs, CSTR |
| Seeds | 200 per configuration |
| δ_m | 0.5 (fixed) |

`dt = 0.01 s`, `T_episode = 10 s`, `T_mod = 1 s`, giving **9** modification
steps per episode (k = 1…9 at t = 100…900; the tenth was a phantom row and has
been removed). `w_e = 1.0`, `w_u = 0.1`, episode noise 0.01.

H and σ_n have **no effect** on L0/L1, so half the configuration count is
replicates rather than distinct conditions (KNOWN_ISSUES M14).

---

## 3. Results

All numbers below come from `results/sweep_results.csv` (972 configurations ×
200 seeds × 9 modification steps = 1,749,600 rows, 1927 s single-threaded).
Regenerate with `PYTHONIOENCODING=utf-8 python experiment.py --seeds 200`
then `python analysis.py`.

**Read the margin numbers as new, not as corrections to the old ones.** Every
pre-audit margin was computed on a reversed polynomial and is unrelated to
these.

### 3.1 Does self-modification drive parameter divergence? Yes — but not because self-modification is inherently destabilising.

Fraction of seeds whose loop diverges, Rohrs, pooled across γ / H / σ_n
(divergence is a property of the loop, not of the discount factor).
n = 5400 seed-configurations per cell. Every Rohrs divergence is `param_drift`;
none is a safety-bound breach.

| Level | γ_a=1, ω=5 | γ_a=1, ω=16.1 | γ_a=10, ω=5 | γ_a=10, ω=16.1 | γ_a=50, ω=5 | γ_a=50, ω=16.1 |
|-------|---|---|---|---|---|---|
| L0 (no adaptation) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 (adapts, no self-mod) | 0.000 | 0.000 | **0.000** | **0.000** | 1.000 | 1.000 |
| L2 (structure self-mod) | 0.000 | 0.000 | **0.223** | **0.329** | 1.000 | 1.000 |
| L3 (+ adaptation-law self-mod) | 0.000 | 0.048 | **0.889** | **0.667** | 1.000 | 1.000 |

The γ_a = 50 column is a **divergent baseline**: L1 diverges there too, so
self-modification is not the cause. The γ_a = 1 column is quiescent. The
load-bearing column is **γ_a = 10**, where the frozen adaptive controller (L1)
is stable in 10,800/10,800 seed-configurations (5400 per ω) while
self-modifying agents are not.

> **This does not mean self-modification is inherently destabilising.** See
> [§3.1c](#31c-the-mechanism-the-agent-walks-across-a-boundary-it-cannot-see),
> which reports a control that rules that reading out. The agent moves γ_a to a
> value that is unstable *for any controller*, self-modifying or not.

Median destabilisation depth k\* at γ_a = 10: **5** for L2 (min 4, max 6),
**4–6** for L3 (min 3, max 9). Divergence takes several modification steps to
build; it is not an immediate consequence of the first modification.

**Paired causal control.** The table above reports the self-modifying arm. Run
against its own paired baseline (same seed, same noise realisation, same γ_a),
200 seeds, γ = 0.95, σ_n = 0.01, γ_a = 10:

| Level | H | ω | self-modifying | paired baseline | final γ_a under self-mod |
|-------|---|---|---|---|---|
| L1 | 100 | 5.0 | 0/200 | 0/200 | 10 → 10 |
| L2 | 100 | 5.0 | 0/200 | 0/200 | 10 → 10 |
| L3 | 100 | 5.0 | **200/200** | **0/200** | 10 → **20–40** |
| L3 | 100 | 16.1 | **200/200** | **0/200** | 10 → **20–40** |
| L3 | 20 | 5.0 | **200/200** | **0/200** | 10 → **20–80** |

L3 repeatedly selects `gamma_up`, walking the adaptation gain out of the stable
regime and into parameter divergence. The baseline, holding γ_a = 10 with an
identical noise realisation, never diverges.

**This is the one place L2 and L3 can be compared.** Broken out by rollout
horizon at γ_a = 10 (fraction of seeds diverging):

| Level | H = 5 | H = 20 | H = 100 |
|-------|-------|--------|---------|
| L1 | 0.000 | 0.000 | 0.000 |
| L2 | 0.66–1.00 | **0.000** | **0.000** |
| L3 | 0.96–1.00 | 0.02–1.00 | **1.000** |

L2 destabilises only when the agent is myopic (H = 5) — give it a longer
rollout and structure-only self-modification is safe. L3 destabilises at
**every** horizon, including H = 100 where it is 1.000 across all γ and both ω
while L2 is 0.000. That contrast is qualitative and all-or-nothing, so the
unpaired-noise confound between L2 and L3 (KNOWN_ISSUES, C5 residual) cannot
manufacture it. A longer planning horizon does not protect an agent that can
rewrite its own adaptation law.

### 3.1c The mechanism: the agent walks across a boundary it cannot see

γ_a is both a swept configuration dimension and a parameter L3 can modify. That
makes the §3.1 comparison ambiguous, so it was tested directly: **run a
non-self-modifying controller started at each γ_a and see where it diverges on
its own.** L1 with adaptation live and self-modification never applied, Rohrs,
200 seeds, γ = 0.95, H = 20, σ_n = 0.01:

| γ_a | 10 | 11 | 12 | 13 | 14 | **15** | 16 | **18** | 20 | 30 | 50 |
|-----|----|----|----|----|----|----|----|----|----|----|----|
| ω = 5.0 | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 | **200/200** | 200/200 | 200/200 | 200/200 |
| ω = 16.1 | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 | **200/200** | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 |

**There is a hard stability boundary in γ_a at roughly 14–15 (ω = 16.1) and
16–18 (ω = 5.0), independent of self-modification.** L3 walks γ_a from 10 to
20–80 — well past it.

So the correct reading of §3.1 is **not** "self-modification destabilises the
loop". It is:

> The agent's objective contains tracking error and control effort and
> **contains no stability term**. γ_a is a free parameter it can raise. Raising
> γ_a improves short-horizon tracking, so the agent raises it, and keeps raising
> it, straight across a stability boundary its objective cannot represent. The
> failure is an **objective-specification failure**, not an emergent property of
> self-modification.

The distinction matters for what generalises. Nothing here shows that an agent
which *could* see the margin would still destroy it. What it shows is that an
agent scoring candidates on a margin-blind objective will spend margin down to
zero, because margin is free from its point of view.

Consistent with this, L2 — which cannot touch γ_a — diverges 0/200 at H = 20 and
H = 100 in the same cell. L2's divergences at H = 5 come from a different route
(a myopic rollout mis-scoring structural changes), not from γ_a.

**The ratchet.** Tracing the accepted modifications on the diverging L3 runs
(γ_a = 10, H = 100, 200 seeds):

| ω | `gamma_up` share of all accepted mods | `gamma_down` ever chosen? | median γ_a by step |
|---|---|---|---|
| 5.0 | **0.667** | yes, 21/600 | 10 → 20 → 40 → 40 |
| 16.1 | 0.333 | yes, 186/1200 | 10 → 10 → 10 → 20 → 20 → 40 → 20 |

`gamma_up` is the single most-selected candidate at ω = 5.0, taken by every seed
at the first two decisions. But it is **not a monotone ratchet**: `gamma_down` is
selected, and 5.3% (ω = 5.0) / 18.6% (ω = 16.1) of step-to-step γ_a changes are
decreases. The drift is strongly upward on net without being one-directional —
the agent does occasionally step back, just never far enough or soon enough.

### 3.1b The CSTR reproduces the finding independently, by a different mechanism

The CSTR's *margin* path is broken (KNOWN_ISSUES M12) and no margin claim is made
for it. Divergence, however, is a pure time-domain criterion that does not touch
the linearisation, so the CSTR result stands on its own.

Fraction of seeds diverging, CSTR, pooled across γ / H / σ_n (n = 5400 per cell):

| Level | γ_a = 1 | γ_a = 10 | γ_a = 50 |
|-------|---------|----------|----------|
| L0 | 0.000 | 0.000 | 0.000 |
| L1 | **0.000** | **0.000** | **0.000** |
| L2 | 0.334 | 0.361 | 0.694 |
| L3 | 0.334 | 0.335 | 0.391 |

This is a **cleaner control than Rohrs**: L1 never diverges on the CSTR at *any*
adaptation gain, including γ_a = 50 where the Rohrs L1 baseline diverges
outright. So there is no divergent-baseline column to discount — every CSTR
divergence is attributable to self-modification.

The mechanism is different. CSTR divergences at γ_a = 1 and 10 are **entirely
sustained safety-bound breach** (`|y − y_sp| > 0.5` held for 0.5 s), with
k\* median 9 and 7. Parameter drift only appears at γ_a = 50, where it accounts
for 1526/3748 (L2) and 1728/2110 (L3) of the divergences.

Two plants, two failure modes — constraint violation on the CSTR, parameter
drift on Rohrs — and in both the frozen adaptive baseline is stable while the
self-modifying agent is not.

### 3.2 Does self-modification consume stability margin? Yes, monotonically.

Median gain margin after modification k, Rohrs, γ = 0.95, H = 20, σ_n = 0.01,
finite values only. These are correct margins on the **full** loop including
the unmodeled second-order block.

| Level | γ_a | ω | GM at k=1 | GM at k=9 | change | PM at k=1 | PM at k=9 |
|-------|-----|---|-----------|-----------|--------|-----------|-----------|
| L2 | 1 | 5.0 | 21.20 | 7.12 | **−14.09** | 66.9° | 25.5° |
| L2 | 1 | 16.1 | 24.79 | 2.69 | **−22.11** | 77.5° | 10.6° |
| L3 | 1 | 5.0 | 21.20 | 5.56 | **−15.64** | 66.9° | 18.4° |
| L3 | 1 | 16.1 | 24.79 | 2.69 | **−22.11** | 77.5° | 10.6° |
| L2 | 10 | 5.0 | 5.62 | 4.12 | −1.50 | | |
| L3 | 10 | 5.0 | 5.62 | 0.03 | **−5.59** | | |

Margin consumption is the clearest and most robust finding here. It happens
**even at γ_a = 1, where nothing ever diverges**: nine modifications take the
gain margin from 24.8 to 2.7 and the phase margin from 77.5° to 10.6° at the
canonical Rohrs excitation. The loop stays stable for the full episode, but by
k = 9 it is one modification away from the stability boundary.

Note the direction of the γ_a = 10 rows: L3 drives the gain margin to 0.03 —
below 1, i.e. nominally unstable — which is the linear-analysis signature of the
parameter divergence reported in §3.1.

**Which modification does the damage cannot be answered from the stored
results**: `chosen`, the selected candidate index, is not a column in
`sweep_results.csv`, so margin change cannot be attributed per candidate without
re-running the sweep. What the stored per-step margins do show is that the drop
is **concentrated, not gradual** — in the headline cell (γ_a = 1, ω = 16.1) a
single modification step, k = 2, accounts for 54.1% of the total 20.21 fall in
gain margin, the top two steps (k = 2 and k = 9) for 74.6%, while steps 3–5
together account for 1.4%; two steps (k = 1 and k = 6) *restore* margin,
by +1.90 and +4.60. Of the total fall, −16.10 occurs within modification steps
and −4.11 accumulates between them as θ₂ adapts, so roughly a fifth of the
margin loss is not attributable to any modification at all.

### 3.3 Does the agent know it is doing this? No.

Fraction of modification decisions where the agent's value went up **and** the
margin went down, at γ = 0.95, H = 20, σ_n = 0.01. "Believed" uses the agent's
own noisy scores — what it acted on. "Realised" uses the exact rollout on the
real plant.

| Level | γ_a | ω | believed | realised | n |
|-------|-----|---|----------|----------|---|
| L2 | 1 | 5.0 | 0.778 | 0.757 | 1800 |
| L2 | 10 | 16.1 | 0.667 | 0.556 | 1800 |
| L3 | 10 | 5.0 | 0.481 | 0.250 | 800 |
| L3 | 10 | 16.1 | 0.669 | 0.558 | 1800 |
| L2 | 50 | 5.0 | 0.890 | **0.000** | 200 |
| L3 | 50 | 5.0 | 0.940 | **0.000** | 200 |

Believed exceeds realised in every row. The gap is widest exactly where the loop
is worst: in the divergent γ_a = 50 regime the agent believes it improved on
~90% of decisions while the exact rollout says it improved on **none of them**.
The agent is not trading stability for performance knowingly — it is losing
stability while believing it is gaining performance.

### 3.3b The believed-vs-realised gap has two confounded explanations

The gap in §3.3 admits two readings that this design cannot separate:

1. **Optimizer's curse** — selection bias from taking the argmax of a noisy
   score. Should shrink toward zero as scoring noise goes to zero.
2. **Model bias** — the agent's internal model is simply wrong (δ_m = 0.5
   replaces the true plant with the nominal first-order design model, discarding
   the unmodeled block entirely). Should persist at zero scoring noise.

From the existing sweep, believed minus realised value-up-and-margin-down,
Rohrs L2/L3 pooled across γ, H and ω:

| σ_n | γ_a = 1 | γ_a = 10 | γ_a = 50 |
|-----|---------|----------|----------|
| 0.0 | 0.138 | 0.156 | 0.066 |
| 0.01 | 0.140 | 0.257 | 0.422 |
| 0.1 | 0.130 | 0.256 | 0.421 |

**The gap does not vanish at σ_n = 0.** At γ_a = 1 it is flat in σ_n (0.138,
0.140, 0.130). At γ_a = 10 and 50 it is smaller at σ_n = 0 but far from zero at
γ_a = 10 and clearly nonzero at γ_a = 50.

**This does not identify the mechanism, and no claim is made that it does.**
σ_n = 0 removes *additive scoring noise* but not model error: at σ_n = 0 the
agent still scores candidates on the wrong plant, and selecting the argmax of a
deterministically biased score is itself a form of selection bias. The two
explanations remain entangled.

**δ_m was never varied.** It is fixed at 0.5 in every one of the 972
configurations, so the model-bias axis was not swept at all and cannot be
separated from the curse on this data. Doing so would require sweeping δ_m —
including δ_m = 0, where the agent's model is the true plant — at σ_n = 0, which
isolates model bias with scoring noise held out. That experiment has not been
run.

### 3.4 Deterioration, reported descriptively

No functional form is fitted and no bound is compared against; see
[Removed claims](#5-removed-claims).

D_k = V_baseline − V_selfmod at γ = 0.95, H = 20, σ_n = 0.01 (median, p05, p95;
n = 1800 per row):

| Plant | Level | γ_a | ω | median | p05 | p95 |
|-------|-------|-----|---|--------|-----|-----|
| Rohrs | L0 | any | any | 0.0000 | 0.0000 | 0.0000 |
| Rohrs | L2 | 1 | 16.1 | 0.3015 | −0.0512 | 0.4439 |
| Rohrs | L2 | 10 | 5.0 | 0.2622 | −4.2354 | 1.7033 |
| Rohrs | L3 | 1 | 16.1 | 0.3016 | −0.0512 | 0.4438 |
| Rohrs | L3 | 10 | 5.0 | **12.7297** | −4.1770 | 17.4998 |
| CSTR | L1 | 50 | – | −10.1930 | −13.1404 | −3.8229 |
| CSTR | L2 | 1 | – | 0.0685 | −0.6020 | 1.5257 |

L0 shows D ≡ 0 exactly, as it must — with no self-modification the two arms are
the same run. That is a wiring check, not a result.

The largest deterioration in the sweep, D ≈ 12.7, is the (L3, γ_a = 10, ω = 5)
cell — the same cell that diverges. Median D_k by step there:
−1.24, −4.18, 6.65, 13.51, 12.73, 17.50, 16.15, 11.86, 13.96. Self-modification
*helps* for the first two steps and then costs an order of magnitude more than
it gained, with the sign flip at k = 3 preceding the median divergence depth
k\* = 4. No growth rate is claimed from nine points.

### 3.5 Realised optimality gap

ε_emp, descriptive only, never differenced against D (they span different
horizons — see Limitations (f)):

| Plant | Level | γ_a | mean | p95 | max | n |
|-------|-------|-----|------|-----|-----|---|
| Rohrs | L2 | 1 | 0.0572 | 0.3885 | 0.883 | 97,200 |
| Rohrs | L3 | 1 | 0.1124 | 0.2613 | 13.03 | 97,200 |
| Rohrs | L3 | 10 | 0.3584 | 1.9448 | 16.69 | 62,824 |
| CSTR | L2 | 1 | 0.000055 | 0.000029 | 0.0265 | 48,406 |

The Rohrs plant induces ε three to four orders of magnitude larger than the
CSTR. L3's ε is larger than L2's, and largest in the divergent regime — the
extra candidates give the agent more ways to be wrong, not fewer. Sample counts
shrink at higher γ_a because diverged seeds stop self-modifying.

### 3.6 What did not happen

- **No Rohrs seed diverged by safety-bound breach.** All 44,032 Rohrs
  divergences are parameter drift, and |y| stays under ~3 throughout. The
  pre-audit `|y| > 10⁶` detector would have reported nothing here — it is blind
  to the only failure mode this plant exhibits. On the CSTR the split is the
  other way: 9,968 safety-bound breaches against 3,254 parameter drifts.
  Neither plant would have registered under the old detector.
- **L0 and L1 never diverge on Rohrs at γ_a = 1 or 10** — 0 out of 43,200
  seed-configurations (21,600 per level). On the CSTR, L0 and L1 never diverge
  at *any* γ_a — 0 out of 32,400. Adaptation alone, at the γ_a values the sweep
  starts from, is not the mechanism. But see §3.1c: adaptation alone at
  γ_a ≥ 18 *is* enough, and that is where the agent takes it.
- **`no_change` is essentially never selected.** There is no accept/reject step;
  the agent modifies at every opportunity. "Accepted modification" here means
  "selected".

---

## 4. Limitations

**(a) Two plants, and only one of them supports the margin claim.** The Rohrs
counterexample is a specific linear plant with a specific unmodeled block. The
CSTR margin path is broken (KNOWN_ISSUES M12) and is excluded from every
margin-based statement.

**(b) The candidate set is hand-designed, finite, and partly inert.** 8 for L2,
14 for L3, including `no_change`. Two L3 candidates are no-ops. This is not
open-ended self-modification.

**(c) The value function is a modelling choice.** The [0, 1] mapping
`r_t = exp(−(w_e·e² + w_u·du²))` makes utility commensurable with Theorem 7's
assumption, but different weights or forms change the absolute values.

**(d) The instability experiment previously had no reachable positive
outcome.** The original detector was `|y| > 10⁶`. Both plants are open-loop
stable with saturating actuators, so `|y| ≤ ~20` (Rohrs) and `|θ| ≤ ~2.4` (CSTR)
for *any* controller — the threshold sat 50,000× outside the reachable set, and
"no destabilisation observed" was a property of the setup, not a result. The
detector is now parameter drift (`|θ₂| > 50`) plus sustained safety-bound
breach, each held for 0.5 s. Divergence results before this change should be
disregarded entirely.

**(d2) The divergence result is an objective-specification failure, not a
property of self-modification.** A non-self-modifying controller started at
γ_a ≥ 18 (ω = 5.0) or γ_a ≥ 15 (ω = 16.1) diverges 200/200 on its own (§3.1c).
The self-modifying agent reaches γ_a = 20–80. It moved to a setting that is
unstable for any controller, because its objective contains no stability term.
Nothing here shows that an agent able to see the margin would still spend it.

**(e) ε_emp is a lower bound on the true ε, of unknown tightness.** It maxes
over the agent's own candidate list, not over policies.

**(f) ε and D are measured over different horizons and must not be
differenced.** ε spans H steps; D spans the episode remainder. At γ = 0.99 and
H = 5, ε cannot exceed 4.90 while D can reach 99.99. This is why the Theorem 7
comparison was removed rather than repaired.

**(g) The baseline is π₀, not π₁ — the contrast is modify-vs-never-modify.**
Theorem 7 compares the policy after t modifications against the policy after
*one*. The L2/L3 baseline here never modifies at all. No result in §3 is a
π_t-vs-π_1 comparison, and none should be read as one (KNOWN_ISSUES M1).

**(h) γ is per-step, so the value horizon is shorter than the modification
period at two of three γ values.** Effective horizon `1/(1−γ)` is 0.1 / 0.2 /
1.0 s at `dt = 0.01`, against `T_mod = 1.0 s`. At γ = 0.9 and 0.95, V_k has
decayed to nothing before the next modification exists, so compounding across
modifications cannot appear in the measured window (KNOWN_ISSUES M8).

**(i) Seeds are weakly independent.** They differ only by additive derivative
noise; the reference is identical across seeds and there is no disturbance
signal. On the CSTR every seed makes the identical candidate choice at every
step. Effective sample size is far below 200.

**(j) Noise injection is not a consistent SDE discretisation.** Noise is added
to the derivative and re-drawn at each of the four RK4 stages, so σ_n and
`ep_noise` do not have the magnitudes their names suggest.

**(k) Per-seed results are not reproducible across seed counts.** Noise is drawn
as one `(S, n)` matrix, so seed 0's realisation depends on `--seeds`
(KNOWN_ISSUES M10). Runs are byte-identical at fixed S.

**(l) No uncertainty is reported on any headline number.** Rows are not
independent — 9 correlated modification steps × 200 weakly independent seeds ×
many configurations — and no clustering correction is applied. All figures are
descriptive summaries of this sweep, not population estimates.

**(m) The optimizer's-curse gap conflates selection bias with model bias.**
`self_deception = in_sample − true_sel` differences a noisy-model rollout
against an exact-plant rollout; no second independent noisy draw separates them.

**(n) One plant's margins rest on a linearisation.** Rohrs is linear so its
margin computation is exact — *now*. Before the audit it was computed on a
reversed polynomial (`np.polyval` was fed decreasing-order coefficients
reversed), which reported GM = 0.149 and PM = −32.3° for a loop whose true
margins are GM ≈ 26 and PM ≈ 60°. Every pre-audit margin number was fiction.

---

## 5. Removed claims

These were in the previous README. They are withdrawn, not restated with new
numbers.

### 5.1 The Theorem 7 bound comparison — removed

Previously: "132,847 of 216,000 L2/L3 rows are flagged as D_k > bound", with the
excess attributed to "trajectory divergence … a known limitation of the
paired-trajectory design, not an implementation error."

**Why removed.** The comparison was not well posed.

1. **ε and D are measured over different horizons.** ε spans H steps; D spans
   the episode remainder. Since r ∈ [0,1], at γ = 0.99, H = 5 the maximum
   attainable ε is 4.90 against a maximum attainable D of 99.99 — a 20×
   mismatch. The flag rate tracked this mismatch directly (0.817 at the worst
   ratio, 0.429 at the best), not "trajectory divergence".
2. **ε_emp is not the theorem's ε.** A max over 14 hand-picked candidates is not
   a sup over policies, so the bound was built from a systematically
   too-small ε.
3. **The baseline is π₀, not π₁** (KNOWN_ISSUES M1).
4. A large share of the flagged rows were phantom `mod_step = 10` rows that
   carried a stale `eps_emp = 0.0` in a stale committed CSV.

The `theorem7_bound` and `bound_violated` columns are gone. Raw `D` is retained
and reported descriptively. The transcription of Theorem 7 itself was verified
correct symbol-by-symbol against the arXiv TeX source and is left in
`metrics.py`, unused.

### 5.2 The heavy-tail and exponential-growth claims — removed

Previously: an exponential fit `A·exp(B·k)` with B = 0.52, R² = 0.134, and
"the tail (max D up to 12.87 across all configs) is much larger than the median,
confirming a heavy-tailed distribution."

**Why removed.** The ten medians being fitted alternated in sign
(−0.22, −0.57, −0.17, −1.18, +0.73, −0.83, −0.42, +0.08, +2.70). `A·exp(B·k)` is
strictly positive, so this was not a poor fit but an invalid model class. The
"tail" compared a median from one configuration against a maximum drawn from all
configurations at a different γ, where values are ~5× larger by construction.
The cited within-config spread was 0.7267 → 0.7298 — a 0.4% range over 200
seeds, which is not a distribution.

D is now reported as median and p05–p95 for one named configuration, with no
fitted model.

### 5.3 The L3-vs-L2 headline — removed

Previously: "Modifying the adaptation law (L3) does **not** make deterioration
meaningfully worse than modifying the controller structure (L2)."

**Why removed.** At γ_a = 1 — the only adaptation gain the original sweep used —
**L3's six exclusive candidates were selected 0 times out of 1800 decisions on
the Rohrs plant.** L3 was behaviourally identical to L2. The observed difference
came from L2 and L3 drawing agent-scoring noise as differently-sized matrices
(1600 vs 2800 rows), so candidate 0 / seed 0 saw different noise in the two
arms. The comparison was L2 against L2 under two RNG streams.

That confound is still present (KNOWN_ISSUES, C5 residual). Where L2 and L3 are
now contrasted, it is on divergence — a qualitative all-or-nothing outcome that
the noise-stream difference cannot manufacture — and never on a small difference
in mean D.

---

## 6. Tests

```bash
python -m pytest tests/ -v
```

20 tests. What they do and do not establish:

| File | Establishes | Caveat |
|------|-------------|--------|
| `test_tf_eval.py` | TF coefficient ordering, on **non-palindromic** denominators, with a DC-gain anchor (2.0 vs 458 if reversed) | added by the audit; the pre-existing suite could not catch this |
| `test_agent_model.py` | agent model is state- and measurement-compatible with the real plant; no silent broadcast | added by the audit |
| `test_nominal_stability.py` | PI on 2/(s+1): PM = 70.53° matches 2·arctan(1/√2) to < 0.01° | validates only the **PI** path, which no L1–L3 run uses |
| `test_epsilon_zero.py` | modifications are proposed and accepted (75% non-`no_change`); ε = 0 | ε = 0 holds **by construction** here, not by good selection (KNOWN_ISSUES m5) |
| `test_rohrs_instability.py` | MRAC with γ_a = 50 produces max \|y\| = 24.8 | this is a **bounded** saturation-limited response, not divergence (KNOWN_ISSUES m4) |
| `test_determinism.py` | same seed → byte-identical results | **at fixed `--seeds` only** (KNOWN_ISSUES M10) |
| `test_value_bounds.py` | r_t ∈ [0, 1] | tautological: `exp(−nonneg)` cannot leave (0,1] (KNOWN_ISSUES m6) |

---

## 7. How to run

```bash
pip install numpy scipy control pandas matplotlib pytest

python -m pytest tests/ -v

# full sweep. On Windows, PYTHONIOENCODING=utf-8 is required when redirecting
# stdout, or the progress line crashes on the γ/σ/ω characters (KNOWN_ISSUES m7)
PYTHONIOENCODING=utf-8 python experiment.py --seeds 200

python analysis.py
```

Outputs: `results/sweep_results.csv`, and
`results/a_divergence_depth.png`, `b_margins.png`, `c_deterioration.png`.

---

## 8. Repository layout

```
plants.py         — Rohrs plant, CSTR, constraints, fixed-step RK4
controllers.py    — PI, MRAC, structure/adaptation-law parameterisation
agent.py          — candidate enumeration, noisy scoring, ε-optimizer, exact rollout
metrics.py        — value, stability margins, constraint accounting
levels.py         — L0–L3 wiring, paired baselines, divergence detection
experiment.py     — CLI, sweeps, results writing
analysis.py       — divergence, margins, deterioration, ε (all descriptive)
tests/            — 20 tests
AUDIT_REPORT.md   — Phase 1 adversarial audit
KNOWN_ISSUES.md   — defects that remain unfixed, named
```

---

## 9. Citations

- Tětek, J., Sklenka, M., & Gavenčiak, T. (2020). *Performance of
  Bounded-Rational Agents With the Ability to Self-Modify.*
  arXiv:2011.06275 [cs.AI]. https://arxiv.org/abs/2011.06275
  (Submitted November 2020; v2 January 2021. Theorem 7 and Corollary 15
  verified against the arXiv TeX source. Probabilistic analysis is flagged as
  future work in **§5**, not §6.)

- Everitt, T., Filan, D., Daswani, M., & Hutter, M. (2016).
  *Self-Modification of Policy and Utility Function in Rational Agents.*
  AGI-16. arXiv:1605.03142 [cs.AI]. https://arxiv.org/abs/1605.03142
  (Theorem 16 = "Realistic policy-modifying agents make safe modifications",
  verified against the arXiv TeX source.)

- Rohrs, C. E., Valavani, L., Athans, M., & Stein, G. (1985).
  *Robustness of continuous-time adaptive control algorithms in the presence
  of unmodeled dynamics.* IEEE Transactions on Automatic Control, 30(9),
  881–889. doi:**10.1109/TAC.1985.1104070**
  (Corrected. The previous README gave 10.1109/TAC.1985.1104058, which resolves
  to a different paper: "Adaptive stabilization of a discrete linear system with
  an unknown high-frequency gain", IEEE TAC 30(8), 798–799.)

- Bequette, B. W. (1998). *Process Dynamics: Modeling, Analysis, and
  Simulation.* Prentice Hall. (CSTR dimensionless form. The specific
  dimensionless groups used here were not verified against a page or section of
  this text.)
