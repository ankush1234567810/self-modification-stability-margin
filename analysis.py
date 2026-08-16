"""Analysis and plots.

Usage:
    python analysis.py                          # load results/sweep_results.csv
    python analysis.py --input results/foo.csv  # custom input

Scope note
----------
Three analyses that were present before the Phase 1 audit have been REMOVED
rather than repaired.  See README section "Removed claims" and KNOWN_ISSUES.md:

  - The Theorem 7 bound comparison (audit C4).  epsilon is measured over an
    H-step rollout while D is measured over the episode remainder, so the two
    are not commensurable; and epsilon_emp is a max over 14 hand-picked
    candidates, not the sup over policies the theorem's epsilon requires.
    The raw D data is kept and reported descriptively.
  - The heavy-tail and exponential-growth claims (audit M7, M14).  The medians
    being fitted alternate in sign, so the model class is invalid, and the
    "tail" compared a median from one configuration against a max from all of
    them.  D is now reported descriptively for a single configuration.
  - The L3-vs-L2 headline (audit C5).  At gamma_a = 1 the L3-exclusive
    candidates were selected 0/1800 times on Rohrs, so the comparison was L2
    against L2 under two different RNG streams.

What remains is the question the corrected code can actually answer: with
correct margins and the Rohrs plant operated in its counterexample regime,
does self-modification consume stability margin, and does it ever drive
parameter divergence?
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# The single configuration used wherever one must be named, so that no figure
# or table silently pools incommensurable discount factors (audit M6).
REF_GAMMA, REF_H, REF_SIGMA = 0.95, 20, 0.01


def load_data(path):
    return pd.read_csv(path)


def _fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# A. Divergence: does self-modification drive parameter divergence?
# ---------------------------------------------------------------------------

def analysis_divergence(df):
    print("\n" + "=" * 72)
    print("A. Divergence -- does self-modification drive parameter divergence?")
    print("=" * 72)

    roh = df[df.plant == 'rohrs']
    if len(roh) == 0:
        print("  no Rohrs rows")
        return

    # one row per (config, seed): divergence is an episode-level property
    per_seed = roh.drop_duplicates(
        subset=['level', 'gamma', 'H', 'sigma_n', 'gamma_a', 'omega', 'seed'])

    print(f"\n  Fraction of seeds diverging, by level and adaptation gain")
    print(f"  (Rohrs, all gamma/H/sigma_n pooled -- divergence is a property of")
    print(f"   the loop, not of the discount factor)\n")
    print(f"    {'level':>5} {'gamma_a':>8} {'omega':>7} {'n':>7} "
          f"{'diverged':>9} {'k* med':>7} {'k* min':>7} {'mode':>14}")
    for lvl in sorted(per_seed.level.unique()):
        for ga in sorted(per_seed.gamma_a.unique()):
            for om in sorted(per_seed.omega.unique()):
                sub = per_seed[(per_seed.level == lvl) &
                               (per_seed.gamma_a == ga) &
                               (per_seed.omega == om)]
                if len(sub) == 0:
                    continue
                div = sub[sub.unstable == True]
                ks = div.k_unstable[div.k_unstable >= 0]
                modes = sorted(set(div.diverged_by.dropna())) if len(div) else []
                modes = [m for m in modes if m] or ['-']
                print(f"    {lvl:>5} {ga:>8.1f} {om:>7.1f} {len(sub):>7} "
                      f"{len(div)/len(sub):>9.3f} "
                      f"{_fmt(np.median(ks) if len(ks) else np.nan, 0):>7} "
                      f"{_fmt(ks.min() if len(ks) else np.nan, 0):>7} "
                      f"{'/'.join(modes):>14}")

    # k* histogram, only for cells where anything diverged
    div_cells = per_seed[per_seed.unstable == True]
    if len(div_cells) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        for lvl in sorted(div_cells.level.unique()):
            ks = div_cells[div_cells.level == lvl].k_unstable
            ks = ks[ks >= 0]
            if len(ks) == 0:
                continue
            ax.hist(ks, bins=np.arange(0.5, ks.max() + 1.5), alpha=0.55,
                    label=f'L{lvl} (n={len(ks)})')
        ax.set_xlabel('Destabilisation depth k* (modification step)')
        ax.set_ylabel('Seeds')
        ax.set_title('A: Depth at which the loop first diverges (Rohrs)')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'a_divergence_depth.png'), dpi=150)
        print("\n  Saved a_divergence_depth.png")
    else:
        print("\n  No divergence anywhere -- no k* histogram written.")


# ---------------------------------------------------------------------------
# B. Margin consumption
# ---------------------------------------------------------------------------

def analysis_margins(df):
    print("\n" + "=" * 72)
    print("B. Margin consumption -- does self-modification eat stability margin?")
    print("=" * 72)

    sub = df[(df.plant == 'rohrs') & (df.level >= 2) &
             (df.gamma == REF_GAMMA) & (df.H == REF_H) &
             (df.sigma_n == REF_SIGMA)]
    if len(sub) == 0:
        print("  no rows at the reference configuration")
        return

    print(f"\n  Reference config: Rohrs, gamma={REF_GAMMA}, H={REF_H}, "
          f"sigma_n={REF_SIGMA}")
    print(f"  Gain margin after modification k (median over seeds, "
          f"finite values only)\n")

    fig, axes = plt.subplots(1, len(sorted(sub.gamma_a.unique())),
                             figsize=(5 * sub.gamma_a.nunique(), 4.5),
                             squeeze=False)
    for ai, ga in enumerate(sorted(sub.gamma_a.unique())):
        ax = axes[0, ai]
        for lvl, color in [(2, 'tab:blue'), (3, 'tab:red')]:
            for om, ls in [(5.0, '-'), (16.1, '--')]:
                s = sub[(sub.level == lvl) & (sub.gamma_a == ga) &
                        (sub.omega == om)]
                if len(s) < 10:
                    continue
                grp = s.groupby('mod_step')['gain_margin']
                med = grp.apply(
                    lambda x: np.nanmedian(x[np.isfinite(x)])
                    if np.any(np.isfinite(x)) else np.nan)
                ax.plot(med.index.values, med.values, ls, color=color,
                        marker='o', markersize=3,
                        label=f'L{lvl} omega={om}')
        ax.set_xlabel('Modification step k')
        ax.set_ylabel('Gain margin (median)')
        ax.set_title(f'gamma_a = {ga}')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'b_margins.png'), dpi=150)
    print("  Saved b_margins.png")

    # numeric table: first vs last modification step
    print(f"    {'level':>5} {'gamma_a':>8} {'omega':>7} "
          f"{'GM k=1':>9} {'GM k=last':>10} {'change':>9}")
    for lvl in [2, 3]:
        for ga in sorted(sub.gamma_a.unique()):
            for om in sorted(sub.omega.unique()):
                s = sub[(sub.level == lvl) & (sub.gamma_a == ga) &
                        (sub.omega == om)]
                if len(s) < 10:
                    continue
                fin = s[np.isfinite(s.gain_margin)]
                if len(fin) == 0:
                    continue
                kmin, kmax = fin.mod_step.min(), fin.mod_step.max()
                g0 = fin[fin.mod_step == kmin].gain_margin.median()
                g1 = fin[fin.mod_step == kmax].gain_margin.median()
                print(f"    {lvl:>5} {ga:>8.1f} {om:>7.1f} "
                      f"{g0:>9.3f} {g1:>10.3f} {g1 - g0:>+9.3f}")

    # believed vs realised value-up / margin-down quadrant
    print("\n  Value-up & margin-down quadrant (per modification decision):")
    print(f"    {'level':>5} {'gamma_a':>8} {'omega':>7} "
          f"{'believed':>9} {'realised':>9} {'n':>8}")
    for lvl in [2, 3]:
        for ga in sorted(sub.gamma_a.unique()):
            for om in sorted(sub.omega.unique()):
                s = sub[(sub.level == lvl) & (sub.gamma_a == ga) &
                        (sub.omega == om) & sub.eps_emp.notna()]
                if len(s) == 0:
                    continue
                print(f"    {lvl:>5} {ga:>8.1f} {om:>7.1f} "
                      f"{s.val_up_margin_down.mean():>9.3f} "
                      f"{s.val_up_true_margin_down.mean():>9.3f} {len(s):>8}")
    print("\n  'believed' uses the agent's own noisy scores (self-deception);")
    print("  'realised' uses the exact rollout on the real plant.")


# ---------------------------------------------------------------------------
# C. Deterioration, reported descriptively
# ---------------------------------------------------------------------------

def analysis_deterioration(df):
    print("\n" + "=" * 72)
    print("C. Deterioration D_k = V_baseline - V_selfmod (descriptive only)")
    print("=" * 72)
    print("  No functional form is fitted and no bound is compared against.")
    print("  See README 'Removed claims' for why (audit C4, M7).")

    # M6: never pool across gamma -- the value scale differs by 10x between
    # gamma=0.9 and gamma=0.99, so a pooled mean is not a meaningful number.
    print(f"\n  Per-level summary at gamma={REF_GAMMA}, H={REF_H}, "
          f"sigma_n={REF_SIGMA} (no pooling across gamma):\n")
    print(f"    {'plant':>6} {'level':>5} {'gamma_a':>8} {'omega':>7} "
          f"{'median':>9} {'p05':>9} {'p95':>9} {'n':>7}")
    ref = df[(df.gamma == REF_GAMMA) & (df.H == REF_H) &
             (df.sigma_n == REF_SIGMA)]
    for plant in sorted(ref.plant.unique()):
        for lvl in sorted(ref.level.unique()):
            for ga in sorted(ref.gamma_a.unique()):
                for om in sorted(ref.omega.unique()):
                    s = ref[(ref.plant == plant) & (ref.level == lvl) &
                            (ref.gamma_a == ga) & (ref.omega == om)]
                    if len(s) < 10:
                        continue
                    D = s.D.values
                    print(f"    {plant:>6} {lvl:>5} {ga:>8.1f} {om:>7.1f} "
                          f"{np.nanmedian(D):>9.4f} "
                          f"{np.nanpercentile(D, 5):>9.4f} "
                          f"{np.nanpercentile(D, 95):>9.4f} {len(D):>7}")

    # distribution at one configuration, no fitted model
    sel = ref[(ref.plant == 'rohrs') & (ref.level == 3) &
              (ref.gamma_a == 10.0) & (ref.omega == 5.0)]
    if len(sel) > 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        grp = sel.groupby('mod_step')['D']
        k = np.array(sorted(sel.mod_step.unique()))
        med = grp.median().values
        p05 = grp.apply(lambda x: np.nanpercentile(x, 5)).values
        p95 = grp.apply(lambda x: np.nanpercentile(x, 95)).values
        ax.plot(k, med, '-o', color='tab:red', label='median')
        ax.fill_between(k, p05, p95, alpha=0.2, color='tab:red',
                        label='p05-p95')
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.set_xlabel('Modification step k')
        ax.set_ylabel('D_k = V_baseline - V_selfmod')
        ax.set_title(f'C: D_k spread, Rohrs L3, gamma={REF_GAMMA}, H={REF_H}, '
                     f'sigma={REF_SIGMA}, gamma_a=10, omega=5.0')
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'c_deterioration.png'), dpi=150)
        print("\n  Saved c_deterioration.png")
        print(f"  Median D_k by step: "
              f"{dict(zip(k.tolist(), np.round(med, 4).tolist()))}")


# ---------------------------------------------------------------------------
# D. Realised optimality gap (descriptive; NOT compared to any bound)
# ---------------------------------------------------------------------------

def analysis_epsilon(df):
    print("\n" + "=" * 72)
    print("D. Realised optimality gap eps_emp (descriptive)")
    print("=" * 72)
    print("  eps_emp = max(true_values) - true_value_of_selected, over the")
    print("  agent's 8 (L2) or 14 (L3) candidates, on an H-step exact rollout.")
    print("  It is NOT the epsilon of Tetek et al. Theorem 7, which is a sup")
    print("  over all policies on the infinite-horizon Q. No bound comparison")
    print("  is made -- see README 'Removed claims'.\n")
    print(f"    {'plant':>6} {'level':>5} {'gamma_a':>8} "
          f"{'mean':>10} {'p95':>10} {'max':>10} {'n':>8}")
    for plant in sorted(df.plant.unique()):
        for lvl in [2, 3]:
            for ga in sorted(df.gamma_a.unique()):
                s = df[(df.plant == plant) & (df.level == lvl) &
                       (df.gamma_a == ga)]
                eps = s.eps_emp.dropna().values
                if len(eps) == 0:
                    continue
                print(f"    {plant:>6} {lvl:>5} {ga:>8.1f} "
                      f"{np.mean(eps):>10.6f} {np.percentile(eps, 95):>10.6f} "
                      f"{np.max(eps):>10.6f} {len(eps):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default=os.path.join(RESULTS_DIR, "sweep_results.csv"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    analysis_divergence(df)
    analysis_margins(df)
    analysis_deterioration(df)
    analysis_epsilon(df)


if __name__ == "__main__":
    main()
