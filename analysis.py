"""Analysis: Q1–Q3, L3-vs-L2 comparison, and plots.

Usage:
    python analysis.py                          # load results/sweep_results.csv
    python analysis.py --input results/foo.csv  # custom input
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
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_data(path):
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Q1: Mean deterioration vs k, with Theorem 7 envelope
# ---------------------------------------------------------------------------

def analysis_q1(df):
    print("\n" + "=" * 60)
    print("Q1: Instantiation — deterioration vs modification step k")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)

    for pi, plant in enumerate(['rohrs', 'cstr']):
        ax = axes[0, pi]
        for level in [1, 2, 3]:
            sub = df[(df.plant == plant) & (df.level == level)]
            if len(sub) == 0:
                continue
            grp = sub.groupby('mod_step')
            k = grp['mod_step'].mean().values
            D_mean = grp['D'].mean().values
            D_p95 = grp['D'].apply(lambda x: np.nanpercentile(x, 95)).values
            ax.plot(k, D_mean, '-o', label=f'L{level}', markersize=4)
            ax.fill_between(k, D_mean, D_p95, alpha=0.2)

            # Theorem 7 bound (mean eps)
            if level >= 2:
                eps_mean = grp['eps_emp'].mean().values
                gamma_val = sub['gamma'].iloc[0]
                bound = np.minimum(
                    eps_mean / gamma_val**(k - 1),
                    1.0 / (1.0 - gamma_val))
                ax.plot(k, bound, '--', label=f'L{level} bound', alpha=0.5)

        ax.set_xlabel('Modification step k')
        ax.set_ylabel('Deterioration D_k = V_baseline - V_selfmod')
        ax.set_title(f'Q1: {plant.upper()}')
        ax.legend(fontsize=7)
        ax.axhline(0, color='gray', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'q1_deterioration.png'), dpi=150)
    print(f"  Saved q1_deterioration.png")

    # Summary table
    for level in [0, 1, 2, 3]:
        sub = df[df.level == level]
        if len(sub) == 0:
            continue
        D = sub.D.values
        print(f"  L{level}: D mean={np.nanmean(D):.4f} "
              f"p95={np.nanpercentile(D, 95):.4f} "
              f"max={np.nanmax(D):.4f}")


# ---------------------------------------------------------------------------
# Q2: Distribution of D_k, typicality
# ---------------------------------------------------------------------------

def analysis_q2(df):
    print("\n" + "=" * 60)
    print("Q2: Typicality — distribution of D_k over seeds")
    print("=" * 60)

    # Fixed k = 5, level 2/3, rohrs, gamma=0.95, H=20, sigma=0.01
    for level in [2, 3]:
        sub = df[(df.level == level) & (df.plant == 'rohrs') &
                 (df.gamma == 0.95) & (df.H == 20) &
                 (df.sigma_n == 0.01) & (df.mod_step == 5)]
        if len(sub) < 10:
            continue
        D = sub.D.values
        print(f"  L{level} k=5: median={np.median(D):.4f} "
              f"p95={np.nanpercentile(D, 95):.4f} "
              f"max={np.nanmax(D):.4f} "
              f"n={len(D)}")

    # Growth fit: median D_k vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    for level, color in [(2, 'blue'), (3, 'red')]:
        sub = df[(df.level == level) & (df.plant == 'rohrs') &
                 (df.gamma == 0.95) & (df.H == 20) &
                 (df.sigma_n == 0.01)]
        if len(sub) < 10:
            continue
        grp = sub.groupby('mod_step')
        k = grp['mod_step'].mean().values
        medians = grp['D'].apply(lambda x: np.nanmedian(x)).values
        p95s = grp['D'].apply(lambda x: np.nanpercentile(x, 95)).values

        ax.plot(k, medians, '-o', color=color, label=f'L{level} median')
        ax.plot(k, p95s, '--', color=color, label=f'L{level} p95')

        # Exponential fit to median (only if positive growth)
        if np.all(np.isfinite(medians)) and len(k) >= 3:
            try:
                def exp_model(x, a, b):
                    return a * np.exp(b * x)
                popt, pcov = curve_fit(exp_model, k, medians,
                                       p0=[0.01, 0.1], maxfev=5000)
                perr = np.sqrt(np.diag(pcov))
                residuals = medians - exp_model(k, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((medians - np.mean(medians))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                print(f"  L{level} median fit: {popt[0]:.4f}*exp({popt[1]:.4f}*k) "
                      f"R²={r2:.3f} (param err: {perr[1]:.4f})")
                ax.plot(k, exp_model(k, *popt), ':', color=color, alpha=0.5)
            except Exception as e:
                print(f"  L{level} fit failed: {e}")

    ax.set_xlabel('Modification step k')
    ax.set_ylabel('Deterioration D_k')
    ax.set_title('Q2: Median and p95 growth (Rohrs, γ=0.95, H=20, σ=0.01)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'q2_distribution.png'), dpi=150)
    print(f"  Saved q2_distribution.png")
    print(f"  Fit method: scipy curve_fit, exponential model A*exp(B*k)")
    print(f"  Note: 200 seeds supports distributional claims weakly.")


# ---------------------------------------------------------------------------
# Q3: Stability margins, value-up-margin-down, destabilisation depth
# ---------------------------------------------------------------------------

def analysis_q3(df):
    print("\n" + "=" * 60)
    print("Q3: Physical addition — margins vs self-modification depth")
    print("=" * 60)

    # Margin evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for pi, plant in enumerate(['rohrs', 'cstr']):
        ax = axes[0, pi]
        for level, color in [(2, 'blue'), (3, 'red')]:
            sub = df[(df.plant == plant) & (df.level == level) &
                     (df.gamma == 0.95) & (df.H == 20) &
                     (df.sigma_n == 0.01)]
            if len(sub) < 10:
                continue
            grp = sub.groupby('mod_step')
            k = grp['mod_step'].mean().values
            gm = grp['gain_margin'].apply(
                lambda x: np.nanmedian(x[x < 1e10]) if np.any(x < 1e10) else np.nan
            ).values
            ax.plot(k, gm, '-o', color=color, label=f'L{level} GM (median)')
        ax.set_xlabel('Modification step k')
        ax.set_ylabel('Gain margin (median, linear)')
        ax.set_title(f'Q3: {plant.upper()} margins')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'q3_margins.png'), dpi=150)
    print(f"  Saved q3_margins.png")

    # Value-up-margin-down fraction
    print("\n  Value-up-margin-down fraction (in-sample improves, margin degrades):")
    for level in [2, 3]:
        sub = df[(df.level == level) & (df.val_up_margin_down == True)]
        total = df[(df.level == level) & (df.eps_emp.notna())]
        if len(total) == 0:
            continue
        frac = len(sub) / len(total) if len(total) > 0 else 0
        print(f"    L{level}: {frac:.3f} ({len(sub)}/{len(total)})")

    # Destabilisation depth k*
    print("\n  Destabilisation depth k* (where loop first destabilises):")
    for level in [2, 3]:
        sub = df[(df.level == level) & (df.unstable == True)]
        if len(sub) == 0:
            print(f"    L{level}: no destabilisation observed")
            continue
        k_stars = sub.drop_duplicates(subset=['seed'])['k_unstable']
        k_stars = k_stars[k_stars >= 0]
        if len(k_stars) == 0:
            print(f"    L{level}: no destabilisation observed")
        else:
            print(f"    L{level}: k* median={np.median(k_stars):.0f} "
                  f"n_unstable={len(k_stars)}")


# ---------------------------------------------------------------------------
# L3 vs L2 comparison
# ---------------------------------------------------------------------------

def analysis_l3_vs_l2(df):
    print("\n" + "=" * 60)
    print("L3 vs L2: Does modifying the adaptation law make things worse?")
    print("=" * 60)

    for plant in ['rohrs', 'cstr']:
        l2 = df[(df.level == 2) & (df.plant == plant)]
        l3 = df[(df.level == 3) & (df.plant == plant)]
        if len(l2) == 0 or len(l3) == 0:
            continue

        D2 = l2.D.values
        D3 = l3.D.values
        eps2 = l2.eps_emp.dropna().values
        eps3 = l3.eps_emp.dropna().values

        print(f"\n  {plant.upper()}:")
        mD2 = np.nanmean(D2)
        mD3 = np.nanmean(D3)
        print(f"    D:  L2 mean={mD2:.4f}  L3 mean={mD3:.4f}  "
              f"diff={mD3-mD2:.4f}")
        if len(eps2) > 0 and len(eps3) > 0:
            me2 = np.mean(eps2)
            me3 = np.mean(eps3)
            print(f"    eps: L2 mean={me2:.6f}  "
                  f"L3 mean={me3:.6f}  "
                  f"ratio={me3/max(me2, 1e-10):.2f}")


# ---------------------------------------------------------------------------
# Epsilon summary
# ---------------------------------------------------------------------------

def epsilon_summary(df):
    print("\n" + "=" * 60)
    print("Empirical epsilon distribution per configuration")
    print("=" * 60)

    for level in [2, 3]:
        for plant in ['rohrs', 'cstr']:
            sub = df[(df.level == level) & (df.plant == plant)]
            eps = sub.eps_emp.dropna().values
            if len(eps) == 0:
                continue
            print(f"  L{level} {plant}: eps mean={np.mean(eps):.6f} "
                  f"p95={np.percentile(eps, 95):.6f} "
                  f"max={np.max(eps):.6f}")

    # Theorem 7 violations (only for L2/L3 where epsilon-optimizer applies)
    viol = df[(df.bound_violated == True) & (df.level >= 2)]
    if len(viol) > 0:
        # Separate: eps=0 violations (trajectory divergence) vs eps>0
        v0 = viol[viol.eps_emp == 0]
        v1 = viol[viol.eps_emp > 0]
        print(f"\n  Theorem 7 flagged rows (L2/L3): {len(viol)}")
        print(f"    eps=0 (trajectory divergence, bound=0): {len(v0)}")
        print(f"    eps>0 (genuine bound exceedance): {len(v1)}")
        if len(v1) > 0:
            print(f"      D range: [{v1.D.min():.4f}, {v1.D.max():.4f}]")
            print(f"      bound range: [{v1.theorem7_bound.min():.4f}, "
                  f"{v1.theorem7_bound.max():.4f}]")
        print(f"    Note: paired-trajectory D_k can exceed the same-state bound")
        print(f"    because the two runs diverge after step 1.")
    else:
        print(f"\n  No Theorem 7 violations (L2/L3).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(RESULTS_DIR, "sweep_results.csv"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    analysis_q1(df)
    analysis_q2(df)
    analysis_q3(df)
    analysis_l3_vs_l2(df)
    epsilon_summary(df)


if __name__ == "__main__":
    main()
