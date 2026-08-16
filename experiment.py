"""Experiment runner: CLI, sweeps, results writing.

Usage:
    python experiment.py                      # full sweep
    python experiment.py --seeds 50           # reduced seeds
    python experiment.py --levels 2 3         # only L2, L3
    python experiment.py --plants rohrs       # only Rohrs
"""

from __future__ import annotations

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd

from levels import run_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def run_sweep(seeds=200, levels=None, gammas=None, Hs=None,
              sigma_ns=None, plants=None, delta_m=0.5,
              dt=0.01, T_episode=10.0, T_mod=1.0,
              w_e=1.0, w_u=0.1, ep_noise=0.01, base_seed=42,
              gamma_as=None, omegas=None):
    """Run the full sweep and return a DataFrame.

    *gamma_as* is the MRAC adaptation gain and *omegas* the Rohrs excitation
    frequency.  Both are swept so that the Rohrs counterexample is exercised in
    the regime where it is actually a counterexample: gamma_a = 1 is quiescent,
    gamma_a = 50 diverges, and 16.1 rad/s is the canonical excitation for the
    unmodeled block (poles at -15 +/- 2j).
    """
    if levels is None:
        levels = [0, 1, 2, 3]
    if gammas is None:
        gammas = [0.9, 0.95, 0.99]
    if Hs is None:
        Hs = [5, 20, 100]
    if sigma_ns is None:
        sigma_ns = [0.0, 0.01, 0.1]
    if plants is None:
        plants = ["rohrs", "cstr"]
    if gamma_as is None:
        gamma_as = [1.0, 10.0, 50.0]
    if omegas is None:
        omegas = [5.0, 16.1]

    all_rows = []
    combos = []
    for level in levels:
        for gamma in gammas:
            for H in Hs:
                for sigma_n in sigma_ns:
                    for plant in plants:
                        for ga in gamma_as:
                            for om in omegas:
                                # omega only shapes the Rohrs reference; the
                                # CSTR uses a setpoint step, so sweeping it
                                # there would only duplicate configurations.
                                if plant != "rohrs" and om != omegas[0]:
                                    continue
                                combos.append(
                                    (level, gamma, H, sigma_n, plant, ga, om))

    n_configs = len(combos)
    ci = 0
    t0 = time.time()

    for (level, gamma, H, sigma_n, plant, ga, om) in combos:
        ci += 1
        cfg_seed = base_seed + ci * 10000
        rows = run_config(
            level=level, plant_type=plant,
            gamma=gamma, H=H, sigma_n=sigma_n,
            delta_m=delta_m, n_seeds=seeds,
            dt=dt, T_episode=T_episode, T_mod=T_mod,
            w_e=w_e, w_u=w_u, ep_noise=ep_noise,
            base_seed=cfg_seed, gamma_a=ga, omega=om)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"  [{ci}/{n_configs}] L{level} γ={gamma} "
              f"H={H} σ={sigma_n} {plant} γ_a={ga} ω={om} "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(all_rows)
    total = time.time() - t0
    print(f"\nSweep complete: {len(df)} rows, {total:.1f}s")
    return df, total


def main():
    parser = argparse.ArgumentParser(
        description="Self-modification deterioration sweep")
    parser.add_argument("--seeds", type=int, default=200,
                        help="seeds per configuration")
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[0, 1, 2, 3])
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.9, 0.95, 0.99])
    parser.add_argument("--Hs", type=int, nargs="+",
                        default=[5, 20, 100])
    parser.add_argument("--sigma-ns", type=float, nargs="+",
                        default=[0.0, 0.01, 0.1])
    parser.add_argument("--plants", type=str, nargs="+",
                        default=["rohrs", "cstr"])
    parser.add_argument("--gamma-as", type=float, nargs="+",
                        default=[1.0, 10.0, 50.0],
                        help="MRAC adaptation gains to sweep")
    parser.add_argument("--omegas", type=float, nargs="+",
                        default=[5.0, 16.1],
                        help="Rohrs excitation frequencies (rad/s)")
    parser.add_argument("--delta-m", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T-episode", type=float, default=10.0)
    parser.add_argument("--T-mod", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, runtime = run_sweep(
        seeds=args.seeds, levels=args.levels, gammas=args.gammas,
        Hs=args.Hs, sigma_ns=args.sigma_ns, plants=args.plants,
        delta_m=args.delta_m, dt=args.dt,
        T_episode=args.T_episode, T_mod=args.T_mod,
        base_seed=args.base_seed,
        gamma_as=args.gamma_as, omegas=args.omegas)

    out = args.output or os.path.join(RESULTS_DIR, "sweep_results.csv")
    df.to_csv(out, index=False)
    print(f"Results written to {out}")

    # quick summary
    for level in args.levels:
        sub = df[df.level == level]
        if len(sub) == 0:
            continue
        D = sub.D.values
        eps = sub.eps_emp.values
        eps = eps[~np.isnan(eps)]
        print(f"\nL{level}: D mean={np.nanmean(D):.6f} "
              f"D p95={np.nanpercentile(D, 95):.6f} "
              f"D max={np.nanmax(D):.6f}")
        if len(eps) > 0:
            print(f"       eps mean={np.mean(eps):.6f} "
                  f"eps p95={np.nanpercentile(eps, 95):.6f} "
                  f"eps max={np.nanmax(eps):.6f}")
        div = sub.drop_duplicates(subset=['seed', 'gamma_a', 'omega'])
        if div.unstable.sum() > 0:
            print(f"       diverged seed-configs: {int(div.unstable.sum())}"
                  f"/{len(div)}")


if __name__ == "__main__":
    main()
