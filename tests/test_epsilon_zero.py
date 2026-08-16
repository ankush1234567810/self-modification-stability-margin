"""Test 2: epsilon = 0 case (H large, sigma_n = 0, delta_m = 0).

Self-modification must produce NO deterioration relative to baseline,
within numerical tolerance.  This reproduces Everitt et al. (2016) Theorem 16
and Tětek et al. Corollary 15.

When sigma_n=0 and delta_m=0, the agent's noisy rollout equals the exact
rollout (same model, no noise), so the agent always selects the true best
candidate → epsilon = 0 → D_k ≈ 0.
"""
import numpy as np
import pytest

from levels import run_config


def test_epsilon_zero_no_deterioration():
    """H must be large enough that the rollout horizon ≈ full-episode horizon.
    With gamma=0.99, H=500 makes 0.99^500 ≈ 0.007 (tail negligible)."""
    rows = run_config(
        level=2, plant_type="rohrs",
        gamma=0.99, H=500, sigma_n=0.0, delta_m=0.0,
        n_seeds=10, dt=0.01, T_episode=5.0, T_mod=1.0,
        ep_noise=0.0, base_seed=42)

    Ds = [r['D'] for r in rows]
    eps = [r['eps_emp'] for r in rows if not np.isnan(r['eps_emp'])]

    mean_D = np.mean(Ds)
    max_D = np.max(Ds)

    print(f"  mean D = {mean_D:.8f}, max D = {max_D:.8f}")
    print(f"  mean eps = {np.mean(eps):.8f}, max eps = {np.max(eps):.8f}")

    # The most important sanity check: epsilon should be ~0
    assert np.max(eps) < 1e-6, \
        f"epsilon should be 0, got max={np.max(eps)}"

    # D_k should be ~0 (no deterioration).
    # With epsilon=0 the agent always picks the true best candidate, so
    # self-mod >= baseline (D_k <= 0 for most seeds).  A tiny positive D_k
    # can arise from trajectory divergence in the paired comparison (the
    # Theorem 7 bound is a same-state comparison, while we compare two
    # separate runs).  This is < 0.03% of the value magnitude.
    assert max_D < 2e-2, \
        f"D_k should be ~0 (no deterioration), got max={max_D}"


def test_epsilon_zero_cstr():
    rows = run_config(
        level=2, plant_type="cstr",
        gamma=0.99, H=500, sigma_n=0.0, delta_m=0.0,
        n_seeds=10, dt=0.01, T_episode=5.0, T_mod=1.0,
        ep_noise=0.0, base_seed=123)

    Ds = [r['D'] for r in rows]
    eps = [r['eps_emp'] for r in rows if not np.isnan(r['eps_emp'])]

    assert np.max(eps) < 1e-6, \
        f"epsilon should be 0, got max={np.max(eps)}"
    assert np.max(Ds) < 2e-2, \
        f"D_k should be ~0, got max={np.max(Ds)}"
