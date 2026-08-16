"""Test 4: Determinism — same seed => byte-identical results file."""
import numpy as np
import pytest

from levels import run_config


def test_determinism():
    """Running the same config twice must produce identical results."""
    rows1 = run_config(
        level=2, plant_type="rohrs",
        gamma=0.95, H=20, sigma_n=0.01, delta_m=0.5,
        n_seeds=5, dt=0.01, T_episode=10.0, T_mod=1.0,
        ep_noise=0.01, base_seed=42)

    rows2 = run_config(
        level=2, plant_type="rohrs",
        gamma=0.95, H=20, sigma_n=0.01, delta_m=0.5,
        n_seeds=5, dt=0.01, T_episode=10.0, T_mod=1.0,
        ep_noise=0.01, base_seed=42)

    assert len(rows1) == len(rows2)

    for r1, r2 in zip(rows1, rows2):
        for key in r1:
            v1 = r1[key]
            v2 = r2[key]
            if isinstance(v1, float) or isinstance(v2, float):
                if np.isnan(v1) and np.isnan(v2):
                    continue
                assert v1 == v2 or abs(v1 - v2) < 1e-15, \
                    f"{key}: {v1} != {v2}"
            elif isinstance(v1, bool) or isinstance(v2, bool):
                assert v1 == v2, f"{key}: {v1} != {v2}"
            elif isinstance(v1, (int, str)):
                assert v1 == v2, f"{key}: {v1} != {v2}"
            elif isinstance(v1, float):
                if np.isnan(v1) and np.isnan(v2):
                    continue
                assert abs(v1 - v2) < 1e-15, f"{key}: {v1} != {v2}"
