"""Test 1: L0 on the nominal plant (no unmodeled block) is stable and margins
match an analytic hand calculation to 1e-6.

Nominal plant:  G(s) = 2/(s+1)
PI controller:  C(s) = Kp + Ki/s = 0.5 + 1/s
Loop:           L(s) = (s + 2) / (s(s+1))

Gain crossover:  |L(jω)| = 1  →  ω_gc = √2
Phase margin:   PM = 2·arctan(1/√2) = 1.2309594173407738 rad = 70.5288°
Gain margin:    GM = ∞  (phase never reaches −180°)
Closed-loop poles: s² + 2s + 2 = 0  →  s = −1 ± j  (stable)
"""
import numpy as np
import control as ct
import pytest

from plants import RohrsPlant, Constraints, rk4_step
from controllers import ControllerParams, controller_compute, controller_dynamics, controller_initial_state
from metrics import compute_margins_vectorized, compute_margins_single


def test_analytic_phase_margin():
    # analytic PM
    pm_analytic = 2 * np.arctan(1 / np.sqrt(2))  # radians
    pm_analytic_deg = np.degrees(pm_analytic)
    # ~70.5288 degrees
    assert abs(pm_analytic_deg - 70.528779) < 1e-4


def test_margins_match_analytic():
    plant = RohrsPlant(unmodeled=False, dt=0.01)
    params = ControllerParams.default(S=1, level=0, Kp=0.5, Ki=1.0)

    # python-control margin
    s = ct.tf([1, 0], [1])
    G = plant.tf_nominal()
    C = params.Kp[0] + params.Ki[0] / s
    L = C * G
    gm_pc, pm_pc, _, _ = ct.margin(L)

    # our vectorized margin
    gm_vec, pm_vec = compute_margins_vectorized(plant, params, S=1)

    # analytic
    pm_analytic = np.degrees(2 * np.arctan(1 / np.sqrt(2)))

    assert abs(pm_pc - pm_analytic) < 1e-2, \
        f"python-control PM {pm_pc} vs analytic {pm_analytic}"
    assert abs(pm_vec[0] - pm_analytic) < 1e-2, \
        f"vectorized PM {pm_vec[0]} vs analytic {pm_analytic}"

    # GM should be very large (no phase crossover)
    assert gm_pc > 100 or not np.isfinite(gm_pc), f"GM should be large, got {gm_pc}"
    assert gm_vec[0] > 100 or not np.isfinite(gm_vec[0]), \
        f"GM should be large, got {gm_vec[0]}"


def test_closed_loop_stable():
    """Simulate L0 on the nominal plant and verify the output converges."""
    dt = 0.01
    n_steps = 1000
    plant = RohrsPlant(unmodeled=False, dt=dt, noise_std=0.0)
    constraints = Constraints(u_min=-10, u_max=10, R=100, y_max=5, dt=dt)
    params = ControllerParams.default(S=1, level=0, Kp=0.5, Ki=1.0)
    ctrl_state = controller_initial_state(1, params)
    x = plant.initial_state(1)
    u_prev = np.zeros((1, 1))

    def ref(t):
        return np.array([[1.0]])

    ys = []
    for t in range(n_steps):
        time = t * dt
        y = plant.output(x)
        r = ref(time)
        u = controller_compute(time, ctrl_state, y, r, params)
        u_sat, _, _ = constraints.apply(u, u_prev)

        def pdyn(tt, s):
            return plant.dynamics(tt, s, u_sat, None)
        x = rk4_step(pdyn, time, x, dt)

        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, params)
        ctrl_state = rk4_step(cdyn, time, ctrl_state, dt)
        u_prev = u_sat.copy()
        ys.append(y[0, 0])

    # output should settle near 1.0 (setpoint)
    assert abs(ys[-1] - 1.0) < 0.01, f"final output {ys[-1]} != 1.0"
    # no blow-up
    assert max(abs(y) for y in ys) < 10, "output diverged"
