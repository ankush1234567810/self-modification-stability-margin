"""Test 3: Reproduce the qualitative Rohrs instability.

Rohrs, Valavani, Athans & Stein (1985): MRAC with high adaptation gain and
unmodeled high-frequency dynamics goes unstable with a sinusoidal reference
in the excitation region.

Plant: G(s) = [2/(s+1)] * [229/(s^2+30s+229)]  (unmodeled poles ~15 rad/s)
MRAC:   theta_dot = -gamma_a * e * phi,  high gamma_a, no sigma-mod, no dead zone
Ref:    sinusoidal at ~5 rad/s
"""
import numpy as np
import pytest

from plants import RohrsPlant, Constraints, rk4_step
from controllers import ControllerParams, controller_compute, controller_dynamics, controller_initial_state


def test_rohrs_instability():
    dt = 0.005  # finer step for stability detection
    n_steps = 4000  # 20 seconds
    plant = RohrsPlant(unmodeled=True, dt=dt, noise_std=0.0)
    constraints = Constraints(u_min=-100, u_max=100, R=1000, y_max=100, dt=dt)

    # high adaptation gain, no sigma-modification, no dead zone
    params = ControllerParams.default(S=1, level=1, a_m=2.0)
    params.gamma_a = np.array([50.0])
    params.sigma = np.array([0.0])
    params.dead_zone = np.array([0.0])

    ctrl_state = controller_initial_state(1, params)
    x = plant.initial_state(1)
    u_prev = np.zeros((1, 1))

    omega = 5.0  # rad/s
    def ref(t):
        return np.array([[1.0 + 0.5 * np.sin(omega * t)]])

    max_y = 0.0
    for t in range(n_steps):
        time = t * dt
        y = plant.output(x)
        r = ref(time)
        u = controller_compute(time, ctrl_state, y, r, params)
        u_sat, _, _ = constraints.apply(u, u_prev)

        max_y = max(max_y, abs(y[0, 0]))

        if max_y > 1e4 or np.isnan(y[0, 0]):
            break

        def pdyn(tt, s):
            return plant.dynamics(tt, s, u_sat, None)
        x = rk4_step(pdyn, time, x, dt)

        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, params)
        ctrl_state = rk4_step(cdyn, time, ctrl_state, dt)
        u_prev = u_sat.copy()

    print(f"  max |y| = {max_y:.2f}")
    # The Rohrs counterexample should show instability (large output)
    assert max_y > 10.0, \
        f"Expected Rohrs instability (|y| > 10), got max |y| = {max_y:.2f}"
