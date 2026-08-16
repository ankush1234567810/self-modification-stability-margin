"""Test 5: Value bounds — r_t in [0, 1] asserted every step."""
import numpy as np
import pytest

from plants import RohrsPlant, CSTRPlant, Constraints, rk4_step
from controllers import ControllerParams, controller_compute, controller_dynamics, controller_initial_state
from metrics import reward


def test_reward_bounds_rohrs():
    dt = 0.01
    n_steps = 500
    plant = RohrsPlant(unmodeled=True, dt=dt, noise_std=0.0)
    constraints = Constraints(u_min=-10, u_max=10, R=100, y_max=5, dt=dt)
    params = ControllerParams.default(S=1, level=2, a_m=2.0)
    ctrl_state = controller_initial_state(1, params)
    x = plant.initial_state(1)
    u_prev = np.zeros((1, 1))

    def ref(t):
        return np.array([[1.0 + 0.3 * np.sin(5.0 * t)]])

    for t in range(n_steps):
        time = t * dt
        y = plant.output(x)
        r = ref(time)
        u = controller_compute(time, ctrl_state, y, r, params)
        u_sat, _, _ = constraints.apply(u, u_prev)
        du = u_sat - u_prev
        e = r - y

        r_t = reward(e[:, 0], du[:, 0], w_e=1.0, w_u=0.1)
        assert np.all(r_t >= 0.0), f"r_t < 0 at step {t}: {r_t}"
        assert np.all(r_t <= 1.0), f"r_t > 1 at step {t}: {r_t}"

        def pdyn(tt, s):
            return plant.dynamics(tt, s, u_sat, None)
        x = rk4_step(pdyn, time, x, dt)

        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, params)
        ctrl_state = rk4_step(cdyn, time, ctrl_state, dt)
        u_prev = u_sat.copy()


def test_reward_bounds_cstr():
    dt = 0.01
    n_steps = 500
    plant = CSTRPlant(dt=dt, noise_std=0.0)
    constraints = Constraints(u_min=0.3, u_max=3.0, R=20, y_max=0.5, dt=dt)
    params = ControllerParams.default(S=1, level=1, a_m=2.0)
    ctrl_state = controller_initial_state(1, params)
    x = plant.initial_state(1)
    u_prev = np.zeros((1, 1))

    def ref(t):
        return np.array([[1.4]])

    for t in range(n_steps):
        time = t * dt
        y = plant.output(x)
        r = ref(time)
        u = controller_compute(time, ctrl_state, y, r, params)
        u_sat, _, _ = constraints.apply(u, u_prev)
        du = u_sat - u_prev
        e = r - y

        r_t = reward(e[:, 0], du[:, 0], w_e=1.0, w_u=0.1)
        assert np.all(r_t >= 0.0), f"r_t < 0 at step {t}: {r_t}"
        assert np.all(r_t <= 1.0), f"r_t > 1 at step {t}: {r_t}"

        def pdyn(tt, s):
            return plant.dynamics(tt, s, u_sat, None)
        x = rk4_step(pdyn, time, x, dt)

        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, params)
        ctrl_state = rk4_step(cdyn, time, ctrl_state, dt)
        u_prev = u_sat.copy()


def test_reward_function_properties():
    """r_t = exp(-(w_e*e^2 + w_u*du^2)) is always in [0, 1]."""
    e = np.array([0.0, 1.0, 10.0, 100.0, -5.0])
    du = np.array([0.0, 1.0, 10.0, 100.0, -5.0])
    r = reward(e, du, w_e=1.0, w_u=0.1)

    assert np.all(r >= 0.0), "reward must be >= 0"
    assert np.all(r <= 1.0), "reward must be <= 1"
    # zero error and zero du → reward = 1
    assert abs(r[0] - 1.0) < 1e-15
