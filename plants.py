"""Plants, constraints, and fixed-step RK4 integrator.

All plant and integrator operations are vectorised across a leading seed
dimension S so that thousands of episodes can be simulated in one pass.

Plants
------
- RohrsPlant : Rohrs counterexample, nominal first-order + fast unmodeled second-order.
- CSTRPlant  : non-isothermal CSTR with Arrhenius kinetics.

Both expose a common interface:
    dynamics(t, state, u, rng)  -> dstate   (S, n)
    output(state)               -> y         (S, p)
    initial_state(S)            -> state     (S, n)
    tf_nominal / tf_full        -> python-control TransferFunction (for margin tests)
"""

from __future__ import annotations

import numpy as np
import control as ct

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class Constraints:
    """Hard actuator and safety constraints applied identically to every seed.

    saturation : u_min <= u <= u_max
    rate limit : |u - u_prev| <= R * dt  (enforced per step)
    safety     : |y - y_sp| <= y_max   (breach recorded, does not clip)
    """

    def __init__(self, u_min=-np.inf, u_max=np.inf, R=np.inf,
                 y_max=np.inf, dt=0.01):
        self.u_min = u_min
        self.u_max = u_max
        self.R = R
        self.y_max = y_max
        self.dt = dt

    def apply(self, u, u_prev):
        """Return saturated and rate-limited control and per-seed violation flags."""
        S = u.shape[0]
        sat_viol = np.zeros(S, dtype=bool)
        rate_viol = np.zeros(S, dtype=bool)

        # rate limit
        du = u - u_prev
        max_du = self.R * self.dt
        if np.isfinite(max_du):
            clipped = np.clip(du, -max_du, max_du)
            rate_viol = np.abs(du) > (max_du + 1e-12)
            u = u_prev + clipped

        # saturation
        u_sat = np.clip(u, self.u_min, self.u_max)
        sat_viol = (u > self.u_max + 1e-12) | (u < self.u_min - 1e-12)
        return u_sat, sat_viol, rate_viol

    def safety_breach(self, y, y_sp):
        """Return boolean array: True where |y - y_sp| > y_max."""
        return np.abs(y - y_sp) > self.y_max


# ---------------------------------------------------------------------------
# RK4 integrator (fixed step, vectorised across seeds)
# ---------------------------------------------------------------------------

def rk4_step(f, t, state, dt):
    """Single fixed-step RK4 update.  *f* has signature f(t, state) -> dstate.

    The caller is responsible for capturing control inputs, RNGs, etc. in a
    closure so that *f* can compute the derivative from the state alone.
    """
    k1 = f(t, state)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Rohrs counterexample plant
# ---------------------------------------------------------------------------

class RohrsPlant:
    """Rohrs, Valavani, Athons & Stein (1985) counterexample plant.

    Nominal (design) model:
        G1(s) = 2 / (s + 1)          (first-order)

    Full (true) plant:
        G(s) = G1(s) * G2(s)
        G2(s) = 229 / (s^2 + 30 s + 229)   (fast unmodeled second-order)

    State vector (full): [x1, x2, x3]
        x1  : output of G1
        x2  : position state of G2
        x3  : velocity state of G2
        y = x2

    When ``unmodeled=False`` the plant collapses to the nominal first-order
    model with state [x1], y = x1.  This is used for Test 1 (margins on the
    nominal loop) and for the agent's internal model.
    """

    def __init__(self, unmodeled=True, dt=0.01, noise_std=0.0):
        self.unmodeled = unmodeled
        self.dt = dt
        self.noise_std = noise_std
        self.n_inputs = 1
        self.n_outputs = 1
        self.n_states = 3 if unmodeled else 1

    # -- continuous-time dynamics (vectorised) ------------------------------

    def dynamics(self, t, state, u, rng=None):
        """state: (S, n_states), u: (S, 1) -> dstate: (S, n_states)."""
        if not self.unmodeled:
            x1 = state[:, 0:1]
            dx1 = -x1 + 2.0 * u
            ds = dx1
            if self.noise_std > 0 and rng is not None:
                ds = ds + self.noise_std * rng.standard_normal(size=ds.shape)
            return ds

        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        x3 = state[:, 2:3]
        dx1 = -x1 + 2.0 * u
        dx2 = x3
        dx3 = 229.0 * x1 - 229.0 * x2 - 30.0 * x3
        ds = np.concatenate([dx1, dx2, dx3], axis=1)
        if self.noise_std > 0 and rng is not None:
            ds = ds + self.noise_std * rng.standard_normal(size=ds.shape)
        return ds

    def output(self, state):
        if not self.unmodeled:
            return state[:, 0:1]
        return state[:, 1:2]

    def initial_state(self, S):
        return np.zeros((S, self.n_states))

    # -- transfer functions for margin computation --------------------------

    @staticmethod
    def tf_nominal():
        """G1(s) = 2 / (s + 1)."""
        s = ct.tf([1, 0], [1])
        return 2.0 / (s + 1.0)

    @staticmethod
    def tf_full():
        """G(s) = [2/(s+1)] * [229/(s^2+30s+229)]."""
        s = ct.tf([1, 0], [1])
        g1 = 2.0 / (s + 1.0)
        g2 = 229.0 / (s**2 + 30.0 * s + 229.0)
        return g1 * g2

    def loop_tf(self, controller_c):
        """Open-loop transfer L(s) = C(s) * G(s) including unmodeled block."""
        g = self.tf_full() if self.unmodeled else self.tf_nominal()
        return controller_c * g


# ---------------------------------------------------------------------------
# Non-isothermal CSTR (Arrhenius kinetics, jacket cooling)
# ---------------------------------------------------------------------------

class CSTRPlant:
    """Dimensionless non-isothermal CSTR.

    States: [c, theta]  (dimensionless concentration, dimensionless temperature)
    Input:  u = theta_c  (dimensionless coolant temperature)

        dc/dt   = (c_in - c) - Da * c * exp(-gamma_arr / theta)
        dtheta/dt = (theta_in - theta)
                   + B * Da * c * exp(-gamma_arr / theta)
                   - beta * (theta - theta_c)

    Parameters are the standard textbook dimensionless groups (Bequette 1998).
    """

    def __init__(self, Da=1.0, B=5.0, gamma_arr=20.0, beta=2.0,
                 c_in=1.0, theta_in=1.0, dt=0.01, noise_std=0.0):
        self.Da = Da
        self.B = B
        self.gamma_arr = gamma_arr
        self.beta = beta
        self.c_in = c_in
        self.theta_in = theta_in
        self.dt = dt
        self.noise_std = noise_std
        self.n_inputs = 1
        self.n_outputs = 1
        self.n_states = 2

    # -- safe Arrhenius rate (vectorised, avoids overflow) -------------------

    def _rate(self, c, theta):
        """Reaction rate Da * c * exp(-gamma_arr / theta)."""
        # clip theta to avoid overflow / division issues
        th = np.clip(theta, 0.05, 50.0)
        exponent = np.clip(-self.gamma_arr / th, -700, 700)
        return self.Da * c * np.exp(exponent)

    def dynamics(self, t, state, u, rng=None):
        c = state[:, 0:1]
        theta = state[:, 1:2]
        theta_c = u  # (S, 1)

        rate = self._rate(c, theta)
        dc = (self.c_in - c) - rate
        dtheta = (self.theta_in - theta) + self.B * rate \
            - self.beta * (theta - theta_c)
        ds = np.concatenate([dc, dtheta], axis=1)
        if self.noise_std > 0 and rng is not None:
            ds = ds + self.noise_std * rng.standard_normal(size=ds.shape)
        return ds

    def output(self, state):
        return state[:, 1:2]  # measure temperature

    def initial_state(self, S):
        # start near a typical steady state
        return np.tile(np.array([[0.5, 1.4]]), (S, 1)).astype(float)

    def steady_state(self, theta_c_ss):
        """Solve for (c_ss, theta_ss) given coolant temperature (scalar)."""
        from scipy.optimize import fsolve

        def eqs(x):
            c, th = x
            rate = self.Da * c * np.exp(np.clip(-self.gamma_arr / th, -700, 700))
            f1 = (self.c_in - c) - rate
            f2 = (self.theta_in - th) + self.B * rate \
                - self.beta * (th - theta_c_ss)
            return [f1, f2]

        sol = fsolve(eqs, [0.5, 1.4], full_output=False)
        return sol[0], sol[1]

    def linearise(self, theta_c_ss):
        """Return (A, B, C, D) of the linearised plant at the steady state."""
        c_ss, th_ss = self.steady_state(theta_c_ss)
        g = self.gamma_arr
        th = th_ss
        rate = self.Da * c_ss * np.exp(np.clip(-g / th, -700, 700))
        drate_dc = rate / c_ss if c_ss != 0 else 0
        drate_dth = rate * g / th**2

        A = np.array([
            [-1 - drate_dc,           drate_dth],
            [ self.B * drate_dc,  -1 + self.B * drate_dth - self.beta],
        ])
        B = np.array([[0.0], [self.beta]])
        C = np.array([[0.0, 1.0]])
        D = np.array([[0.0]])
        return A, B, C, D

    def loop_tf(self, controller_c, theta_c_ss=0.8):
        """Open-loop L(s) = C(s) * G_lin(s) at the given operating point."""
        A, B, C, D = self.linearise(theta_c_ss)
        g = ct.ss2tf(A, B, C, D)
        return controller_c * g
