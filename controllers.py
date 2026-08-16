"""Controllers: PI, MRAC, and the structure / adaptation-law parameterisation
for the L0–L3 self-modification ladder.

All operations are vectorised across a leading seed dimension *S*.

Controller state vector (fixed size 6, unused entries stay 0):
    [0] integral        integral action accumulator
    [1] theta1          MRAC adapted parameter (reference gain)
    [2] theta2          MRAC adapted parameter (output feedback gain)
    [3] lead_filt       lead-term filter state
    [4] out_filt        output-filter state
    [5] ref_model       reference-model state y_m
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# controller state layout
INT = 0
TH1 = 1
TH2 = 2
LEAD = 3
OFILT = 4
REF = 5
N_CTRL = 6


def _c(a):
    """Reshape (S,) → (S, 1) for broadcasting with (S, 1) signals."""
    return np.asarray(a).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Controller parameter container (per-seed arrays, shape (S,))
# ---------------------------------------------------------------------------

@dataclass
class ControllerParams:
    S: int
    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray
    lead_tau: np.ndarray
    filter_tau: np.ndarray
    integral_on: np.ndarray
    lead_on: np.ndarray
    filter_on: np.ndarray
    mrac_on: np.ndarray
    gamma_a: np.ndarray
    sigma: np.ndarray
    dead_zone: np.ndarray
    theta1_init: np.ndarray
    theta2_init: np.ndarray
    a_m: np.ndarray

    def copy(self):
        return ControllerParams(
            S=self.S,
            Kp=self.Kp.copy(),
            Ki=self.Ki.copy(),
            Kd=self.Kd.copy(),
            lead_tau=self.lead_tau.copy(),
            filter_tau=self.filter_tau.copy(),
            integral_on=self.integral_on.copy(),
            lead_on=self.lead_on.copy(),
            filter_on=self.filter_on.copy(),
            mrac_on=self.mrac_on.copy(),
            gamma_a=self.gamma_a.copy(),
            sigma=self.sigma.copy(),
            dead_zone=self.dead_zone.copy(),
            theta1_init=self.theta1_init.copy(),
            theta2_init=self.theta2_init.copy(),
            a_m=self.a_m.copy(),
        )

    @classmethod
    def default(cls, S, level, Kp=0.5, Ki=1.0, Kd=0.3, a_m=2.0):
        mrac = level >= 1
        return cls(
            S=S,
            Kp=np.full(S, Kp),
            Ki=np.full(S, Ki),
            Kd=np.full(S, Kd),
            lead_tau=np.full(S, 0.1),
            filter_tau=np.full(S, 0.05),
            integral_on=np.full(S, True),
            lead_on=np.full(S, False),
            filter_on=np.full(S, False),
            mrac_on=np.full(S, mrac),
            gamma_a=np.full(S, 1.0),
            sigma=np.full(S, 0.01),
            dead_zone=np.full(S, 0.0),
            theta1_init=np.full(S, 1.0),
            theta2_init=np.full(S, -0.5),
            a_m=np.full(S, a_m),
        )

    def to_scalar(self):
        """Extract a single-seed dict (for margin TF building)."""
        return dict(
            Kp=float(self.Kp[0]), Ki=float(self.Ki[0]),
            Kd=float(self.Kd[0]), lead_tau=float(self.lead_tau[0]),
            filter_tau=float(self.filter_tau[0]),
            integral_on=bool(self.integral_on[0]),
            lead_on=bool(self.lead_on[0]),
            filter_on=bool(self.filter_on[0]),
            mrac_on=bool(self.mrac_on[0]),
            theta1_init=float(self.theta1_init[0]),
            theta2_init=float(self.theta2_init[0]),
            a_m=float(self.a_m[0]),
        )


# ---------------------------------------------------------------------------
# Controller compute and dynamics (vectorised)
# ---------------------------------------------------------------------------

def controller_compute(t, ctrl_state, y, r, params: ControllerParams):
    """Compute raw (pre-constraint) control u.  y, r: (S, 1) → u: (S, 1)."""
    e = r - y  # (S, 1)

    th1 = ctrl_state[:, TH1:TH1 + 1]
    th2 = ctrl_state[:, TH2:TH2 + 1]

    # MRAC: u = theta1 * r + theta2 * y
    u_mrac = th1 * r + th2 * y
    # PI: u = Kp * e
    u_pi = _c(params.Kp) * e

    u = np.where(_c(params.mrac_on), u_mrac, u_pi)

    # integral action
    integral = ctrl_state[:, INT:INT + 1]
    u = u + np.where(_c(params.integral_on), _c(params.Ki) * integral, 0.0)

    # lead term: Kd/tau * (e - lead_filt)
    lead_filt = ctrl_state[:, LEAD:LEAD + 1]
    lead_contrib = _c(params.Kd) / _c(params.lead_tau) * (e - lead_filt)
    u = u + np.where(_c(params.lead_on), lead_contrib, 0.0)

    # output filter
    out_filt = ctrl_state[:, OFILT:OFILT + 1]
    u = np.where(_c(params.filter_on), out_filt, u)

    return u


def controller_dynamics(t, ctrl_state, y, r, u, params: ControllerParams):
    """Controller state derivative.  Returns (S, N_CTRL)."""
    S = ctrl_state.shape[0]
    ds = np.zeros((S, N_CTRL))

    e = r - y  # (S, 1)
    y_m = ctrl_state[:, REF:REF + 1]

    # reference model: y_m_dot = -a_m * y_m + a_m * r
    ds[:, REF:REF + 1] = -_c(params.a_m) * y_m + _c(params.a_m) * r

    # integral action
    ds[:, INT:INT + 1] = np.where(_c(params.integral_on), e, 0.0)

    # MRAC adaptation
    e_m = y - y_m  # MRAC tracking error
    active = np.abs(e_m) >= _c(params.dead_zone)
    grad1 = np.where(active, -_c(params.gamma_a) * e_m * r, 0.0)
    grad2 = np.where(active, -_c(params.gamma_a) * e_m * y, 0.0)
    th1 = ctrl_state[:, TH1:TH1 + 1]
    th2 = ctrl_state[:, TH2:TH2 + 1]
    ds[:, TH1:TH1 + 1] = np.where(
        _c(params.mrac_on), grad1 - _c(params.sigma) * th1, 0.0)
    ds[:, TH2:TH2 + 1] = np.where(
        _c(params.mrac_on), grad2 - _c(params.sigma) * th2, 0.0)

    # lead filter
    lead_filt = ctrl_state[:, LEAD:LEAD + 1]
    ds[:, LEAD:LEAD + 1] = np.where(
        _c(params.lead_on), (-lead_filt + e) / _c(params.lead_tau), 0.0)

    # output filter
    out_filt = ctrl_state[:, OFILT:OFILT + 1]
    ds[:, OFILT:OFILT + 1] = np.where(
        _c(params.filter_on), (-out_filt + u) / _c(params.filter_tau), 0.0)

    return ds


def controller_initial_state(S, params: ControllerParams):
    cs = np.zeros((S, N_CTRL))
    cs[:, TH1] = params.theta1_init
    cs[:, TH2] = params.theta2_init
    return cs


# ---------------------------------------------------------------------------
# Transfer function for margin computation (single seed)
# ---------------------------------------------------------------------------

def controller_tf(ps, s_var):
    """Build C(s) for a single seed's scalar params dict *ps*.

    In the feedback loop the reference is zero for margin analysis, so the
    controller reduces to the feedback path  u = -C_fb * y.
    For PI:   C_fb = Kp
    For MRAC: C_fb = -theta2  (output-feedback gain from y to u)
    Integral action is ADDITIVE on top of either path, matching
    ``controller_compute``: C_fb += Ki/s when integral_on.

    ``ps['theta2']`` (the live adapted gain) is used when present; otherwise
    ``ps['theta2_init']``, which is only correct before adaptation starts.
    """
    if ps['mrac_on']:
        # u = theta1*r + theta2*y → feedback gain = theta2
        C = -ps.get('theta2', ps['theta2_init'])
    else:
        C = ps['Kp']

    if ps['integral_on']:
        C = C + ps['Ki'] / s_var

    if ps['lead_on']:
        Kd = ps['Kd']
        tau = ps['lead_tau']
        C = C + Kd * s_var / (tau * s_var + 1)

    if ps['filter_on']:
        tau_f = ps['filter_tau']
        C = C / (tau_f * s_var + 1)

    return C


# ---------------------------------------------------------------------------
# Candidate modifications for L2 / L3
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    level: int

    def apply(self, p: ControllerParams) -> ControllerParams:
        q = p.copy()
        if self.name == "no_change":
            pass
        elif self.name == "toggle_integral":
            q.integral_on = ~q.integral_on
        elif self.name == "toggle_lead":
            q.lead_on = ~q.lead_on
        elif self.name == "toggle_filter":
            q.filter_on = ~q.filter_on
        elif self.name == "Kp_up":
            q.Kp = q.Kp * 1.5
        elif self.name == "Kp_down":
            q.Kp = q.Kp / 1.5
        elif self.name == "Ki_up":
            q.Ki = q.Ki * 1.5
        elif self.name == "Ki_down":
            q.Ki = q.Ki / 1.5
        elif self.name == "gamma_up":
            q.gamma_a = q.gamma_a * 2.0
        elif self.name == "gamma_down":
            q.gamma_a = q.gamma_a / 2.0
        elif self.name == "sigma_up":
            q.sigma = q.sigma * 2.0
        elif self.name == "sigma_down":
            q.sigma = q.sigma / 2.0
        elif self.name == "deadzone_up":
            q.dead_zone = q.dead_zone * 2.0
        elif self.name == "deadzone_down":
            q.dead_zone = np.maximum(q.dead_zone / 2.0, 0.0)
        else:
            raise ValueError(f"unknown candidate {self.name}")
        return q


def get_candidates(level):
    cands = [Candidate("no_change", max(level, 2))]
    if level <= 1:
        return cands
    cands += [
        Candidate("toggle_integral", 2),
        Candidate("toggle_lead", 2),
        Candidate("toggle_filter", 2),
        Candidate("Kp_up", 2),
        Candidate("Kp_down", 2),
        Candidate("Ki_up", 2),
        Candidate("Ki_down", 2),
    ]
    if level <= 2:
        return cands
    cands += [
        Candidate("gamma_up", 3),
        Candidate("gamma_down", 3),
        Candidate("sigma_up", 3),
        Candidate("sigma_down", 3),
        Candidate("deadzone_up", 3),
        Candidate("deadzone_down", 3),
    ]
    return cands
