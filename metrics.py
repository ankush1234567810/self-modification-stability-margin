"""Metrics: reward, value, stability margins, Theorem 7 envelope, constraints.

All functions are vectorised across a leading seed dimension *S*.
"""

from __future__ import annotations

import numpy as np
import control as ct


# ---------------------------------------------------------------------------
# Reward and value
# ---------------------------------------------------------------------------

def reward(e, du, w_e=1.0, w_u=0.1):
    """Instantaneous utility r_t = exp(-(w_e*e^2 + w_u*du^2)) ∈ (0, 1].

    e  : tracking error  (S, 1) or (S,)
    du : control move    (S, 1) or (S,)
    """
    e = np.asarray(e).reshape(-1)
    du = np.asarray(du).reshape(-1)
    return np.exp(-(w_e * e**2 + w_u * du**2))


def discounted_value(rewards, gamma):
    """V = sum_t gamma^t * r_t  for a 1-D reward array."""
    t = np.arange(len(rewards))
    return np.sum(gamma**t * rewards)


def value_from_step(rewards, k, gamma):
    """Discounted value from step k onward: V_k = sum_{t>=k} gamma^(t-k) r_t."""
    r = np.asarray(rewards)[k:]
    t = np.arange(len(r))
    return np.sum(gamma**t * r)


def value_from_step_vec(rewards_all, mod_steps, gamma):
    """Vectorised: rewards_all (S, T), mod_steps (K,) → values (S, K).

    For each seed *s* and modification step index *k*, compute the discounted
    value from simulation step mod_steps[k] onward.
    """
    S, T = rewards_all.shape
    K = len(mod_steps)
    vals = np.zeros((S, K))
    for ki, step in enumerate(mod_steps):
        r = rewards_all[:, step:]  # (S, T-step)
        t = np.arange(T - step)
        vals[:, ki] = np.sum(gamma**t * r, axis=1)
    return vals


# ---------------------------------------------------------------------------
# Theorem 7 envelope
# ---------------------------------------------------------------------------

def theorem7_bound(eps, k, gamma):
    """min(eps / gamma^(k-1), 1/(1-gamma)).

    eps : realised optimality gap (scalar or array)
    k   : modification step index (1-indexed scalar or array)
    gamma: discount factor
    """
    k = np.asarray(k, dtype=float)
    return np.minimum(eps / gamma**(k - 1), 1.0 / (1.0 - gamma))


# ---------------------------------------------------------------------------
# Stability margins (vectorised across seeds)
# ---------------------------------------------------------------------------

def _eval_tf_on_grid(num_coeffs, den_coeffs, omega):
    """Evaluate a transfer function num/den at frequencies *omega* (rad/s).

    *num_coeffs* and *den_coeffs* are in DECREASING power order, the standard
    control convention (``[1, 31, 259, 229]`` means s^3 + 31 s^2 + 259 s + 229).
    ``np.polyval`` expects exactly that order, so the coefficients are passed
    through unreversed.

    Returns complex array of same shape as omega.
    """
    s = 1j * omega
    num = np.polyval(num_coeffs, s)
    den = np.polyval(den_coeffs, s)
    return num / den


def compute_margins_vectorized(plant, params, S, n_freq=500):
    """Compute gain margin and phase margin for *S* seeds simultaneously.

    Fully vectorised: evaluates L(jω) = C(jω) * G(jω) on a log-spaced
    frequency grid for all seeds at once, then finds gain and phase crossovers
    by sign-change detection with linear interpolation.

    Returns (gm, pm) arrays of shape (S,).
    """
    omega = np.logspace(-2, 3, n_freq)  # 0.01 .. 1000 rad/s
    s = 1j * omega  # (n_freq,)

    # --- plant frequency response (same for all seeds) ---
    if hasattr(plant, 'unmodeled'):
        g_num, g_den = _tf_coeffs_rohrs(plant.unmodeled)
    else:
        A, B, C, D = plant.linearise(0.8)
        g_tf = ct.tf(ct.ss(A, B, C, D))
        g_num = np.array(g_tf.num[0][0])
        g_den = np.array(g_tf.den[0][0])
    G_jw = _eval_tf_on_grid(g_num, g_den, omega)  # (n_freq,)

    # --- controller frequency response (fully vectorised) ---
    # Broadcast: params (S,) × s (n_freq,) → (S, n_freq)
    Kp = np.asarray(params.Kp)[:, None]
    Ki = np.asarray(params.Ki)[:, None]
    Kd = np.asarray(params.Kd)[:, None]
    lead_tau = np.asarray(params.lead_tau)[:, None]
    filter_tau = np.asarray(params.filter_tau)[:, None]
    th2 = np.asarray(params.theta2_init)[:, None]

    s_b = s[None, :]  # (1, n_freq)

    # MRAC: C = -theta2 (static gain)
    C_mrac = -th2 * np.ones_like(s_b)
    # PI: C = Kp + Ki/s
    C_pi = Kp + Ki / s_b

    mrac_on = np.asarray(params.mrac_on)[:, None]
    C = np.where(mrac_on, C_mrac, C_pi)

    # integral action (only for PI path, already included via Ki/s)
    # lead term: Kd * s / (tau*s + 1)
    lead_on = np.asarray(params.lead_on)[:, None]
    lead = Kd * s_b / (lead_tau * s_b + 1)
    C = C + np.where(lead_on, lead, 0.0)

    # output filter: C / (tau_f*s + 1)
    filter_on = np.asarray(params.filter_on)[:, None]
    C = np.where(filter_on, C / (filter_tau * s_b + 1), C)

    L_jw = C * G_jw[None, :]  # (S, n_freq)

    mag = np.abs(L_jw)                    # (S, n_freq)
    phase = np.rad2deg(np.unwrap(np.angle(L_jw), axis=1))  # unwrapped degrees

    # --- phase margin: gain crossover where |L| = 1 ---
    diff_m = mag - 1.0
    sc_m = np.diff(np.sign(diff_m), axis=1) != 0  # (S, n_freq-1)
    has_gc = np.any(sc_m, axis=1)
    first_gc = np.argmax(sc_m, axis=1)  # first sign-change index per seed

    pm = np.full(S, np.inf)
    idx = np.where(has_gc)[0]
    if len(idx) > 0:
        i0 = first_gc[idx]
        x0 = omega[i0]; x1 = omega[i0 + 1]
        y0 = diff_m[idx, i0]; y1 = diff_m[idx, i0 + 1]
        denom = (y1 - y0)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
        w_gc = x0 - y0 * (x1 - x0) / denom
        frac = (np.log(w_gc) - np.log(x0)) / (np.log(x1) - np.log(x0) + 1e-30)
        p0 = phase[idx, i0]; p1 = phase[idx, i0 + 1]
        ph_gc = p0 + frac * (p1 - p0)
        pm[idx] = 180.0 + ph_gc

    # --- gain margin: phase crossover where phase = -180° ---
    diff_p = phase + 180.0
    sc_p = np.diff(np.sign(diff_p), axis=1) != 0
    has_pc = np.any(sc_p, axis=1)
    first_pc = np.argmax(sc_p, axis=1)

    gm = np.full(S, np.inf)
    idx2 = np.where(has_pc)[0]
    if len(idx2) > 0:
        i0 = first_pc[idx2]
        x0 = omega[i0]; x1 = omega[i0 + 1]
        y0 = diff_p[idx2, i0]; y1 = diff_p[idx2, i0 + 1]
        denom = (y1 - y0)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
        w_pc = x0 - y0 * (x1 - x0) / denom
        frac = (np.log(w_pc) - np.log(x0)) / (np.log(x1) - np.log(x0) + 1e-30)
        m0 = mag[idx2, i0]; m1 = mag[idx2, i0 + 1]
        mag_pc = m0 + frac * (m1 - m0)
        mag_pc = np.where(mag_pc > 1e-30, mag_pc, 1e-30)
        gm[idx2] = 1.0 / mag_pc

    return gm, pm


def _tf_coeffs_rohrs(unmodeled):
    """Return (num, den) coefficient arrays for the Rohrs plant."""
    if unmodeled:
        # G(s) = [2/(s+1)] * [229/(s^2+30s+229)]
        # = 458 / (s^3 + 31s^2 + 259s + 229)
        num = np.array([458.0])
        den = np.array([1.0, 31.0, 259.0, 229.0])
    else:
        # G(s) = 2/(s+1)
        num = np.array([2.0])
        den = np.array([1.0, 1.0])
    return num, den


def _controller_freq_resp(ps, s):
    """Evaluate C(jω) for a single seed's scalar params.

    s = jω array.
    Returns complex array same shape as s.
    """
    if ps['mrac_on']:
        C = np.full_like(s, -ps['theta2_init'], dtype=complex)
    else:
        C = np.full_like(s, ps['Kp'], dtype=complex)
        if ps['integral_on']:
            C = C + ps['Ki'] / s

    if ps['lead_on']:
        Kd = ps['Kd']
        tau = ps['lead_tau']
        C = C + Kd * s / (tau * s + 1)

    if ps['filter_on']:
        tau_f = ps['filter_tau']
        C = C / (tau_f * s + 1)

    return C


def compute_margins_single(plant, ps, s_var=None):
    """Compute (gm, pm) for a single seed using python-control's margin().

    Used for test verification and single-seed cases.
    """
    if s_var is None:
        s_var = ct.tf([1, 0], [1])

    if hasattr(plant, 'unmodeled'):
        g = plant.tf_full() if plant.unmodeled else plant.tf_nominal()
    else:
        A, B, C, D = plant.linearise(0.8)
        g = ct.ss2tf(A, B, C, D)

    # build controller TF
    if ps['mrac_on']:
        C = -ps['theta2_init']  # static gain
    else:
        C = ps['Kp']
        if ps['integral_on']:
            C = C + ps['Ki'] / s_var

    if ps['lead_on']:
        C = C + ps['Kd'] * s_var / (ps['lead_tau'] * s_var + 1)
    if ps['filter_on']:
        C = C / (ps['filter_tau'] * s_var + 1)

    L = C * g
    try:
        gm, pm, w_gc, w_pc = ct.margin(L)
        if gm == 0 or gm is None or not np.isfinite(gm):
            gm = np.inf
        if pm is None or not np.isfinite(pm):
            pm = np.inf
    except Exception:
        gm, pm = np.inf, np.inf
    return gm, pm


# ---------------------------------------------------------------------------
# Constraint accounting
# ---------------------------------------------------------------------------

def constraint_summary(sat_violations, rate_violations, safety_breaches,
                       n_steps):
    """Summarise constraint violations over an episode.

    Each input is a (n_steps, S) boolean array.
    Returns dict of per-seed arrays (S,).
    """
    return dict(
        sat_frac=np.mean(sat_violations, axis=0),
        rate_frac=np.mean(rate_violations, axis=0),
        safety_frac=np.mean(safety_breaches, axis=0),
        any_safety=np.any(safety_breaches, axis=0),
    )
