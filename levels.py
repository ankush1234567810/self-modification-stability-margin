"""Levels L0–L3 wiring: episode simulation, paired baselines, metric recording.

Each episode is vectorised across *S* seeds.  For every configuration we run
two paired episodes — self-modifying and baseline (self-mod disabled after
step 1) — with identical disturbance realisations (same plant RNG seed).
"""

from __future__ import annotations

import numpy as np
import hashlib
import json

from plants import RohrsPlant, CSTRPlant, Constraints, rk4_step
from controllers import (
    ControllerParams, controller_compute, controller_dynamics,
    controller_initial_state, get_candidates, TH2,
)
from agent import make_agent_model, epsilon_optimize, apply_choices, rollout
from metrics import (
    reward, value_from_step_vec, theorem7_bound,
    compute_margins_vectorized,
)

# ---------------------------------------------------------------------------
# Divergence detection thresholds
# ---------------------------------------------------------------------------
#
# The previous detector was |y| > 1e6.  That can never fire: both plants are
# open-loop stable and the actuator saturates (|u| <= 10 on Rohrs, u in
# [0.3, 3.0] on the CSTR), so |y| <= ~20 and |theta| <= ~2.4 respectively, for
# ANY controller.  The threshold sat 50,000x outside the reachable set, which
# made "no destabilisation observed" a property of the setup rather than a
# result.
#
# The genuine adaptive-control failure mode here is parameter drift, not output
# divergence.  Measured over a 10 s episode (dt = 0.01, ep_noise = 0.01):
#
#   omega  gamma_a   max|theta2|   max|y|   fraction |y| > y_max
#     5.0      1.0        0.79       1.26        0.000
#     5.0     10.0        4.44       1.75        0.000
#     5.0     50.0     1240          2.87        0.000     <- diverging
#    16.1      1.0        0.77       1.22        0.000
#    16.1     10.0        4.31       1.63        0.000
#    16.1     50.0     1209          2.83        0.000     <- diverging
#
# Healthy adaptation caps at |theta2| ~ 4.4; the divergent regime reaches ~1240
# and is still growing at episode end.  THETA2_DIVERGE = 50 sits >10x above the
# healthy maximum and ~25x below the divergent value.  Note that |y| stays under
# 2.9 even while theta2 runs away, which is exactly why an output-magnitude
# detector cannot see this failure.
#
# Both criteria must hold for SUSTAIN_SECONDS of continuous simulation time, so
# that a large bounded transient or a momentary saturation excursion does not
# register as divergence.
THETA2_DIVERGE = 50.0
SUSTAIN_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def make_plant(plant_type, dt=0.01, noise_std=0.01):
    if plant_type == "rohrs":
        return RohrsPlant(unmodeled=True, dt=dt, noise_std=noise_std)
    elif plant_type == "cstr":
        return CSTRPlant(dt=dt, noise_std=noise_std)
    raise ValueError(plant_type)


def make_ref_fn(plant_type, S, omega=16.1):
    """Reference signal.

    For the Rohrs plant *omega* is the sinusoidal excitation frequency.  The
    unmodeled block 229/(s^2 + 30s + 229) has poles at -15 +/- 2j, i.e. a
    natural frequency of 15.13 rad/s; the canonical destabilising input in
    Rohrs et al. (1985) is at 16.1 rad/s.  The previous default of 5.0 rad/s
    sat well below that corner and did not excite the unmodeled dynamics -- the
    old comment claiming it did was incorrect.
    """
    if plant_type == "rohrs":
        def ref(t):
            return np.full((S, 1), 1.0 + 0.3 * np.sin(omega * t))
        return ref
    elif plant_type == "cstr":
        def ref(t):
            sp = 1.4 if t < 5.0 else 1.6
            return np.full((S, 1), sp)
        return ref


def make_constraints(plant_type, dt=0.01):
    if plant_type == "rohrs":
        return Constraints(u_min=-10.0, u_max=10.0, R=100.0,
                          y_max=5.0, dt=dt)
    elif plant_type == "cstr":
        return Constraints(u_min=0.3, u_max=3.0, R=20.0,
                          y_max=0.5, dt=dt)


def config_hash(config):
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Episode simulation (vectorised across S seeds)
# ---------------------------------------------------------------------------

def run_episode(level, plant_type, config, S, plant_seed, agent_seed,
                self_mod=True):
    """Run one vectorised episode for *S* seeds.

    Returns a dict of arrays.
    """
    dt = config['dt']
    n_steps = config['n_steps']
    steps_per_mod = config['steps_per_mod']
    gamma = config['gamma']
    w_e = config['w_e']
    w_u = config['w_u']
    H = config['H']
    sigma_n = config['sigma_n']
    delta_m = config['delta_m']
    gamma_a = config.get('gamma_a', 1.0)
    omega = config.get('omega', 16.1)

    plant_rng = np.random.default_rng(plant_seed)
    agent_rng = np.random.default_rng(agent_seed)

    # plants
    ep_noise = config.get('ep_noise', 0.01)
    episode_plant = make_plant(plant_type, dt, ep_noise)
    exact_plant = make_plant(plant_type, dt, 0.0)  # no noise for exact rollout
    agent_model = make_agent_model(episode_plant, delta_m, sigma_n, dt)
    constraints = make_constraints(plant_type, dt)
    ref_fn = make_ref_fn(plant_type, S, omega=omega)

    # controller
    params = ControllerParams.default(S, level, gamma_a=gamma_a)
    ctrl_state = controller_initial_state(S, params)

    # state
    x = episode_plant.initial_state(S)
    u_prev = np.zeros((S, 1))

    # recording
    rewards = np.zeros((S, n_steps))
    sat_viol = np.zeros((n_steps, S), dtype=bool)
    rate_viol = np.zeros((n_steps, S), dtype=bool)
    safety_viol = np.zeros((n_steps, S), dtype=bool)
    unstable = np.zeros(S, dtype=bool)
    k_unstable = np.full(S, -1, dtype=int)
    # divergence detection: per-seed run-length counters (see THETA2_DIVERGE)
    sustain_steps = max(1, int(SUSTAIN_SECONDS / dt))
    drift_run = np.zeros(S, dtype=int)
    breach_run = np.zeros(S, dtype=int)
    diverged_by = np.full(S, "", dtype=object)

    # Modification step records.
    # Modifications occur at t = steps_per_mod, 2*steps_per_mod, ... while
    # t < n_steps, i.e. k = 1 .. (n_steps-1)//steps_per_mod.  The old
    # n_steps//steps_per_mod overcounted by one whenever n_steps is an exact
    # multiple: with n_steps=1000, steps_per_mod=100 the loop runs t=0..999 so
    # the last modification is at t=900 (k=9), but n_mod was 10.  That phantom
    # row carried eps=NaN, chosen=-1 and -- because mod_steps was clipped to
    # n_steps-1 -- a value V computed from a SINGLE reward sample rather than an
    # episode remainder.  It was 21,600 rows of the sweep and, with the stale
    # eps=0.0 in the committed CSV, the single largest source of spurious
    # Theorem 7 bound violations.
    n_mod = (n_steps - 1) // steps_per_mod
    mod_records = []
    candidates = get_candidates(level)
    n_cand = len(candidates)

    # per-mod-step arrays (for L2/L3)
    eps_per_mod = np.full((n_mod, S), np.nan)
    in_sample_per_mod = np.full((n_mod, S), np.nan)
    true_sel_per_mod = np.full((n_mod, S), np.nan)
    true_best_per_mod = np.full((n_mod, S), np.nan)
    chosen_per_mod = np.full((n_mod, S), -1, dtype=int)
    gm_per_mod = np.full((n_mod, S), np.nan)
    pm_per_mod = np.full((n_mod, S), np.nan)
    val_up_margin_down = np.zeros((n_mod, S), dtype=bool)

    # previous margins (for detecting value-up-margin-down)
    prev_gm = np.full(S, np.inf)
    prev_pm = np.full(S, np.inf)

    frozen = False  # for L1 baseline: freeze MRAC after step 1

    for t in range(n_steps):
        time = t * dt

        # --- modification step ---
        if t > 0 and t % steps_per_mod == 0:
            k = t // steps_per_mod  # 1-indexed modification step

            # compute current margins (L2/L3 only — Q3)
            if level >= 2:
                # pass the LIVE adapted MRAC gain, not params.theta2_init
                gm, pm = compute_margins_vectorized(
                    episode_plant, params, S,
                    theta2=ctrl_state[:, TH2])
                gm_per_mod[k - 1] = gm
                pm_per_mod[k - 1] = pm
            else:
                gm = np.full(S, np.inf)
                pm = np.full(S, np.inf)

            # M5: self-modification is gated PER SEED, not by np.any(unstable).
            # The old global gate let one diverged seed halt self-modification
            # for the whole batch, which both coupled independent seeds and made
            # results depend on the batch size S.
            active = ~unstable
            if self_mod and level >= 2 and np.any(active):
                # epsilon-optimizer
                decision = epsilon_optimize(
                    x, ctrl_state, params, candidates,
                    exact_plant, agent_model, constraints, ref_fn, u_prev,
                    H, dt, gamma, w_e, w_u, agent_rng)

                # record only for seeds that have not diverged; a diverged seed
                # stops self-modifying and leaves NaN from here on
                nan_if_dead = np.where(active, 1.0, np.nan)
                eps_per_mod[k - 1] = decision['eps_emp'] * nan_if_dead
                in_sample_per_mod[k - 1] = decision['in_sample'] * nan_if_dead
                true_sel_per_mod[k - 1] = decision['true_sel'] * nan_if_dead
                true_best_per_mod[k - 1] = decision['true_best'] * nan_if_dead
                chosen_per_mod[k - 1] = np.where(active, decision['chosen'], -1)

                # value-up-margin-down detection
                val_up = decision['true_sel'] > decision['true_best'] * 0.99
                # actually: did the accepted mod improve in-sample value?
                # use: true_sel vs true_values[0] (no_change)
                no_change_val = decision['true_values'][0]
                val_up = decision['true_sel'] > no_change_val + 1e-10
                margin_down = (gm < prev_gm * 0.99) | (pm < prev_pm * 0.99)
                val_up_margin_down[k - 1] = val_up & margin_down

                # apply chosen modifications -- diverged seeds keep their params
                params = apply_choices(
                    params, candidates,
                    np.where(active, decision['chosen'], 0))

            elif not self_mod and level == 1 and not frozen:
                # L1 baseline: freeze MRAC after step 1
                params = params.copy()
                params.gamma_a = np.zeros(S)
                params.sigma = np.zeros(S)
                frozen = True

            prev_gm = gm.copy()
            prev_pm = pm.copy()

        # --- simulation step ---
        y = episode_plant.output(x)
        r = ref_fn(time)

        # freeze controller state for L1 baseline
        cs_eff = ctrl_state
        p_eff = params
        if frozen:
            p_eff = params  # gamma_a=0 already set

        u = controller_compute(time, cs_eff, y, r, p_eff)
        u_sat, sv, rv = constraints.apply(u, u_prev)
        du = u_sat - u_prev
        e = r - y

        r_t = reward(e[:, 0], du[:, 0], w_e, w_u)
        rewards[:, t] = r_t
        sat_viol[t] = sv[:, 0]
        rate_viol[t] = rv[:, 0]
        safety_viol[t] = constraints.safety_breach(y, r)[:, 0]

        # --- divergence check (replaces the unreachable |y| > 1e6 threshold) ---
        # Two criteria, each required to hold for `sustain_steps` consecutive
        # steps so that a bounded transient or a brief saturation excursion is
        # not mistaken for divergence:
        #   (a) adapted-parameter drift  |theta2| > THETA2_DIVERGE
        #   (b) sustained safety-bound breach |y - y_sp| > y_max
        # Non-finite state is treated as immediate divergence.
        # Warm-up: both plants start away from equilibrium, and the CSTR
        # baseline breaches its own |y - y_sp| <= 0.5 bound for steps 31..103
        # (0.31 s to 1.03 s) purely as a startup transient -- after t = 1 s the
        # breach fraction falls to 0.4%.  Detection therefore begins after the
        # first modification period.  The same window is applied to every level
        # and to both arms, so it cannot bias the paired comparison.
        warm = t >= steps_per_mod

        th2_now = np.abs(ctrl_state[:, TH2])
        drift_now = (th2_now > THETA2_DIVERGE) & warm
        breach_now = safety_viol[t] & warm
        drift_run = np.where(drift_now, drift_run + 1, 0)
        breach_run = np.where(breach_now, breach_run + 1, 0)

        blown = (~np.isfinite(y[:, 0]) | ~np.isfinite(ctrl_state[:, TH2])) & warm
        drift_div = drift_run >= sustain_steps
        breach_div = breach_run >= sustain_steps
        newly_unstable = drift_div | breach_div | blown

        new_mask = newly_unstable & ~unstable
        k_unstable[new_mask] = t // steps_per_mod if t > 0 else 0
        for lbl, m in (("param_drift", drift_div | blown),
                       ("safety_breach", breach_div)):
            sel = new_mask & m & (diverged_by == "")
            if np.any(sel):
                diverged_by[sel] = lbl
        unstable |= newly_unstable

        # integrate plant (RK4, ZOH on u_sat)
        def pdyn(tt, s):
            return episode_plant.dynamics(tt, s, u_sat, plant_rng)
        x = rk4_step(pdyn, time, x, dt)

        # integrate controller (RK4, ZOH on y, r, u_sat)
        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, p_eff)
        ctrl_state = rk4_step(cdyn, time, ctrl_state, dt)

        u_prev = u_sat.copy()

    # compute values at each modification step (every one is a real step now,
    # so no clipping is needed and every V_k spans a genuine episode remainder)
    mod_steps = np.arange(1, n_mod + 1) * steps_per_mod
    assert mod_steps.max() < n_steps, "phantom modification step"
    V_self = value_from_step_vec(rewards, mod_steps, gamma)

    # constraint summaries (cumulative up to each mod step)
    sat_frac = np.zeros((n_mod, S))
    rate_frac = np.zeros((n_mod, S))
    safety_frac = np.zeros((n_mod, S))
    for ki, step in enumerate(mod_steps):
        sat_frac[ki] = np.mean(sat_viol[:step + 1], axis=0)
        rate_frac[ki] = np.mean(rate_viol[:step + 1], axis=0)
        safety_frac[ki] = np.mean(safety_viol[:step + 1], axis=0)

    return dict(
        rewards=rewards,
        V=V_self,  # (S, n_mod) — value at each mod step
        mod_steps=mod_steps,
        n_mod=n_mod,
        eps=eps_per_mod,  # (n_mod, S)
        in_sample=in_sample_per_mod,
        true_sel=true_sel_per_mod,
        true_best=true_best_per_mod,
        chosen=chosen_per_mod,
        gm=gm_per_mod,
        pm=pm_per_mod,
        val_up_margin_down=val_up_margin_down,
        sat_frac=sat_frac,
        rate_frac=rate_frac,
        safety_frac=safety_frac,
        unstable=unstable,
        k_unstable=k_unstable,
        diverged_by=diverged_by,
        candidates=candidates,
        final_params=params,
    )


# ---------------------------------------------------------------------------
# Run a full configuration (self-mod + paired baseline)
# ---------------------------------------------------------------------------

def run_config(level, plant_type, gamma, H, sigma_n, delta_m,
               n_seeds=200, dt=0.01, T_episode=10.0, T_mod=1.0,
               w_e=1.0, w_u=0.1, ep_noise=0.01, base_seed=0,
               gamma_a=1.0, omega=16.1):
    """Run one configuration: self-mod + baseline, *n_seeds* paired.

    Returns a list of result-row dicts (one per seed × mod_step).
    """
    n_steps = int(T_episode / dt)
    steps_per_mod = int(T_mod / dt)

    config = dict(
        dt=dt, n_steps=n_steps, steps_per_mod=steps_per_mod,
        gamma=gamma, w_e=w_e, w_u=w_u, H=H,
        sigma_n=sigma_n, delta_m=delta_m, ep_noise=ep_noise,
        level=level, plant=plant_type,
        gamma_a=gamma_a, omega=omega,
    )
    chash = config_hash(config)

    # self-modifying run
    sm = run_episode(level, plant_type, config, n_seeds,
                     plant_seed=base_seed, agent_seed=base_seed + 999999,
                     self_mod=True)

    # baseline run (same plant noise, self-mod disabled)
    bl = run_episode(level, plant_type, config, n_seeds,
                     plant_seed=base_seed, agent_seed=base_seed + 999999,
                     self_mod=False)

    rows = []
    n_mod = sm['n_mod']
    for s in range(n_seeds):
        for k in range(n_mod):
            v_sm = sm['V'][s, k]
            v_bl = bl['V'][s, k]
            D = v_bl - v_sm

            eps = sm['eps'][k, s]
            if np.isnan(eps):
                # No epsilon-optimizer (L0/L1) — Theorem 7 doesn't apply
                bound = np.nan
                violated = False
            else:
                bound = theorem7_bound(eps, k + 1, gamma)
                violated = bool(D > bound + 1e-8)

            gm = sm['gm'][k, s]
            pm = sm['pm'][k, s]
            if np.isnan(gm):
                gm = np.inf
            if np.isnan(pm):
                pm = np.inf

            row = dict(
                config_hash=chash,
                level=level, plant=plant_type,
                gamma=gamma, H=H, sigma_n=sigma_n, delta_m=delta_m,
                gamma_a=gamma_a, omega=omega,
                seed=s, mod_step=k + 1,
                V_selfmod=v_sm, V_baseline=v_bl, D=D,
                eps_emp=eps,
                theorem7_bound=bound,
                bound_violated=violated,
                gain_margin=gm, phase_margin=pm,
                sat_frac=sm['sat_frac'][k, s],
                rate_frac=sm['rate_frac'][k, s],
                safety_frac=sm['safety_frac'][k, s],
                unstable=bool(sm['unstable'][s]),
                k_unstable=int(sm['k_unstable'][s]),
                diverged_by=str(sm['diverged_by'][s]),
                in_sample=sm['in_sample'][k, s],
                true_sel=sm['true_sel'][k, s],
                true_best=sm['true_best'][k, s],
                self_deception=(sm['in_sample'][k, s] -
                                sm['true_sel'][k, s])
                                if not np.isnan(sm['in_sample'][k, s])
                                else np.nan,
                val_up_margin_down=bool(sm['val_up_margin_down'][k, s]),
            )
            rows.append(row)

    return rows
