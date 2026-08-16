"""Agent: candidate enumeration, noisy scoring, epsilon-optimizer selection,
and exact-rollout ground truth for the realised optimality gap.

The agent is an epsilon-optimizer in the Tětek sense: it scores each candidate
modification by a finite-horizon rollout of length *H* on a noisy internal
model (process noise *sigma_n*, model mismatch *delta_m*) and selects the
argmax of the NOISY score.  epsilon is induced by (H, sigma_n, delta_m)
rather than assumed.

After each decision we compute the TRUE value of every candidate by an exact
rollout on the real plant (no noise, no mismatch) and record the realised
optimality gap  = max(true_values) - true_value_of_selected.
"""

from __future__ import annotations

import numpy as np
from plants import rk4_step, RohrsPlant, CSTRPlant, Constraints
from controllers import (
    ControllerParams, controller_compute, controller_dynamics,
    controller_initial_state, get_candidates, Candidate,
)
from metrics import reward


# ---------------------------------------------------------------------------
# Rollout (vectorised across seeds)
# ---------------------------------------------------------------------------

def rollout(plant_state, ctrl_state, params, plant, constraints,
            ref_fn, u_prev, H, dt, gamma, w_e, w_u, rng=None,
            t_start=0.0):
    """Simulate *H* steps from the given state.  Returns discounted reward (S,).

    The plant is integrated with fixed-step RK4.  Control is held constant
    (zero-order hold) within each step — standard digital-control simulation.
    """
    S = plant_state.shape[0]
    x = plant_state.copy()
    cs = ctrl_state.copy()
    up = u_prev.copy()
    rewards = np.zeros(S)

    for h in range(H):
        t = t_start + h * dt
        y = plant.output(x)                       # (S, 1)
        r = ref_fn(t)                              # (S, 1)
        u = controller_compute(t, cs, y, r, params)  # (S, 1)
        u_sat, _, _ = constraints.apply(u, up)    # (S, 1)
        du = u_sat - up
        e = r - y

        r_t = reward(e[:, 0], du[:, 0], w_e, w_u)  # (S,)
        rewards += gamma**h * r_t

        # RK4 for plant (constant u_sat within step)
        def pdyn(tt, s):
            return plant.dynamics(tt, s, u_sat, rng)
        x = rk4_step(pdyn, t, x, dt)

        # RK4 for controller (constant y, r, u_sat within step)
        def cdyn(tt, s):
            return controller_dynamics(tt, s, y, r, u_sat, params)
        cs = rk4_step(cdyn, t, cs, dt)

        up = u_sat.copy()

    return rewards


# ---------------------------------------------------------------------------
# Agent model creation (mismatch via delta_m)
# ---------------------------------------------------------------------------

def make_agent_model(real_plant, delta_m, sigma_n, dt):
    """Create the agent's internal model, which may differ from the real plant.

    delta_m = 0  → perfect model (same plant, just add noise)
    delta_m > 0  → mismatched model (nominal plant or perturbed parameters)
    """
    if isinstance(real_plant, RohrsPlant):
        unmodeled = delta_m == 0.0  # perfect = include unmodeled block
        return RohrsPlant(unmodeled=unmodeled, dt=dt, noise_std=sigma_n)
    elif isinstance(real_plant, CSTRPlant):
        if delta_m == 0.0:
            return CSTRPlant(Da=real_plant.Da, B=real_plant.B,
                             gamma_arr=real_plant.gamma_arr,
                             beta=real_plant.beta,
                             c_in=real_plant.c_in,
                             theta_in=real_plant.theta_in,
                             dt=dt, noise_std=sigma_n)
        else:
            # perturb parameters by delta_m
            return CSTRPlant(
                Da=real_plant.Da * (1 + 0.3 * delta_m),
                B=real_plant.B * (1 - 0.2 * delta_m),
                gamma_arr=real_plant.gamma_arr,
                beta=real_plant.beta * (1 + 0.1 * delta_m),
                c_in=real_plant.c_in,
                theta_in=real_plant.theta_in,
                dt=dt, noise_std=sigma_n)
    else:
        raise TypeError(f"unknown plant type {type(real_plant)}")


# ---------------------------------------------------------------------------
# Epsilon-optimizer
# ---------------------------------------------------------------------------

def _stack_candidate_params(params, candidates, S):
    """Stack all candidate params into one ControllerParams with n_cand*S virtual seeds.

    Virtual seed index ci*S + s corresponds to (candidate ci, real seed s).
    """
    n_cand = len(candidates)
    fields = ['Kp', 'Ki', 'Kd', 'lead_tau', 'filter_tau',
              'integral_on', 'lead_on', 'filter_on',
              'mrac_on', 'gamma_a', 'sigma', 'dead_zone',
              'theta1_init', 'theta2_init', 'a_m']

    stacked = {}
    for f in fields:
        orig = getattr(params, f)
        dtype = orig.dtype if hasattr(orig, 'dtype') else type(orig)
        stacked[f] = np.zeros(n_cand * S, dtype=dtype)

    for ci, cand in enumerate(candidates):
        cp = cand.apply(params)
        sl = slice(ci * S, (ci + 1) * S)
        for f in fields:
            stacked[f][sl] = getattr(cp, f)

    return ControllerParams(
        S=n_cand * S,
        Kp=stacked['Kp'], Ki=stacked['Ki'], Kd=stacked['Kd'],
        lead_tau=stacked['lead_tau'], filter_tau=stacked['filter_tau'],
        integral_on=stacked['integral_on'], lead_on=stacked['lead_on'],
        filter_on=stacked['filter_on'], mrac_on=stacked['mrac_on'],
        gamma_a=stacked['gamma_a'], sigma=stacked['sigma'],
        dead_zone=stacked['dead_zone'],
        theta1_init=stacked['theta1_init'],
        theta2_init=stacked['theta2_init'], a_m=stacked['a_m'],
    )


def epsilon_optimize(plant_state, ctrl_state, params, candidates,
                     real_plant, agent_model, constraints, ref_fn, u_prev,
                     H, dt, gamma, w_e, w_u, agent_rng):
    """Evaluate all candidates, select argmax of noisy score, return decision.

    Vectorised across candidates: all candidates are stacked into one large
    rollout so that only TWO rollout calls are needed (one noisy, one exact)
    instead of 2*n_cand.
    """
    S = plant_state.shape[0]
    n_cand = len(candidates)

    # Stack all candidate params
    big_params = _stack_candidate_params(params, candidates, S)
    big_S = n_cand * S

    # Repeat state for all candidates
    big_plant_state = np.tile(plant_state, (n_cand, 1))
    big_ctrl_state = np.tile(ctrl_state, (n_cand, 1))
    big_u_prev = np.tile(u_prev, (n_cand, 1))

    # Reference function for the big batch
    def big_ref_fn(t):
        return np.repeat(ref_fn(t), n_cand, axis=0)

    # Noisy rollout on agent model (all candidates at once)
    noisy_flat = rollout(
        big_plant_state, big_ctrl_state, big_params, agent_model,
        constraints, big_ref_fn, big_u_prev,
        H, dt, gamma, w_e, w_u, rng=agent_rng)
    noisy_scores = noisy_flat.reshape(n_cand, S)

    # Exact rollout on real plant (all candidates at once, no noise)
    true_flat = rollout(
        big_plant_state, big_ctrl_state, big_params, real_plant,
        constraints, big_ref_fn, big_u_prev,
        H, dt, gamma, w_e, w_u, rng=None)
    true_values = true_flat.reshape(n_cand, S)

    # epsilon-optimizer: select argmax of NOISY score
    chosen = np.argmax(noisy_scores, axis=0)  # (S,)

    # realised optimality gap
    true_best = np.max(true_values, axis=0)  # (S,)
    true_sel = true_values[chosen, np.arange(S)]  # (S,)
    eps_emp = true_best - true_sel  # (S,)

    in_sample = noisy_scores[chosen, np.arange(S)]  # (S,)

    return dict(
        chosen=chosen,
        noisy_scores=noisy_scores,
        true_values=true_values,
        eps_emp=eps_emp,
        in_sample=in_sample,
        true_sel=true_sel,
        true_best=true_best,
    )


# ---------------------------------------------------------------------------
# Apply chosen modifications per seed
# ---------------------------------------------------------------------------

def apply_choices(params, candidates, chosen):
    """Apply the chosen candidate per seed to the params (in-place per seed).

    Since each seed may choose a different candidate, we need to apply
    per-seed modifications.  We do this by applying each candidate to a copy
    and then selecting per seed.
    """
    S = params.S
    # For efficiency, apply each candidate to a copy and merge
    # Since modifications are simple array operations, we can vectorise
    # by applying all candidates and then selecting per seed.

    # Start with no_change (candidate 0)
    result = params.copy()

    # For each candidate except no_change, apply to the seeds that chose it
    for ci, cand in enumerate(candidates):
        if cand.name == "no_change":
            continue
        mask = chosen == ci  # (S,) bool
        if not np.any(mask):
            continue
        cand_params = cand.apply(params)
        # copy the modified fields for these seeds
        for field_name in ['Kp', 'Ki', 'Kd', 'lead_tau', 'filter_tau',
                           'integral_on', 'lead_on', 'filter_on',
                           'gamma_a', 'sigma', 'dead_zone']:
            arr = getattr(result, field_name)
            cand_arr = getattr(cand_params, field_name)
            arr[mask] = cand_arr[mask]

    return result
