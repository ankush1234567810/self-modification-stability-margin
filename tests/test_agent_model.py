"""Regression tests for the agent's internal model (defect M11).

Two bugs corrupted the agent's actual decisions, not just its reporting:

1. With ``delta_m > 0`` the agent model was ``RohrsPlant(unmodeled=False)`` with
   ``n_states = 1``, but the rollout seeded it with the 3-state real plant
   state.  ``rk4_step`` then evaluated ``(S,3) + (dt/6)*(S,1)``, which NumPy
   BROADCASTS -- x2 and x3 silently received x1's derivative.
2. The nominal model's ``output`` returned x1 while the real plant returns x2,
   so the agent scored candidates against a different signal from the one it was
   controlling.

Both are now fixed by keeping the state layout at 3 and the measurement at x2.
"""
import numpy as np
import control as ct
import pytest

from plants import RohrsPlant, rk4_step
from agent import make_agent_model


DT = 0.01


def _step(plant, n=1000, u_val=1.0):
    x = plant.initial_state(1)
    u = np.full((1, 1), u_val)
    ys = []
    for t in range(n):
        ys.append(plant.output(x)[0, 0])
        x = rk4_step(lambda a, b: plant.dynamics(a, b, u), t * DT, x, DT)
    return np.array(ys)


def test_state_layouts_are_compatible():
    """Nominal model and full plant must share a state layout."""
    full = RohrsPlant(unmodeled=True, dt=DT)
    nom = RohrsPlant(unmodeled=False, dt=DT)
    assert full.n_states == nom.n_states == 3


def test_no_silent_broadcast_in_rollout():
    """dynamics() must return the same width as the state it was given.

    If it returns (S,1) for an (S,3) state, rk4_step broadcasts instead of
    failing, which is what corrupted the agent's model.
    """
    for unmodeled in (True, False):
        plant = RohrsPlant(unmodeled=unmodeled, dt=DT)
        state = np.zeros((7, 3))
        ds = plant.dynamics(0.0, state, np.ones((7, 1)))
        assert ds.shape == state.shape, \
            f"unmodeled={unmodeled}: dynamics returned {ds.shape} for {state.shape}"


def test_both_models_measure_the_same_state_variable():
    """output() must read x2 for the nominal model as well as the full plant."""
    state = np.array([[1.0, 2.0, 3.0]])
    assert RohrsPlant(unmodeled=True).output(state)[0, 0] == 2.0
    assert RohrsPlant(unmodeled=False).output(state)[0, 0] == 2.0


def test_nominal_model_reproduces_first_order_response():
    """The nominal model's output must follow G(s) = 2/(s+1)."""
    ys = _step(RohrsPlant(unmodeled=False, dt=DT))
    t = np.arange(len(ys)) * DT
    exact = 2.0 * (1.0 - np.exp(-t))
    assert np.isfinite(ys).all(), "nominal model diverged (passthrough too fast)"
    assert np.abs(ys - exact).max() < 0.05
    assert abs(ys[-1] - 2.0) < 1e-3


def test_full_plant_response_unchanged():
    """The real plant must still match tf_full() exactly."""
    ys = _step(RohrsPlant(unmodeled=True, dt=DT))
    t = np.arange(len(ys)) * DT
    _, ref = ct.step_response(RohrsPlant.tf_full(), T=t)
    assert np.abs(ys - ref).max() < 1e-5


def test_agent_model_rollout_is_state_compatible():
    """A mismatched agent model must accept the real plant's state directly."""
    real = RohrsPlant(unmodeled=True, dt=DT, noise_std=0.0)
    model = make_agent_model(real, delta_m=0.5, sigma_n=0.0, dt=DT)
    assert model.n_states == real.n_states

    x_real = np.array([[0.4, 0.7, -0.2]])
    ds = model.dynamics(0.0, x_real, np.ones((1, 1)))
    assert ds.shape == x_real.shape
    # and the model reads the same measurement the real plant would
    assert model.output(x_real)[0, 0] == real.output(x_real)[0, 0]
