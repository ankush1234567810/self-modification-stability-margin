"""Regression test for _eval_tf_on_grid coefficient ordering (defect C1).

The original implementation reversed the coefficient arrays before calling
``np.polyval``.  Since the arrays are already in decreasing-power order (the
standard control convention), that evaluated the *reciprocal-reversed*
polynomial instead of the intended one.

The bug was invisible in the existing test suite because the only tested plant
was the nominal first-order model G(s) = 2/(s+1), whose denominator [1, 1] is a
PALINDROME.  Every test below therefore uses a NON-PALINDROMIC denominator so
that a reintroduced ``[::-1]`` fails loudly.
"""
import numpy as np
import control as ct
import pytest

from metrics import _eval_tf_on_grid, _tf_coeffs_rohrs
from plants import RohrsPlant


def test_eval_tf_non_palindromic_denominator():
    """A hand-checked non-palindromic case: 1 / (s^2 + 3s + 2)."""
    num = np.array([1.0])
    den = np.array([1.0, 3.0, 2.0])  # reversed would be 2s^2 + 3s + 1
    omega = np.array([0.5, 1.0, 2.0, 7.0])
    s = 1j * omega

    got = _eval_tf_on_grid(num, den, omega)
    expected = 1.0 / (s**2 + 3.0 * s + 2.0)
    wrong = 1.0 / (2.0 * s**2 + 3.0 * s + 1.0)

    assert np.allclose(got, expected), f"got {got}, expected {expected}"
    # guard: the two must actually differ, or the test proves nothing
    assert not np.allclose(expected, wrong), "denominator is palindromic"
    assert not np.allclose(got, wrong), "coefficients are being reversed"


def test_eval_tf_matches_python_control_on_rohrs():
    """The full Rohrs loop: 458 / (s^3 + 31 s^2 + 259 s + 229).

    Denominator [1, 31, 259, 229] is strongly non-palindromic, so this is the
    case the original bug corrupted.
    """
    num, den = _tf_coeffs_rohrs(unmodeled=True)
    assert list(den) != list(den[::-1]), "test requires a non-palindromic den"

    omega = np.logspace(-2, 3, 200)
    got = _eval_tf_on_grid(num, den, omega)

    G = ct.tf(num, den)
    ref = np.array([complex(G(1j * w)) for w in omega])

    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_eval_tf_matches_plant_transfer_function():
    """_tf_coeffs_rohrs must agree with RohrsPlant.tf_full()."""
    num, den = _tf_coeffs_rohrs(unmodeled=True)
    omega = np.logspace(-2, 3, 100)

    got = _eval_tf_on_grid(num, den, omega)
    G = RohrsPlant.tf_full()
    ref = np.array([complex(G(1j * w)) for w in omega])

    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_dc_gain_is_two():
    """Sanity anchor: G(0) = 2 for the Rohrs plant (458/229).

    The reversed polynomial has G(0) = 458/1 = 458, so this alone catches it.
    """
    num, den = _tf_coeffs_rohrs(unmodeled=True)
    dc = _eval_tf_on_grid(num, den, np.array([1e-9]))[0]
    assert abs(dc.real - 2.0) < 1e-6, f"DC gain {dc} should be 2.0"
