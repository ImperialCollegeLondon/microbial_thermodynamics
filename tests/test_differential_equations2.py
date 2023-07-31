"""Test module for differential_equations2.py.

This module tests the functionality of the differential_equations2 dummy script.
"""
import pytest


@pytest.mark.parametrize(
    "u,expected_value", [(0.1, 0.018), (1.0, 0.0), (10.0, -18.0), (25.0, -120.0)]
)
def test_f(u, expected_value):
    """Test the function to be integrated 'f'."""
    from microbial_thermodynamics.differential_equations2 import f

    calculated_value = f(t=0.0, u=u)

    # Approx needed as floating point numbers won't be exactly the same
    assert expected_value == pytest.approx(calculated_value)


@pytest.mark.parametrize(
    "u0, T, N, expected_t, expected_u",
    [
        (
            0.1,
            40,
            10,
            [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            [
                0.1,
                0.172,
                0.2859328,
                0.44927299,
                0.6472144,
                0.82987674,
                0.94282181,
                0.98594889,
                0.99703183,
                0.99939932,
                0.99987957,
            ],
        )
    ],
)
def test_forward_euler(u0, T, N, expected_t, expected_u):
    """Test that the forward euler function works."""
    from microbial_thermodynamics.differential_equations2 import f, forward_euler

    t, u = forward_euler(f, u0, T, N)
    assert t == pytest.approx(expected_t)
    assert u == pytest.approx(expected_u)
