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
