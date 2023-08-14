"""Test module fo scipy_demo."""
import numpy as np
import pytest


def test_full_equation_set():
    """Tests function that combines all equations into one set that can be simulated."""
    from microbial_thermodynamics.scipy_demo import full_equation_set

    # Set of values to test
    t = 0
    y = np.array([2.3e01, 2.3e00, 3.0e-01, 3.0e-01, 1.0e06, 1.0e06])
    number_of_species = 2

    # Set of expected values to test against
    expected_full_equation_set = np.array(
        [
            -2.29999988e00,
            -2.29999988e-01,
            4.36539265e-07,
            4.36539265e-07,
            1.99984679e05,
            1.99984679e05,
        ]
    )

    # Calculate values using the function
    actual_full_equation_set = full_equation_set(t, y, number_of_species)

    # Check that actual and expected values match
    assert actual_full_equation_set == pytest.approx(expected_full_equation_set)
