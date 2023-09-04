"""Test module fo scipy_demo."""
import numpy as np
import pytest


def test_full_equation_set():
    """Tests function that combines all equations into one set that can be simulated."""
    from microbial_thermodynamics.scipy_demo import full_equation_set

    # Set of values to test
    t = 0
    y = np.array([2.3e01, 2.3e00, 3.0e-01, 3.0e-01, 1.0e06, 1.0e06, 1.0, 2.0])
    number_of_species = 2
    reaction_energies = [1.0, 1.5]

    # Set of expected values to test against
    expected_full_equation_set = np.array(
        [
            -0.00135285760962,
            -0.00013528576096,
            -5.3189406871e-08,
            -5.3189406871e-08,
            828765.5625,
            1244854.375,
            -0.00029835000168,
            -0.00048000001697,
        ]
    )

    # Calculate values using the function
    actual_full_equation_set = full_equation_set(
        t, y, number_of_species, reaction_energies
    )

    # Check that actual and expected values match
    assert actual_full_equation_set == pytest.approx(expected_full_equation_set)
