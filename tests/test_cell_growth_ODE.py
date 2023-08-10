"""Test module for cell_growth_ODE.py.

This module tests the functionality of the differential_equations2 dummy script.
"""

import numpy as np
import pytest


def test_calculate_gam():
    """Test calculation of the cellular growth rate."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_gam

    # Set of values to test
    a = np.array([1e5, 1e6, 1e7, 1e8])

    # Set of expected values to test against
    expected_gam = [0.25194961, 2.51497006, 24.70588235, 210.0]

    # Calculate gamma values using the function
    actual_gam = calculate_gam(a)

    # Check that actual and expected values match
    assert actual_gam == pytest.approx(expected_gam)


def test_calculate_time_scale():
    """Test calculation of the characteristic time scale of proteome shifting."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_time_scale

    # Set of values to test
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_time_scale = [
        31392597132.524,
        2219935941.7614627,
        153667422.0764222,
        15065433.536904138,
    ]

    # Calculate time scale of proteome shifting values using the function
    actual_time_scale = calculate_time_scale(a, R)

    # Check that actual and expected values match
    assert actual_time_scale == pytest.approx(expected_time_scale)


def test_calculate_r_star():
    """Test calculation of the ideal ribosome fraction."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_r_star

    # Set of values to test
    a = np.array([1e5, 1e6, 1e7, 1e8])

    # Set of expected values to test against
    expected_r_star = [
        5.4994500549945015e-05,
        0.0005494505494505495,
        0.005445544554455445,
        0.05000000000000001,
    ]

    # Calculate  ideal ribosome fraction values using the function
    actual_time_scale = calculate_r_star(a)

    # Check that actual and expected values match
    assert actual_time_scale == pytest.approx(expected_r_star)


def test_calculate_lam():
    """Test calculation of species growth."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_lam

    # Set of values to test
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_lam = [
        2.1163767246550686e-10,
        2.9928143712574845e-09,
        4.323529411764706e-08,
        4.41e-07,
    ]

    # Calculate species growth values using the function
    actual_lam = calculate_lam(a, R)

    # Check that actual and expected values match
    assert actual_lam == pytest.approx(expected_lam)


def test_dN():
    """Test calculation of rate of change of population growth."""
    from microbial_thermodynamics.cell_growth_ODE import dN

    # Set of values to test
    t = np.array([10, 20, 30, 40])
    N = np.array([2, 5, 10, 23])
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_dN = [
        -0.19999999957672468,
        -0.4999999850359282,
        -0.9999995676470588,
        -2.299989857,
    ]

    # Calculate rate of change of population growth values using the function
    actual_dN = dN(t, N, a, R)

    # Check that actual and expected values match
    assert actual_dN == pytest.approx(expected_dN)


def test_dr():
    """Test calculation of rate of change of ribosome fration."""
    from microbial_thermodynamics.cell_growth_ODE import dr

    # Set of values to test
    t = np.array([10, 20, 30, 40])
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_dr = [
        1.7136046540797907e-10,
        2.474308105551502e-07,
        0.0003543704632298229,
        0.3318855503064051,
    ]

    # Calculate rate of change of ribosome fration values using the function
    actual_dr = dr(t, a, R)

    # Check that actual and expected values match
    assert actual_dr == pytest.approx(expected_dr)


def test_da():
    """Test calculation of rate of change of internal energy (ATP) production."""
    from microbial_thermodynamics.cell_growth_ODE import da

    # Set of values to test
    t = np.array([10, 20, 30, 40])
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_da = [199999.38622958606, 199991.31784550898, 199874.18529411766, 198677.0]

    # Calculate change of internal energy (ATP) production values using the function
    actual_da = da(t, a, R)

    # Check that actual and expected values match
    assert actual_da == pytest.approx(expected_da)
