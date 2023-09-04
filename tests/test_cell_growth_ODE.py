"""Test module for cell_growth_ODE.py.

This module tests the functionality of the cell_growth_ODE script.
"""

import numpy as np
import pytest


def test_calculate_gam():
    """Test calculation of the cellular growth rate."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_gam

    # Set of values to test
    a = np.array([1e5, 1e6, 1e7, 1e8])

    # Set of expected values to test against
    expected_gam = [0.00419916, 0.04191616, 0.41176470, 3.5]

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
    expected_time_scale = [1.404944348e8, 9.935103130e6, 687723.190585, 67423.8412510]

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
    expected_lam = [4.728910e-8, 6.687254e-7, 9.660654e-6, 9.853867e-5]

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
    expected_dN = [-0.0001199054, -0.0002966563, -0.0005033934, 0.00088638959]

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
    expected_dr = [-8.5373492e-10, -1.70557413e-8, -3.55600129e-7, -3.70788723e-6]

    # Calculate rate of change of ribosome fraction values using the function
    actual_dr = dr(t, a, R)

    # Check that actual and expected values match
    assert actual_dr == pytest.approx(expected_dr)


def test_calculate_E():
    """Test Calculation of enzyme copy number."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_E

    # Set of values to test
    v = np.array([1.0, 0.7, 0.5, 0.1])
    R = np.array([0.12, 0.17, 0.25, 0.3])

    # Set of expected values to test against
    expected_E = [143333.3333, 88666.66666, 50000.0, 8333.33333]

    # Calculate enzyme copy number values using the function
    actual_E = calculate_E(v, R)

    # Check that actual and expected values match
    assert actual_E == pytest.approx(expected_E)


def test_calculate_kappa():
    """Test calculation of the kappa constant."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_kappa

    # Set of values to test
    reaction_energy = np.array([0.3333, 1.0, 1.5, 2.0])

    # Set of expected values to test against
    expected_kappa = [1.8803993202e22, 2.31357064720e13, 4.8099590925e6, 1.0]

    # Calculate kappa values using the function
    actual_kappa = calculate_kappa(reaction_energy)

    # Check that actual and expected values match
    assert actual_kappa == pytest.approx(expected_kappa)


@pytest.mark.parametrize(
    "s,w,reaction_energy,expected_theta",
    [
        (10.3, 5.4, 0.3333, 2.794975776e-23),
        (2e3, 3e3, 1.0, 6.493090256e-14),
        (4e4, 3e4, 1.5, 1.5592648202e-7),
        (8e5, 3e5, 2.0, 0.375),
    ],
)
def test_calculate_theta(s, w, reaction_energy, expected_theta):
    """Test calculation of a thermodynamic factor that stops reaction at equilibrium."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_theta

    # Calculate thermodynamic factor values using the function
    actual_theta = calculate_theta(reaction_energy, s, w)

    # Check that actual and expected values match
    assert actual_theta == pytest.approx(expected_theta)


@pytest.mark.parametrize(
    "s,w,reaction_energy,R,v,expected_q",
    [
        (10.3, 5.4, 0.3333, 0.12, 1.0, 1.433142015e6),
        (2e3, 3e3, 1.0, 0.17, 0.7, 886666.0570),
        (4e4, 3e4, 1.5, 0.25, 0.5, 499999.1252),
        (8e5, 3e5, 2.0, 0.3, 0.1, 10964.91227),
    ],
)
def test_q(s, w, reaction_energy, R, v, expected_q):
    """Test reaction rate calculation."""
    from microbial_thermodynamics.cell_growth_ODE import q

    # Calculate reaction rate values using the function
    actual_q = q(R, s, w, reaction_energy, v)

    # Check that actual and expected values match
    assert actual_q == pytest.approx(expected_q)


@pytest.mark.parametrize(
    "s,w,reaction_energy,R,v,expected_j",
    [
        (10.3, 5.4, 0.3333, 0.12, 1.0, 477666.2337),
        (2e3, 3e3, 1.0, 0.17, 0.7, 886666.0570),
        (4e4, 3e4, 1.5, 0.25, 0.5, 749998.6878),
        (8e5, 3e5, 2.0, 0.3, 0.1, 21929.82454),
    ],
)
def test_calculate_j(s, w, reaction_energy, R, v, expected_j):
    """Tests calculation of rate of ATP production in a species."""
    from microbial_thermodynamics.cell_growth_ODE import calculate_j

    # Calculate change of ATP production in a species values using the function
    actual_j = calculate_j(R, s, w, v, reaction_energy)

    # Check that actual and expected values match
    assert actual_j == pytest.approx(expected_j)


def test_dc():
    """Test calculation of change in metabolite concentration."""
    from microbial_thermodynamics.cell_growth_ODE import dc

    # Set of values to test
    c = np.array([12, 23])
    N = np.array([2, 5, 10, 23])
    R = np.array([0.12, 0.17, 0.25, 0.3])
    reaction_energy = np.array([0.3333, 1.0, 1.5, 2.0])
    v = np.array([1.0, 0.7, 0.5, 0.1])

    expected_dc = [-0.0064770305, -0.01104]

    actual_dc = dc(c=c, reaction_energy=reaction_energy, N=N, R=R, v=v)

    # Check that actual and expected values match
    assert actual_dc == pytest.approx(expected_dc)


@pytest.mark.parametrize(
    "s,w,expected_da",
    [
        (10.3, 5.4, [477529.1, 884608.4, 721786.5, -282915.44]),
        (2e3, 3e3, [477592.53, 884726.06, 721884.44, -300824.38]),
        (4e4, 3e4, [477592.84, 884726.7, 721886.2, -290714.06]),
        (8e5, 3e5, [477592.84, 884726.7, 721886.9, -273686.22]),
    ],
)
def test_da(s, w, expected_da):
    """Test calculation of rate of change of internal energy (ATP) production."""
    from microbial_thermodynamics.cell_growth_ODE import da

    # Set of values to test
    t = np.array([10, 20, 30, 40])
    a = np.array([1e5, 1e6, 1e7, 1e8])
    R = np.array([0.12, 0.17, 0.25, 0.3])
    reaction_energy = np.array([0.3333, 1.0, 1.5, 2.0])
    v = np.array([1.0, 0.7, 0.5, 0.1])

    # Calculate change of internal energy (ATP) production values using the function
    actual_da = da(t, a, R, s=s, w=w, reaction_energy=reaction_energy, v=v)

    # Check that actual and expected values match
    assert actual_da == pytest.approx(expected_da)
