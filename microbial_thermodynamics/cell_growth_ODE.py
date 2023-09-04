"""Module to calculate cell growth rates for the ODE model.

This calculates rates of change of population growth, ribosome fraction and
internal energy production.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def calculate_gam(
    a: NDArray[np.float32], ym: float = 21, y_half: float = 5e8
) -> NDArray[np.float32]:
    """Calculate the cellular growth rate.

    Args:
     a: internal energy (ATP) concentration
     ym: Maximum elongation rate (1260 steps per minute = 21 per second)
     y_half: Elongation half saturation constant
    """
    u = (ym * a) / (y_half + a)
    return u


def calculate_time_scale(
    a: NDArray[np.float32], R: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Calculate characteristic time scale for growth.

    Args:
     a: internal energy (ATP) concentration
     R: Ribosome fraction
    """
    lam = calculate_lam(a, R)
    u = (1 / lam) * (np.log10(100) / np.log10(2))
    return u


def calculate_r_star(
    a: NDArray[np.float32], om: float = 1e9, Q: float = 0.45
) -> NDArray[np.float32]:
    """Calculate ideal ribosome fraction.

    Args:
       a: internal energy (ATP) concentration
       om: Ribosome fraction half saturation constant
       Q: housekeeping fraction
    """
    u = (a * (1 - Q)) / (a + om)
    return u


def calculate_lam(
    a: NDArray[np.float32], R: NDArray[np.float32], nr: float = 7459, fb: float = 0.7
) -> NDArray[np.float32]:
    """Calculate species growth.

    Args:
     a: internal energy (ATP) concentration
     R: Ribosome fraction
     nr: Average ribosome mass (amino acids)
     fb: Average fraction of ribosomes bound
    """
    gam = calculate_gam(a)
    u = (gam * fb * R) / nr
    return u


def dN(
    tN: float,
    N: NDArray[np.float32],
    a: NDArray[np.float32],
    R: NDArray[np.float32],
    d: float = 6e-5,
) -> NDArray[np.float32]:
    """Rate of change of population growth.

    Args:
     tN: time
     lam: species' growth rate
     N: current population
     a: internal energy (ATP) concentration
     d: rate of biomass loss.
     R: ribosome fraction
    """
    lam = calculate_lam(a, R)
    udN = (lam - d) * N
    return udN


def dr(
    tr: float, a: NDArray[np.float32], R: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Rate of change of ribosome fration.

    Args:
     tr: time
     r_star: ideal ribosome fraction
     R: Ribosome fraction
     a: internal energy (ATP) concentration.
    """
    time_scale = calculate_time_scale(a, R)
    r_star = calculate_r_star(a)
    udr = (1 / time_scale) * (r_star - R)

    return udr


def calculate_E(
    v: float, R: float, Q: float = 0.45, m: float = 1e8, avg_enzyme_mass: float = 300
) -> float:
    """Calculate enzyme copy number.

    Args:
     v: the ith species' proportional expression level for reaction alpha
     R: Ribosome fraction
     Q: Housekeeping proteome fraction
     m: Cell mass
     avg_enzyme_mass: Average metabolic protein mass
    """
    u = (m * v * (1 - R - Q)) / avg_enzyme_mass

    return u


def calculate_kappa(
    reaction_energy: float,
    Gatp: float = 75000,
    G0: float = -1.5e5,
    R: float = 8.314,
    T: float = 293.15,
) -> float:
    """Calculate the kappa constant.

    Args:
     reaction_energy: the ATP generated for each species for each reaction alpha
     Gatp: ATP free energy
     G0: the standard Gibbs free-energy change when one mole of reaction alpha occurs
     R: the gas constant
     T: temperature
    """
    u = np.exp(((-G0) - (reaction_energy * Gatp)) / (R * T))
    return u


def calculate_theta(reaction_energy: float, s: float, w: float) -> float:
    """Calculate the thermodynamic factor that stops a reaction at equilibrium.

    Args:
     reaction_energy: the ATP generated for each species for each reaction alpha
     s: substrate concentration
     w: waste product concentration
    """
    if s <= 0.0:
        u = 1.0
    else:
        kappa = calculate_kappa(reaction_energy)
        u = (w / s) / kappa
    return u


def q(
    R: float,
    s: float,
    w: float,
    reaction_energy: float,
    v: float,
    ka: float = 10,
    ks: float = 1.375e-3,
    ra: float = 10,
) -> float:
    """Calculate reaction rate.

    Args:
     R: Ribosome fraction
     s: substrate concentration
     w: waste product concentration
     reaction_energy: the ATP generated for each species for each reaction alpha
     v: the ith species' proportional expression level for reaction alpha
     ka: the maximum forward rate for reaction alpha for the ith species
     ks: substrate half saturation constant
     ra: reversibility factor for reaction alpha
    """
    E = calculate_E(v, R)
    theta = calculate_theta(reaction_energy, s, w)
    u = (ka * E * s * (1 - theta)) / (ks + s * (1 + ra * theta))

    return u


def calculate_j(
    R: float,
    s: float,
    w: float,
    v: float,
    reaction_energy: float,
    Gatp: float = 75000,
) -> float:
    """Calculate rate of ATP production in a species.

    Args:
     R: Ribosome fraction
     s: substrate concentration
     w: waste product concentration
     reaction_energy: the ATP generated for each species for each reaction alpha
     v: the ith species' proportional expression level for reaction alpha
     Gatp: ATP free energy
    """
    u = reaction_energy * q(R, s, w, reaction_energy, v)

    return u


def da(
    ta: float,
    a: NDArray[np.float32],
    R: NDArray[np.float32],
    s: float,
    w: float,
    reaction_energy: NDArray[np.float32],
    v: NDArray[np.float32],
    chi: float = 29,
    m: float = 1e8,
) -> NDArray[np.float32]:
    """Calculate rate of change of internal energy (ATP) production.

    Args:
    ta: time
    a: internal energy (ATP) concentration
    R: Ribosome fraction
    s: substrate concentration
    w: waste product concentration
    reaction_energy: the ATP generated for each species for each reaction alpha
    v: the ith species' proportional expression level for reaction alpha
    chi: is ATP use per elongation step
    m: total mass of the cell (in units of amino acids)
    """
    a_changes = np.zeros(len(a), dtype=np.float32)
    for ind, _ in enumerate(a):
        j = calculate_j(R[ind], s, w, v[ind], reaction_energy[ind])
        lam = calculate_lam(a[ind], R[ind])
        a_changes[ind] = j - chi * m * lam - a[ind] * lam

    return a_changes


def dc(
    c: NDArray[np.float32],
    reaction_energy: NDArray[np.float32],
    N: NDArray[np.float32],
    R: NDArray[np.float32],
    v: NDArray[np.float32],
    k: float = 3.3e-7,
    p: float = 6e-5,
    avogadros_number: float = 6.022e23,
) -> NDArray[np.float32]:
    """Calculate the change in metabolite concentration.

    Args:
     c: metabolite concentration
     reaction_energy: the ATP generated for each species for each reaction alpha
     N: current population
     R: Ribosome fraction
     v: the ith species' proportional expression level for reaction alpha
     k: substrate supply rate
     p: metabolite dilution rate
     avogadros_number: atoms per mole
    """
    c_changes = np.zeros(len(c), dtype=np.float32)
    for ind, _ in enumerate(c):
        if ind == 0:
            c_changes[ind] = (
                k
                - p * c[ind]
                - N[ind]
                * q(R[ind], c[0], c[1], reaction_energy[ind], v[ind])
                / avogadros_number
            )
        for metabolite_num, _ in enumerate(c):  # loops over all metabolites
            # loops over species to calculate metabolite contribution for each species
            for species_num, _ in enumerate(N):
                if metabolite_num == 0:
                    c_changes[metabolite_num] += (
                        k
                        - p * c[metabolite_num]
                        - N[species_num]
                        * q(
                            R[species_num],
                            c[0],
                            c[1],
                            reaction_energy[species_num],
                            v[species_num],
                        )
                        / avogadros_number
                    )
                else:
                    c_changes[metabolite_num] += (
                        -p * c[metabolite_num]
                        + N[species_num]
                        * q(
                            R[species_num],
                            c[0],
                            c[1],
                            reaction_energy[species_num],
                            v[species_num],
                        )
                        / avogadros_number
                    )
    return c_changes


def forward_euler(
    dN: Callable[[float, float, float, float], float],
    dr: Callable[[float, float, float], float],
    da: Callable[[float, float, float], float],
    u0: float,
    T: float,
    N: int,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Solve u’=f(t, u), u(0)=u0 with n steps until t=T.

    Args:
        dN:
            differential equation for rate of change of population growth
        dr:
            differential equation for rate of change of ribosome fration
        da:
            differential equation for rate of change of internal energy (ATP) production
        u0:
            The initial value or the value of the dependent variable at the initial
            time (or initial point) of the ODE
        T:
            The maximum time
        N:
            The number of time steps used

    Returns:
        An array for t and an array for u for each of the three functions
        These represent an approximated solution to the three ODEs.
    """
    t = np.zeros(N + 1, dtype=np.float32)
    u_N = np.zeros(N + 1, dtype=np.float32)
    u_r = np.zeros(N + 1, dtype=np.float32)
    u_a = np.zeros(N + 1, dtype=np.float32)
    u_N[0] = u_r[0] = u_a[0] = u0
    dt = T / N
    for n in range(N):
        t[n + 1] = t[n] + dt
        u_N[n + 1] = u_N[n] + dt * dN(t[n], u_N[n], u_a[n], u_r[n])
        u_r[n + 1] = u_r[n] + dt * dr(t[n], u_a[n], u_r[n])
        u_a[n + 1] = u_a[n] + dt * da(t[n], u_a[n], u_r[n])
    return t, u_N, u_a, u_r


"""
def run_and_plot_N(u0: float = 0.1, T: float = 40, N: int = 10) -> None:
    Runs the forward euler equation then plots it.
    t, u_N, u_a, u_r = forward_euler(dN, dr, da, u0, T, N)

    plt.plot(t, u_N)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("population abundance")
    plt.show()
"""
