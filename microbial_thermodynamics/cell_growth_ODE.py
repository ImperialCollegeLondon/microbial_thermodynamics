"""Summary:
This calculates rates of change of population growth, ribosome fraction and
internal energy production.

Desctiption:
I am uncertain if the R in calculate_lam and the r in dr are the same thing (ribosome
fraction). Should I find values for gam and R or write functions.
"""  # noqa: D205

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def calculate_r_star(a: float, om: float = 1e9, Q: float = 0.45) -> float:
    """Calculates ideal ribosome fraction.

    Args:
       a: internal energy (ATP) concentration
       om: Ribosome fraction half saturation constant
       Q: housekeeping fraction
    """
    u = (a * (1 - Q)) / (a + om)
    return u


def calculate_lam(gam: float, R: float, m: float = 1e8, fb: float = 0.7) -> float:
    """Calculates species growth.

    Args:
     gam: Effective translation elongation rate
     R: Ribosome fraction
     m: Cell mass
     fb: Average fraction of ribosomes bound
    """
    u = (gam * fb * R) / m
    return u


def dN(tN: float, N: float, R: float, d: float = 0.1) -> float:
    """Rate of change of population growth.

    Args:
     tN: time
     lam: species’ growth rate
     N: current population
     d: rate of biomass loss.
     R: ribosome fraction
    """
    lam = calculate_lam(1, R)  # 1 is just there untill i find a real value
    udN = (lam - d) * N
    return udN


def dr(tr: float, t: float, a: float, r: float) -> float:
    """Rate of change of ribosome fration.

    Args:
     tr: time
     t: characteristic time scale for growth
     r_star: ideal ribosome fraction
     r: Ribosome fraction?
     a: internal energy (ATP) concentration.
    """
    r_star = calculate_r_star(a)
    udr = (1 / t) * (r_star * a - r)
    return udr


def da(ta: float, a: float, j: float = 2e5, chi: float = 29, m: float = 1e8) -> float:
    """Rate of change of internal energy (ATP) production.

    Args:
    ta: time
    j: ATP production rate
    chi: is ATP use per elongation step
    m: total mass of the cell (in units of amino acids)
    a: internal energy (ATP) concentration
    lam: species’ growth rate.
    """
    lam = calculate_lam(1, 2)  # 1 and 2 here until I find values for gam and R
    uda = j - chi * m * lam - a * lam
    return uda


def forward_euler(
    dN: Callable[[float, float, float], float],
    dr: Callable[[float, float, float, float], float],
    da: Callable[[float, float], float],
    u0: float,
    T: float,
    N: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
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
    t = np.zeros(N + 1)
    u_N = np.zeros(N + 1)
    u_r = np.zeros(N + 1)
    u_a = np.zeros(N + 1)
    u_N[0] = u_r[0] = u_a[0] = u0
    dt = T / N
    for n in range(N):
        t[n + 1] = t[n] + dt
        u_N[n + 1] = u_N[n] + dt * dN(t[n], u_N[n], u_r[n])
        u_r[n + 1] = u_r[n] + dt * dr(
            t[n], 12, u_a[n], 34
        )  # dont know values of t and r
        u_a[n + 1] = u_a[n] + dt * da(t[n], u_a[n])
    return t, u_N, u_a, u_r


def run_and_plot_N(u0: float = 0.1, T: float = 40, N: int = 10) -> None:
    """Runs the forward euler equation then plots it."""
    t, u_N, u_a, u_r = forward_euler(dN, dr, da, u0, T, N)

    plt.plot(t, u_N)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("population abundance")
    plt.show()


run_and_plot_N()
