"""Forward Euler method to numerically solve a logistic growth model.

This then plots the solution using matplotlib.
"""
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def f(t: float, u: float) -> float:
    """The logistic growth model function."""
    alpha = 0.2
    R = 1.0
    unext = alpha * (u) * (1 - (u / R))
    return unext


def forward_euler(
    f: Callable[[float, float], float], u0: float, T: float, N: int
) -> tuple[list[float], list[float]]:
    """Solve u’=f(t, u), u(0)=u0 (in this case the logistic growth model).

    This uses N steps until t=T.

    Args:
        f:
            The differential equation being solved
        u0:
            The initial value or the value of the dependent variable at the initial
            time (or initial point) of the ODE
        T:
            The maximum time
        N:
            The number of time steps used

    Returns:
        An array for t and an array for u. These represent an approximated solution to
        an ODE.
    """
    t = np.zeros(N + 1)
    u = np.zeros(N + 1)  # u[n] is the solution at time t[n]
    u[0] = u0
    dt = T / N
    for n in range(N):
        t[n + 1] = t[n] + dt
        u[n + 1] = u[n] + dt * f(t[n], u[n])
    return t, u


def run_and_plot(u0: float = 0.1, T: float = 40, N: int = 10) -> None:
    """Runs the forward euler equation then plots it."""
    t, u = forward_euler(f, u0, T, N)

    plt.plot(t, u)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Model for logistic growth")
    plt.show()
