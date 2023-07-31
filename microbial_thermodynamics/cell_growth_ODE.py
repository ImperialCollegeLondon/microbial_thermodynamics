from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def dN(tN: float, uN: float) -> float:
    """rate of change of population growth"""
    lam = 1 #species’ growth rate
    N = 2 #population abundance
    d = 10 #rate of biomass loss
    udN = (lam - d)*N
    return udN
def dr(tr: float, ur: float) -> float:
    """rate of change of ribosome fration"""
    t = 3 # characteristic time scale for growth
    r = 4 #ribosome fraction
    r_star = 5 #ideal ribosome fraction
    a = 5 #internal energy (ATP) concentration
    udr = (1 / t)*(r_star*a - r)
    return udr
def da(ta: float, ua: float) -> float:
    """rate of change of internal energy (ATP) production"""
    j = 7 #ATP production rate
    chi = 8 #is ATP use per elongation step
    m = 9 # total mass of the cell (in units of amino acids)
    a = 5 #internal energy (ATP) concentration
    lam = 1 #species’ growth rate
    uda = j - chi*m*lam - a*lam
    return uda


def forward_euler(dN: Callable[[float, float], float],
                  dr: Callable[[float, float], float],
                  da: Callable[[float, float], float],
                  u0: float,
                  T: float,
                  N: int
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
    u_N = np.zeros(N + 1)  # u1[n] is the solution for function 1 at time t[n]
    u_r = np.zeros(N + 1)  # u2[n] is the solution for function 2 at time t[n]
    u_a = np.zeros(N + 1)  # u3[n] is the solution for function 3 at time t[n]
    u_N[0] = u_r[0] = u_a[0] = u0
    dt = T / N
    for n in range(N):
        t[n + 1] = t[n] + dt
        u_N[n + 1] = u_N[n] + dt * dN(t[n], u_N[n])
        u_a[n + 1] = u_a[n] + dt * da(t[n], u_a[n])
        u_r[n + 1] = u_r[n] + dt * dr(t[n], u_r[n])
    return t, u_N, u_a, u_r



def run_and_plot_N(u0: float = 0.1, T: float = 40, N: int = 10) -> None:
    """Runs the forward euler equation then plots it"""
    t, u_N, u_a, u_r = forward_euler(dN, dr, da, u0, T, N)

    plt.plot(t, u_N)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('population abundance')
    plt.show()

run_and_plot_N()

