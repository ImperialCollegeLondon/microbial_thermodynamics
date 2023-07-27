#forward Euler method to numerically solve a logistic growth model
#and then plots the solution using matplotlib
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def f(t: float, u: float) -> float:
    alpha = 0.2
    R = 1.0
    unext = alpha*(u) * (1-(u/R))
    return unext
u0 = 0.1
T = 40
N = 10


def forward_euler(f: Callable[[float, float], float], 
    u0: float, 
    T: float, 
    N: int
) -> tuple[list[float], list[float]]:
    """Solve u’=f(t, u), u(0)=u0, with n steps until t=T."""
    t = np.zeros(N + 1)
    u = np.zeros(N + 1) # u[n] is the solution at time t[n]
    u[0] = u0
    dt = T / N
    for n in range(N):
        t[n + 1] = t[n] + dt
        u[n + 1] = u[n] + dt * f(t[n], u[n])
    return t, u
    
t, u = forward_euler(f, u0, T, N)

plt.plot(t, u)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Model for logistic growth')
plt.show()
