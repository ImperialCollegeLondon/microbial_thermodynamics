"""A script to demonstrate how to use scipy to integrate."""

# Import numpy to handle the vector stuff (call it np for short)
import numpy as np
from cell_growth_ODE import (
    calculate_gam,
    calculate_lam,
    calculate_r_star,
    calculate_time_scale,
    da,
    dN,
    dr,
)

# This is only imported so that a type hint can be provided
from numpy.typing import NDArray

# Importing solve_ivp as the specific solver to use this might change in future
from scipy.integrate import solve_ivp


def full_equation_set(
    t: float,
    N: NDArray[np.float32],
    a: NDArray[np.float32],
    R: NDArray[np.float32],
    y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Function that combines all equations together into one set that can be simulated.

    Args:
        t: Current time [seconds]. At present the model has no explicit time dependence,
            but the function must still be accept a time value to allow it to be
            integrated.
        pools: An array containing all soil pools in a single vector
        N: Number of sets of equations that we are integrating
        a: internal energy (ATP) concentration
        R: Ribosome fraction
        y: An array of all changes of the three equations.

    Returns:
        The solution for all equations in the set
    """

    # Extract the relevant values from the vector of values y, then supply them to the
    # relevant functions to calculate the change in each variable set
    calculate_time_scale(a, R)
    calculate_r_star(a)
    calculate_lam(a, R)
    calculate_gam(a)

    change_in_N = dN(t, N=y[0:N], a=y[3 * N : 4 * N], R=y[N : 2 * N])
    change_in_r = dr(t, a=y[N : 2 * N], R=y[2 * N : 3 * N])
    change_in_a = da(t, a=y[N : 2 * N], R=y[2 * N : 3 * N])

    # Then combine these changes into a single vector and return that
    return np.concatenate((change_in_N, change_in_r, change_in_a))


def integrate() -> NDArray[np.float32]:
    """Integrate the model using solve_ivp.

    Returns:
        These results of the integration.
    """

    # Set the update time (in seconds)
    update_time = 100.0
    # Use this to make the time span (starts at time = 0s)
    t_span = (0.0, update_time)

    # In this case we simulate 2 lots of the same set of equations
    N = 2

    # This means that each variable has 2 initial conditions
    N0 = [23.0, 2.3]
    r0 = [33.1, 3.0]
    a0 = [17.1, 1.7]

    # Construct vector of initial values y0
    y0 = np.concatenate((N0, r0, a0))

    # Carry out simulation, by supplying the set of equations, time span, initial
    # condition, and any extra arguments
    output = solve_ivp(
        full_equation_set,
        t_span,
        y0,
        args=(N,),
    )

    return output
