"""A script to demonstrate how to use scipy to integrate."""

# Import numpy to handle the vector stuff (call it np for short)
import numpy as np
from cell_growth_ODE import da, dN, dr

# This is only imported so that a type hint can be provided
from numpy.typing import NDArray

# Importing solve_ivp as the specific solver to use this might change in future
from scipy.integrate import solve_ivp


def full_equation_set(
    t: float,
    y: NDArray[np.float32],
    number_of_species: int,
) -> NDArray[np.float32]:
    """Function that combines all equations together into one set that can be simulated.

    Args:
        t: Current time [seconds]. At present the model has no explicit time dependence,
            but the function must still be accept a time value to allow it to be
            integrated.
        pools: An array containing all soil pools in a single vector
        number_of_species: Number of sets of equations that we are integrating
        y: An array of all changes of the three equations.

    Returns:
        The solution for all equations in the set
    """

    # Extract the relevant values from the vector of values y, then supply them to the
    # relevant functions to calculate the change in each variable set

    change_in_N = dN(
        t,
        N=y[0:number_of_species],
        a=y[2 * number_of_species : 3 * number_of_species],
        R=y[number_of_species : 2 * number_of_species],
    )
    change_in_r = dr(
        t,
        a=y[2 * number_of_species : 3 * number_of_species],
        R=y[number_of_species : 2 * number_of_species],
    )
    change_in_a = da(
        t,
        a=y[2 * number_of_species : 3 * number_of_species],
        R=y[number_of_species : 2 * number_of_species],
    )

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
    number_of_species = 2

    # This means that each variable has 2 initial conditions
    N0 = [23.0, 2.3]
    r0 = [0.3, 0.3]
    a0 = [1e6, 1e6]

    # Construct vector of initial values y0
    y0 = np.concatenate((N0, r0, a0))
    print(y0)
    # Carry out simulation, by supplying the set of equations, time span, initial
    # condition, and any extra arguments
    output = solve_ivp(
        full_equation_set,
        t_span,
        y0,
        args=(number_of_species,),
    )

    return output
