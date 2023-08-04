"""A script to demonstrate how to use scipy to integrate."""

# Import numpy to handle the vector stuff (call it np for short)
import numpy as np

# This is only imported so that a type hint can be provided
from numpy.typing import NDArray

# Importing solve_ivp as the specific solver to use this might change in future
from scipy.integrate import solve_ivp


# These three functions are just here as examples, they really could be anything
def da(
    b: NDArray[np.float32], c: NDArray[np.float32], d: float = 0.5
) -> NDArray[np.float32]:
    """Calculate the change in the a variables.

    Args:
        b: the b variables
        c: the c variables
        d: a multiplicative constants

    Returns:
        The rate of change of the a variables
    """

    # Multiplication can be done using vectors, not just floats. This is element wise,
    # i.e. the first element in b is multiplied by the first element in c, and this
    # results in a vector.
    return b * c * d


def db(
    a: NDArray[np.float32], c: NDArray[np.float32], e: float = -0.5
) -> NDArray[np.float32]:
    """Calculate the change in the b variables.

    Args:
        a: the a variables
        c: the c variables
        e: a multiplicative constants

    Returns:
        The rate of change of the b variables
    """

    return a * c * e


def dc(
    a: NDArray[np.float32], b: NDArray[np.float32], f: float = 1.0
) -> NDArray[np.float32]:
    """Calculate the change in the c variables.

    Args:
        a: the a variables
        b: the b variables
        f: a multiplicative constants

    Returns:
        The rate of change of the c variables
    """

    return a * b * f


def full_equation_set(
    t: float,
    y: NDArray[np.float32],
    N: int,
) -> NDArray[np.float32]:
    """Function that combines all equations together into one set that can be simulated.

    Args:
        t: Current time [seconds]. At present the model has no explicit time dependence,
            but the function must still be accept a time value to allow it to be
            integrated.
        pools: An array containing all soil pools in a single vector
        N: Number of sets of equations that we are integrating
        y: does something

    Returns:
        The solution for all equations in the set
    """

    # Extract the relevant values from the vector of values y, then supply them to the
    # relevant functions to calculate the change in each variable set
    change_in_a = da(b=y[N : 2 * N], c=y[2 * N : 3 * N])
    change_in_b = db(a=y[0:N], c=y[2 * N : 3 * N])
    change_in_c = dc(a=y[0:N], b=y[N : 2 * N])

    # Then combine these changes into a single vector and return that
    return np.concatenate((change_in_a, change_in_b, change_in_c))


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
    a0 = [23.0, 2.3]
    b0 = [33.1, 3.0]
    c0 = [17.1, 1.7]

    # Construct vector of initial values y0
    y0 = np.concatenate((a0, b0, c0))

    # Carry out simulation, by supplying the set of equations, time span, initial
    # condition, and any extra arguments
    output = solve_ivp(
        full_equation_set,
        t_span,
        y0,
        args=(N,),
    )

    return output
