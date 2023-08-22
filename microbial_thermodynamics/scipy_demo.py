"""A script to demonstrate how to use scipy to integrate."""

# Import numpy to handle the vector stuff (call it np for short)
import matplotlib.pyplot as plt
import numpy as np

# This is only imported so that a type hint can be provided
from numpy.typing import NDArray

# Importing solve_ivp as the specific solver to use this might change in future
from scipy.integrate import solve_ivp

from microbial_thermodynamics.cell_growth_ODE import da, dc, dN, dr


def full_equation_set(
    t: float, y: NDArray[np.float32], number_of_species: int, reaction_energies: float
) -> NDArray[np.float32]:
    """Function that combines all equations together into one set that can be simulated.

    Args:
        t: Current time [seconds]. At present the model has no explicit time dependence,
            but the function must still be accept a time value to allow it to be
            integrated.
        pools: An array containing all soil pools in a single vector
        number_of_species: Number of sets of equations that we are integrating
        y: An array of all changes of the three equations.
        reaction_energies: the ATP generated for each species for each reaction alpha

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
        s=y[3 * number_of_species],
        w=y[3 * number_of_species + 1],
        reaction_energy=reaction_energies,
        v=np.array([1.0, 1.0]),
    )
    change_in_c = dc(
        c=y[3 * number_of_species : 4 * number_of_species],
        reaction_energy=reaction_energies,
        N=y[0:number_of_species],
        R=y[number_of_species : 2 * number_of_species],
        v=np.array([1.0, 1.0]),
    )
    # Then combine these changes into a single vector and return that
    return np.concatenate((change_in_N, change_in_r, change_in_a, change_in_c))


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
    c0 = [0, 0]

    # Construct vector of initial values y0
    y0 = np.concatenate((N0, r0, a0, c0))
    reaction_energies = np.array([1.0, 1.0])
    # Carry out simulation, by supplying the set of equations, time span, initial
    # condition, and any extra arguments
    output = solve_ivp(
        full_equation_set, t_span, y0, args=(number_of_species, reaction_energies)
    )
    print(t_span)
    print(y0)
    print(number_of_species)
    return output


def run_and_plot_population() -> None:
    """Runs the integrate function and plots population against time."""
    plot_time = integrate().t
    species_1_population = integrate().y[0]
    species_2_population = integrate().y[1]
    plt.plot(plot_time, species_1_population, label="species 1")
    plt.plot(plot_time, species_2_population, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Population change")
    plt.show()


def run_and_plot_r() -> None:
    """Runs the integrate function and plots ribosome fraction against time."""
    plot_time = integrate().t
    species_1_r = integrate().y[2]
    species_2_r = integrate().y[3]
    plt.plot(plot_time, species_1_r, label="species 1")
    plt.plot(plot_time, species_2_r, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Ribosome fraction")
    plt.title("Ribosome fraction change")
    plt.show()


def run_and_plot_a() -> None:
    """Integrate set of equations and plot internal energy (ATP) against time."""
    plot_time = integrate().t
    species_1_a = integrate().y[4]
    species_2_a = integrate().y[5]
    plt.plot(plot_time, species_1_a, label="species 1")
    plt.plot(plot_time, species_2_a, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Internal energy (ATP) production")
    plt.title("Internal energy (ATP) production change")
    plt.show()


integrate()
