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
    t: float,
    y: NDArray[np.float32],
    number_of_species: int,
    reaction_energies: NDArray[np.float32],
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
    update_time = 200000.0
    # Use this to make the time span (starts at time = 0s)
    t_span = (0.0, update_time)

    # In this case we simulate 2 lots of the same set of equations
    number_of_species = 2

    # This means that each variable has 2 initial conditions
    N0 = [20.3, 20.3]
    r0 = [0.3, 0.3]
    a0 = [1e6, 1e6]
    c0 = [2000, 2000]

    # Construct vector of initial values y0
    y0 = np.concatenate((N0, r0, a0, c0))
    reaction_energies = np.array([1.0, 1.5])
    # Carry out simulation, by supplying the set of equations, time span, initial
    # condition, and any extra arguments
    output = solve_ivp(
        full_equation_set, t_span, y0, args=(number_of_species, reaction_energies)
    )

    return output


def run_and_plot_population() -> None:
    """Run the integrate function and plots population against time."""
    output = integrate()
    plot_time = output.t
    species_1_population = output.y[0]
    species_2_population = output.y[1]
    plt.plot(plot_time, species_1_population, label="species 1")
    plt.plot(plot_time, species_2_population, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Population change")
    plt.legend()
    plt.show()


def run_and_plot_r() -> None:
    """Run the integrate function and plots ribosome fraction against time."""
    output = integrate()
    plot_time = output.t
    species_1_r = output.y[2]
    species_2_r = output.y[3]
    plt.plot(plot_time, species_1_r, label="species 1")
    plt.plot(plot_time, species_2_r, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Ribosome fraction")
    plt.title("Ribosome fraction change")
    plt.show()


def run_and_plot_a() -> None:
    """Integrate set of equations and plot internal energy (ATP) against time."""
    output = integrate()
    plot_time = output.t
    species_1_a = output.y[4]
    species_2_a = output.y[5]
    plt.plot(plot_time, species_1_a, label="species 1")
    plt.plot(plot_time, species_2_a, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Internal energy (ATP) production")
    plt.title("Internal energy (ATP) production change")
    plt.show()


def run_and_plot_c() -> None:
    """Plot metabolite concentration against time."""
    output = integrate()
    plot_time = output.t
    species_1_c = output.y[6]
    species_2_c = output.y[7]
    plt.plot(plot_time, species_1_c, label="species 1")
    plt.plot(plot_time, species_2_c, label="species 2")
    plt.xlabel("Time")
    plt.ylabel("Metabolite concentration")
    plt.title("Metabolite concentration change")
    plt.show()


def plot_lambda() -> None:
    """Plot labda from integrate output."""
    output = integrate()
    plot_time = output.t
    # Find a and R values for species 1 for the whole simulation
    species_1_a = output.y[4]
    species_1_R = output.y[2]
    # Use the a and R values to calculate lambda from the function you already defined
    from microbial_thermodynamics.cell_growth_ODE import calculate_lam

    species_1_lambda = calculate_lam(a=species_1_a, R=species_1_R)
    # Then plot lambda against time
    plt.plot(plot_time, species_1_lambda)
    plt.xlabel("Time")
    plt.ylabel("lambda")
    plt.show()


def plot_j() -> None:
    """Plot j from integrate output."""
    output = integrate()
    plot_time = output.t
    # Find a and R values for species 1 for the whole simulation
    species_1_R = output.y[2]
    metabolite_1 = output.y[6]
    metabolite_2 = output.y[7]
    # Use the a and R values to calculate lambda from the function you already defined
    from microbial_thermodynamics.cell_growth_ODE import calculate_j

    # Create empty vector to store j value for each time point in plot_time
    species_1_j = np.zeros(len(plot_time))
    for ind, _ in enumerate(plot_time):
        species_1_j[ind] = calculate_j(
            R=species_1_R[ind],
            s=metabolite_1[ind],
            w=metabolite_2[ind],
            v=1.0,
            reaction_energy=1.0,
        )
    plt.plot(plot_time, species_1_j)
    plt.xlabel("Time")
    plt.ylabel("j")
    plt.show()
