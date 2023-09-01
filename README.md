# `microbial_thermodynamics`

This repository contains code for simulating thermodynamically limited microbial ecosystems.

TODO - Flesh this out with details of previous work on the problem.

This is currently being developed by Jack Richards and Jacob Cook.

This model looked at the competition between 2 species of bacteria for 2 metabolites. In
each bacteria species the population, ribosome fraction and internal ATP concentration
were considered. The metabolite concentration was represented by the dc function in the
cell_growth_ODE file. The population was represented by the dN function, ribosome
fraction was the dr function and the internal ATP concentration was the da function -
all in the cell_growth_ODE file.

The dN function called the lambda function (which calculates species growth) which calls
the gamma function (which calculates cellular growth rate). The dr function called
r_star (ideal ribosome fraction) and time_scale (characteristic time scale for growth).
The da function called lambda and j (rate of ATP production in a species). j called
function q (reaction rate). The q function contained reaction kinetic constants and
called functions E (enzyme copy number) and theta (a thermodynamic factor that stops a
reaction at equilibrium). This theta function called kappa which contained  the
thermodynamic constants: Gibbs free-energy per mole of ATP, standard Gibbs free-energy
change when one mole of reaction alpha occurs, gas constant and temperature. This means
that thermodynamic considerations went into calculating internal energy (ATP)
concentration. Since internal energy (ATP) concentration is a parameter of dN, the
population will be affected by thermodynamic constraints.

The dN, dr, da, dc equations were then integrated using the solve_ivp solver form the
scipy.integrate subpackage. The model was ran for 200,000 seconds. Species 1 always had
the amount of ATP generated for each species for each reaction (reaction_energy) set
to 1. The model was ran twice - when species 2 had generated 2 moles and 1.5 moles of
ATP per reaction.

When reaction_energy was set to 1.5 in species 2, the population of species 2 grew at a
faster rate than species 1 - species 2 outcompeted species 1 for resources. When
reaction_energy was set to 2 in species 2, the population of species 1 grew at a faster
rate than species 2 - species 1 outcompeted species 2 for resources.

This shows that the thermodynamic constrains in this model lead to a limit to the amount
of ATP that can be produced per reaction before it becomes unfavourable to the species'
survival.
