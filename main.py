import grid
import matplotlib.pyplot as plt
from sim import Simulation

# Problem of interest
problem = "burgers"

# Time integrator
integrator = "RK4"

# Numerical flux function
num_flux = "GLF"

# IC type
ic = "sine"

# Grid bounds
xmin = 0.0
xmax = 1.0

# Number of spatial points
nx = 32

# Order of accuracy
order = 3
ng = order + 1
g = grid.Grid1d(nx, ng, bc="periodic")

# Maximum evolution time based on period for unit velocity
tmax = (xmax - xmin) / 1.0

# CFL number: 0 < C < 1
C = 0.5

s = Simulation(g, C, order, problem, num_flux, integrator)

# Set initial condition
s.init_cond(ic)

# Evolve simulation in time until t = tmax
### s.evolve(tmax)

plt.clf()

for i in range(0, 10):
    tend = (i + 1) * 0.02 * tmax
    s.init_cond(ic)

    uinit = s.grid.u.copy()

    s.evolve(tend)

    c = 1.0 - (0.1 + i * 0.1)
    g = s.grid
    plt.plot(g.x[g.ilo : g.ihi + 1], g.u[g.ilo : g.ihi + 1], color=str(c))

g = s.grid
plt.plot(
    g.x[g.ilo : g.ihi + 1], uinit[g.ilo : g.ihi + 1], ls=":", color="0.9", zorder=-1
)

plt.xlabel("$x$")
plt.ylabel("$u$")
plt.savefig("sine.pdf")
