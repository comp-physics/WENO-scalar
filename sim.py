import numpy as np
import params
import weno

class Simulation(params.Params):
    def __init__(
        self,
        grid,
        C=0.5,
        weno_order=3,
        problem="burgers",
        num_flux="GLF",
        integrator="RK4",
    ):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C  # CFL number
        self.weno_order = weno_order
        self.problem = problem
        self.num_flux = num_flux
        self.time_stepper = integrator

    def physical_flux(self, q):
        if self.problem == "burgers":
            return 0.5 * q ** 2
        elif self.problem == "advection":
            return q
        else:
            raise NotImplementedError

    def numerical_flux(self, f, u):
        """ Compute numerical flux function """
        if self.num_flux == "GLF":
            # Global Lax-Fridrichs
            alpha = np.max(abs(u))
            fp = (f + alpha * u) / 2
            fm = (f - alpha * u) / 2
            return fp, fm
        else:
            raise NotImplementedError

    def substep(self):
        """ Compute RHS """
        g = self.grid
        g.fill_BCs()

        f = self.physical_flux(g.u)
        fp, fm = self.numerical_flux(f, g.u)
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[1:] = weno.reconstruct(self.weno_order, fp[:-1])
        fml[-1::-1] = weno.reconstruct(self.weno_order, fm[-1::-1])
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:])
        return rhs

    def time_integrator(self, g, dt):
        if self.time_stepper == "RK4":
            u_start = g.u.copy()
            k1 = dt * self.substep()
            g.u = u_start + k1 / 2
            k2 = dt * self.substep()
            g.u = u_start + k2 / 2
            k3 = dt * self.substep()
            g.u = u_start + k3
            k4 = dt * self.substep()
            g.u = u_start + (k1 + 2 * (k2 + k3) + k4) / 6
        else:
            raise NotImplementedError

        return g.u

    def evolve(self, tmax):
        self.t = 0.0
        g = self.grid

        # main evolution loop
        while self.t < tmax:
            # fill boundary conditions (e.g. periodic)
            g.fill_BCs()

            # get timestep size
            dt = self.timestep(self.C)

            # adjust step size if needed
            if self.t + dt > tmax:
                dt = tmax - self.t

            g.u = self.time_integrator(g, dt)
            self.t += dt

        return g.u
