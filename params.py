import numpy as np
import sys


class Params(object):
    def __init__(self, grid):
        self.grid = grid
        self.t = 0.0

    def init_cond(self, type="sine"):
        # Some initial conditions
        if type == "tophat":
            self.grid.u[
                np.logical_and(self.grid.x >= 0.333, self.grid.x <= 0.666)
            ] = 1.0
        elif type == "sine":
            self.grid.u[:] = 1.0

            index = np.logical_and(self.grid.x >= 0.333, self.grid.x <= 0.666)
            self.grid.u[index] += 0.5 * np.sin(
                2.0 * np.pi * (self.grid.x[index] - 0.333) / 0.333
            )
        elif type == "rarefaction":
            self.grid.u[:] = 1.0
            self.grid.u[self.grid.x > 0.5] = 2.0
        else:
            raise NotImplementedError

    def timestep(self, C):
        """
        Compute time step according to
        CFL number and maximum velocity 'u'
        """
        return (
            C * self.grid.dx / max(abs(self.grid.u[self.grid.ilo : self.grid.ihi + 1]))
        )
