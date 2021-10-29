import numpy as np
import sys


class Grid1d(object):
    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, bc="periodic"):

        self.nx = nx
        self.ng = ng  # = number of ghost cells = order + 1

        self.xmin = xmin
        self.xmax = xmax

        self.bc = bc

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng  # where the 'real' data starts (outside of ghost cell region)
        self.ihi = ng + nx - 1  # where the 'real' data ends

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin) / (nx)
        self.x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * self.dx

        # Actual number of points in state variable / grid = nx + 2* ng
        # Since you have `ng` ghost cells on each side of domain

        # storage for the solution
        self.u = np.zeros((nx + 2 * ng), dtype=np.float64)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)

    def fill_BCs(self):
        """ fill all ghostcells as periodic """

        if self.bc == "periodic":

            # left boundary
            self.u[0 : self.ilo] = self.u[self.ihi - self.ng + 1 : self.ihi + 1]

            # right boundary
            self.u[self.ihi + 1 :] = self.u[self.ilo : self.ilo + self.ng]

        elif self.bc == "outflow":

            # left boundary
            self.u[0 : self.ilo] = self.u[self.ilo]

            # right boundary
            self.u[self.ihi + 1 :] = self.u[self.ihi]

        else:
            raise NotImplementedError
