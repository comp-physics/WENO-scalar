# WENO Scalar

This solves a 1D Scalar PDE via the finite volume method.
The implemented example is Burgers equation for a given grid resolution, time and space domain, initial condition, etc. 
WENO reconstructs the state variable and global Lax-Fridrichs is used for the numerical flux.

Run via `python main.py`
