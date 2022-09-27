import numpy as np

import ufl
import dolfinx
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

L = 1.
alpha = 1e-6