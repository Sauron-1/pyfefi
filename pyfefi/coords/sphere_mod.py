"""
Modified spherical coordinates
"""
import numpy as np
import numba as nb
import math

from .coord import Coordinate
from .lib.sphere_mod import SphereModCoref, solve_grid
kernel = SphereModCoref()

class SphereM(Coordinate):

    @property
    def coords_independent(self):
        return True

    @property
    def should_save(self):
        return True

    def to_cartesian(self, p, q, w):
        p, q, w = np.broadcast_arrays(p, q, w)
        p = np.require(p, dtype=np.float32)
        q = np.require(q, dtype=np.float32)
        w = np.require(w, dtype=np.float32)
        return kernel.to_cartesian(p, q, w)

    def from_cartesian(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        x = np.require(x, dtype=np.float32)
        y = np.require(y, dtype=np.float32)
        z = np.require(z, dtype=np.float32)
        return kernel.from_cartesian(x, y, z)

    def convert_limits(self, limits):
        plim = (solve_grid(limits[0][0]), solve_grid(limits[0][1]))
        qlim = (lim*np.pi/180 for lim in limits[1])
        wlim = (lim*np.pi/180 for lim in limits[2])
        return (plim, qlim, wlim)
