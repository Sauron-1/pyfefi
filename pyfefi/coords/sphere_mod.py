"""
Modified spherical coordinates
"""
import numpy as np
import numba as nb
import math

from .coord import Coordinate

@nb.njit(inline='always', error_model='numpy')
def _fgrid(p):
    return (0.11*p  - math.erf((p-10e0)/12.) - math.erf((p+10e0)/12.))*100.*2./4./0.44

@nb.njit(inline='always', error_model='numpy')
def _fgrid_diff(p):
    return 100.*2./4./0.44 * (0.11 - 1/6/np.sqrt(np.pi) * (np.exp(-((p-10)/12)**2) + np.exp(-((p+10)/12)**2)))

#@nb.vectorize([nb.f8(nb.f8), nb.f4(nb.f4)], cache=True, target='parallel')
@nb.vectorize([nb.f8(nb.f8), nb.f4(nb.f4)], cache=True)
def fgrid(p):
    return _fgrid(p)

#@nb.vectorize([nb.f8(nb.f8), nb.f4(nb.f4)], cache=True, target='parallel')
@nb.vectorize([nb.f8(nb.f8), nb.f4(nb.f4)], cache=True)
def solve_grid(x):
    tol = 1e-6
    max_step = 100
    p = 2
    for i in range(max_step):
        xp = _fgrid(p)
        if abs(xp - x) < tol:
            break
        diff = _fgrid_diff(p)
        p -= (xp - x) / diff
    return p


class SphereM(Coordinate):

    @property
    def coords_independent(self):
        return True

    @property
    def should_save(self):
        return True

    def to_cartesian(self, p, q, w):
        r = fgrid(p)
        theta = q
        phi = w

        y = r * np.cos(theta)
        rho = r * np.sin(theta)
        x = rho * np.cos(phi)
        z = -rho * np.sin(phi)

        return x, y, z

    def from_cartesian(self, x, y, z):
        rho = np.hypot(x, z)
        r = np.hypot(rho, y)
        p = solve_grid(r)
        q = np.arctan2(rho, y)
        w = np.arctan2(-z, x)
        return p, q, w

    def convert_limits(self, limits):
        plim = solve_grid(limits[0])
        qlim = (lim*np.pi/180 for lim in limits[1])
        wlim = (lim*np.pi/180 for lim in limits[2])
        return (plim, qlim, wlim)
