"""
Modified cartesian coordinates
"""
import numpy as np
import numba as nb
import math
import functools

if __name__ == "__main__":
    from coord import Coordinate
    from lib.cartesian_mod import CartesianModCoref
else:
    from .coord import Coordinate
    from .lib.cartesian_mod import CartesianModCoref

class CartesianM(Coordinate):

    def __init__(self, config):
        self.config = config

        dx0 = self.config.names['input_parameters']['dx0']
        dy0 = self.config.names['input_parameters']['dy0']
        dz0 = self.config.names['input_parameters']['dz0']
        lims = np.array(self.config.limits, dtype=np.float32).reshape(3, 2)
        deltas = np.array([dx0, dy0, dz0], dtype=np.float32)
        self.kernel = CartesianModCoref(deltas, lims)

        self._grid_size = self.kernel.grid_sizes()
        self.config.grid_size = self._grid_size

    @property
    def grid_size(self):
        return self._grid_size

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
        #print(p.strides, q.strides, w.strides)
        return self.kernel.to_cartesian(p, q, w)

    def from_cartesian(self, x, y, z):
        x, y, z = np.broadcast_arrays(x, y, z)
        x = np.require(x, dtype=np.float32)
        y = np.require(y, dtype=np.float32)
        z = np.require(z, dtype=np.float32)
        #print(x.strides, y.strides, z.strides)
        return self.kernel.from_cartesian(x, y, z)

    def convert_limits(self, limits):
        gs = self._grid_size
        return (
                (0, gs[0]-1),
                (0, gs[1]-1),
                (0, gs[2]-1))
      

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    class DummyConfig:
        def __init__(self):
            self.names = {'input_parameters': {
                'dx0': 0.012,
                'dy0': 0.012,
                'dz0': 0.012}}
            self.limits = (
                (-7, 3),
                (-3, 3),
                (-5, 5)
            )
    cm = CartesianM(DummyConfig())
    print(cm.grid_size)
    p = np.arange(cm.grid_size[0])
    x, _, _ = cm.to_cartesian(p, 10, 10)
    plt.plot(p, x)
    print(x.min(), x.max())
    print(np.diff(x).max())
    p1, _, _ = cm.from_cartesian(x, 0, 0)
    plt.plot(p, p1)
    plt.plot(p, p)
    print(np.max(np.abs(p - p1)), np.min(np.diff(p)))
    print(cm.convert_limits(DummyConfig().limits))
    plt.show()
