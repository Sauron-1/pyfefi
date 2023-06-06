import numpy as np
from abc import ABC, abstractmethod

class Coordinate:

    def __init__(self, config=None):
        pass

    @property
    def coords_independent(self):
        """
        If the coordinates are independent (i.e. orthogonal).
        """
        return False

    @property
    def should_save(self):
        """
        Whether should data reader try to save grid points.
        Should return True when converting is expensive.
        """
        return False

    def to_cartesian(self, p, q, w):
        """
        Convert from the coordinate system to cartesian.
        """
        raise NotImplementedError

    def from_cartesian(self, x, y, z):
        """
        Convert from cartesian to the coordinate system.
        """
        raise NotImplementedError

    def convert_limits(self, limits):
        """
        Convert input limits to limits in p, q, w coords
        """
        return (
            self.from_cartesian(limits[0], limits[1][0], limits[2][0])[0],
            self.from_cartesian(limits[0][0], limits[1], limits[2][0])[1],
            self.from_cartesian(limits[0][0], limits[1][0], limits[2])[2])
