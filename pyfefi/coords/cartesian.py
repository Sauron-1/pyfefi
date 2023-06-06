from .coord import Coordinate

class Cartesian(Coordinate):

    @property
    def coords_independent(self):
        return True

    def to_cartesian(self, p, q, w):
        return p, q, w

    def from_cartesian(self, x, y, z):
        return x, y, z
