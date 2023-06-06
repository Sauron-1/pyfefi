import numpy as np

def get_coord_from_id(id_num, config):
    match id_num:
        case None:
            from .cartesian import Cartesian
            return Cartesian()
        case 16:
            from .sphere_mod import SphereM
            return SphereM()
        case -13:
            from .cartesian_mod import CartesianM
            #from .cartesian_mod_old import CartesianM
            return CartesianM(config)
