import warnings
import numpy as np
try:
    import picinterp
except ImportError as e:
    print('Package "picinterp" is not installed. Please download and install it')
    print('from https://github.com/Sauron-1/picinterp.')
    raise(e)

def interp(coords, pqw, var_list, order=2):
    """
    Interpolate using picinterp.gather.

    Parameters
    ----------
    coords : list of ndarray
        Curve-linear coordinates to interpolate into.

    pqw : list of 1D-ndarray
        Curve-linear coordinates of the original data.
        Only array length and the first/last elements are
        used to obtain starting coordinate and step size.
        Must be unifomly spaced, otherwise the result will
        be wrong.

    var_list : ndarray, or list of ndarray
        Data to be interpolated. Can be one of the following
        three forms:
        1. list of ndarray, each array must have `ndim == len(pqw)`.
        2. ndarray with `ndim == len(pqw)` and `shape == coords[0].shape`.
        3. ndarray with `ndim > len(pqw)` and `shape[:len(pqw)] == coords[0].shape`.

    Returns:
    results: ndarray, or list of ndarray
        The first `len(pqw)` dims have same shape as elements in
        `coords`.
    """
    # check coords shape
    coords = list(coords)
    pqw = list(pqw)
    if len(coords) != len(pqw):
        raise ValueError('Different number of coordinates provided for `coords` and `pqw`')
    c_shape = coords[0].shape
    for c in coords[1:]:
        if c.shape != c_shape:
            raise ValueError('Shape of elements in `coords` must be the same')
    if isinstance(var_list, np.ndarray):
        if var_list.ndim > len(pqw):
            shape = var_list.shape
            var_list = var_list.reshape(var_list.shape[:len(pqw)] + (-1,))
            var_list = [var_list[..., i] for i in var_list.shape[len(pqw)]]
            kind = 'vec_arr'
        else:
            var_list = [var_list]
            kind = 'arr'
    else:
        kind = 'arr_list'

    var_shape = tuple(len(c) for c in pqw)
    if var_list[0].ndim != len(pqw):
        raise ValueError('Dimension mismatch for interp data and coordinates')
    if var_list[0].shape != var_shape:
        warnings.warn('Shape mismatch for interp data and coordinates.')

    bases = []
    scales = []
    for c in pqw:
        bases.append(c[0])
        scales.append((len(c)-1) / (c[-1]-c[0]))

    results = picinterp.gather(var_list, coords, bases, scales, order=order)

    if kind == 'arr_list':
        return results
    if kind == 'arr':
        return results[0]
    if kind == 'vec_arr':
        result = np.stack(results, axis=-1)
        return result.reshape(shape)
