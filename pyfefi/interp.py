import warnings
import numpy as np
try:
    import picinterp
except ImportError as e:
    print('Package "picinterp" is not installed. Please download and install it')
    print('from https://github.com/Sauron-1/picinterp.')
    raise(e)

def interp(pqw_to, pqw_from, var_list, order=2):
    """
    Interpolate using picinterp.gather.

    Parameters
    ----------
    pqw_to : list of ndarray
        Curve-linear coordinates to interpolate into.

    pqw_from : list of 1D-ndarray
        Curve-linear coordinates of the original data.
        Only array length and the first/last elements are
        used to obtain starting coordinate and step size.
        Must be unifomly spaced, otherwise the result will
        be wrong.

    var_list : ndarray, or list of ndarray
        Data to be interpolated. The first 3 dimensions
        of the array must match `pqw_from`.

    Returns
    -------
    results: ndarray, or list of ndarray
        The first `len(pqw_from)` dims have same shape as elements in
        `pqw_to`, other dimensions matches the last dimensions
        of `var_list`.
    """
    var_shape = tuple(len(c) for c in pqw_from)

    # check pqw_to shape
    pqw_to = list(pqw_to)
    pqw_from = list(pqw_from)
    if len(pqw_to) != len(pqw_from):
        raise ValueError('Different number of coordinates provided for `pqw_to` and `pqw_from`')
    c_shape = pqw_to[0].shape
    for c in pqw_to[1:]:
        if c.shape != c_shape:
            raise ValueError('Shape of elements in `pqw_to` must be the same')

    def convert(var):
        shape = var.shape
        if var.shape[:len(pqw_from)] != var_shape:
            raise ValueError('Shape mismatch for interp data and coordinates')
        var = var.reshape(var.shape[:len(pqw_from)] + (-1,))
        var_list = [var[..., i] for i in range(var.shape[len(pqw_from)])]
        return var_list, shape

    if isinstance(var_list, np.ndarray):
        return_one = True
        var_list, shape = convert(var_list)
        shapes = [shape]
        index_ranges = [(0, len(var_list))]
    else:
        return_one = False
        _vl = []
        shapes = []
        index_ranges = []
        for var in var_list:
            vl, shape = convert(var)
            shapes.append(shape)
            index_ranges.append((len(_vl), len(_vl)+len(vl)))
            _vl.extend(vl)
        var_list = _vl

    bases = []
    scales = []
    for c in pqw_from:
        bases.append(c[0])
        scales.append((len(c)-1) / (c[-1]-c[0]))

    results = picinterp.gather(var_list, pqw_to, bases, scales, order=order)

    ret = []
    for shape, index_range in zip(shapes, index_ranges):
        ret.append(
            np.stack(
                results[index_range[0]:index_range[1]], axis=-1
            ).reshape(tuple(c_shape) + shape[len(pqw_from):])
        )
    if return_one:
        return ret[0]
    else:
        return ret
