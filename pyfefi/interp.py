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
        Data to be interpolated. The first 3 dimensions
        of the array must match `pqw`.

    Returns
    -------
    results: ndarray, or list of ndarray
        The first `len(pqw)` dims have same shape as elements in
        `coords`, other dimensions matches the last dimensions
        of `var_list`.
    """
    var_shape = tuple(len(c) for c in pqw)

    # check coords shape
    coords = list(coords)
    pqw = list(pqw)
    if len(coords) != len(pqw):
        raise ValueError('Different number of coordinates provided for `coords` and `pqw`')
    c_shape = coords[0].shape
    for c in coords[1:]:
        if c.shape != c_shape:
            raise ValueError('Shape of elements in `coords` must be the same')

    def convert(var):
        shape = var.shape
        if var.shape[:len(pqw)] != var_shape:
            raise ValueError('Shape mismatch for interp data and coordinates')
        var = var.reshape(var.shape[:len(pqw)] + (-1,))
        var_list = [var[..., i] for i in range(var.shape[len(pqw)])]
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
    for c in pqw:
        bases.append(c[0])
        scales.append((len(c)-1) / (c[-1]-c[0]))

    results = picinterp.gather(var_list, coords, bases, scales, order=order)

    ret = []
    for shape, index_range in zip(shapes, index_ranges):
        ret.append(
            np.stack(
                results[index_range[0]:index_range[1]], axis=-1
            ).reshape(tuple(c_shape) + shape[len(pqw):])
        )
    if return_one:
        return ret[0]
    else:
        return ret
