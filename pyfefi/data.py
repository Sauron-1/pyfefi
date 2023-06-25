import numpy as np
import netCDF4
import numba as nb
import math
import f90nml
import os
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from .coords import get_coord_from_id

#@nb.guvectorize([(nb.f8[:], nb.f8[:, :], nb.f8[:]), (nb.f4[:], nb.f4[:, :], nb.f4[:])], '(n), (n, n) -> (n)', target='parallel', cache=True, fastmath=True)
@nb.guvectorize([(nb.f8[:], nb.f8[:, :], nb.f8[:]), (nb.f4[:], nb.f4[:, :], nb.f4[:])], '(n), (n, n) -> (n)', cache=True, fastmath=True)
def pmatmul(vec, mat, out):
    """
    mat: 3x3
    vec: 3, out: 3
    perform out = vec @ mat
    """
    for i in range(3):
        out[i] = 0
        for j in range(3):
            out[i] += vec[j] * mat[j, i]


class _Slice:
    """
    Proxy class for slicing ndarray
    """

    def convert_slice(self, s):
        """
        Convert a slice object to on that keep dims.
        """
        if np.isscalar(s):
            return slice(s, s+1)
        else:
            return s

    def __getitem__(self, args):
        """
        Return a `slice` object, which can be used to slice a np.ndarray.
        The difference is, the returned object will always keep dimension
        of the original ndarray.
        """
        return tuple(self.convert_slice(s) for s in args)

    def get_shape(self, path):
        """
        Get shape of saved array from given path.
        """
        conf_fn = os.path.join(path, 'fefi.input')
        names = f90nml.read(conf_fn)
        return (
                names['changeparameters']['nxd'],
                names['changeparameters']['nyd'],
                names['changeparameters']['nzd']
            )

    def get_lim(self, path):
        """
        Get coord limits from given path.
        """
        conf_fn = os.path.join(path, 'fefi.input')
        names = f90nml.read(conf_fn)
        return np.array(names['changeparameters']['ddomain']).reshape(3, 2)

Slice = _Slice()


class Config:

    def __init__(self, path, idx=None, dont_save=False, use_saved=True, version=2):
        """
        Generate grid info from configuration file.

        Parameters
        ----------
        path : str
            Folder contains the configuration file, or the file's path.

        idx : int, optional
            Which output is to read, will used as suffix of configuration
            key items if greater than 0.

        dont_save : bool, default False
            If True, will not attempt to save grid data. Otherwise
            `grid.npz` might be generated for latter use.

        use_saved : bool, default True
            If True, will try to load grid data from `grid.npz` if exists.

        version : int, optional
            For compatibility with older version. For older version,
            set to 1. Default is 2.
        """
        self.use_saved = use_saved
        if os.path.isfile(path):
            conf_fn = path
            self.path = os.path.dirname(path)
        else:
            conf_fn = os.path.join(path, 'fefi.input')
            self.path = path

        if idx is None or idx <= 0:
            self.idx = 0
        else:
            self.idx = int(idx)

        names = f90nml.read(conf_fn)

        coord_id = names['input_parameters']['coordinate']
        grid_type = names['changeparameters'][self._name('idiagbox')] % 10

        if version <= 1 or grid_type != 0:
            self.grid_size = (
                names['changeparameters'][self._name('nxd')],
                names['changeparameters'][self._name('nyd')],
                names['changeparameters'][self._name('nzd')],
            )
            self.limits = np.array(names['changeparameters'][self._name('ddomain')]).reshape(3, 2)
        else:
            self.grid_size = (
                names['input_parameters']['nx'] + 1,
                names['input_parameters']['ny'] + 1,
                names['input_parameters']['nz'] + 1,
            )
            self.limits = np.array(names['input_parameters']['domain']).reshape(3, 2)

        self.names = names

        self.coord_args = names.get('pyfefiparams', {}).get('coord_args', None)

        if grid_type == 1:
            coord_id = None

        self.coord = get_coord_from_id(coord_id, self)
        if self.coord is None:
            raise ValueError('Unsupported coordinate type: %d' % coord_id)

        self.fn_prefix = self.__get_file_name()
        all_field_data_fns = [fn for fn in os.listdir(self.path) if fn.startswith(self.fn_prefix) and fn.endswith('.nc')]
        self.nframes = len(all_field_data_fns)
        self.dont_save = dont_save

    def _name(self, base):
        if self.idx > 0:
            return base + self.idx
        return base

    def __get_file_name(self):
        prefixes = ['fieldds', 'fieldns', 'fieldmp', 'fieldeq']
        prefix = prefixes[self.idx]

        # out first one, newer version is used.
        if self.idx > 0:
            return prefix

        # pass one: test first output
        if os.path.exists(os.path.join(self.path, prefix + '%05d.nc' % 1)):
            return prefix
        if os.path.exists(os.path.join(self.path, 'field' + '%05d.nc' % 1)):
            return 'field'

        # pass two: list all files
        fns = os.listdir(self.path)
        for fn in fns:
            if fn.startswith(prefix):
                return prefix
        return 'field'

    def pqw(self, slices=None, data_type='float32'):
        """
        Return the coordinates in curvilinear coordinate system.

        Parameters
        ----------
        slices : tuple of slice objects, optional
            The slices to apply on the coordinates. If not given, will
            return all coordinates.

        data_type : str, optional
            The data type of the returned coordinates. Default is 'float32'.

        Returns
        -------
        p, q, w : tuple of 1D np.ndarray
            The coordinates in curvilinear coordinate system.
        """
        if slices is None:
            slices = Slice[:, :, :]
        lims = self.coord.convert_limits(self.limits)
        coords = (
                np.linspace(*lims[i], self.grid_size[i], dtype=np.dtype(data_type))[slices[i]]
                for i in range(3) )
        return coords

    def calc_xyz(self, slices, data_type='float32'):
        """
        Calculate the coordinates in Cartesian coordinate system of the
        grid points. For most cases, you should use `to_xyz` instead.

        Parameters
        ----------
        slices : tuple of slice objects
            The slices to apply on the coordinates.

        data_type : str, optional
            The data type of the returned coordinates. Default is 'float32'.

        Returns
        -------
        X, Y, Z : tuple of 3D np.ndarray
            The coordinates in Cartesian coordinate system.
        """
        lims = self.coord.convert_limits(self.limits)
        plim, qlim, wlim = lims
        ps = np.linspace(*plim, self.grid_size[0], dtype=np.dtype(data_type))
        qs = np.linspace(*qlim, self.grid_size[1], dtype=np.dtype(data_type))
        ws = np.linspace(*wlim, self.grid_size[2], dtype=np.dtype(data_type))
        if self.coord.coords_independent:
            ps = ps[slices[0]]
            qs = qs[slices[1]]
            ws = ws[slices[2]]
        P, Q, W = np.meshgrid(ps, qs, ws, indexing='ij')
        X, Y, Z = self.coord.to_cartesian(P, Q, W)
        if not self.coord.coords_independent:
            X = X[slices]
            Y = Y[slices]
            Z = Z[slices]
        return X, Y, Z

    def _try_load_xyz(self, slices):
        if os.path.exists(os.path.join(self.path, 'grid.npz')):
            data = np.load(os.path.join(self.path, 'grid.npz'))
            x = data['x'][slices]
            y = data['y'][slices]
            z = data['z'][slices]
            return x, y, z
        return None

    def to_xyz(self, slices, data_type='float32', try_save=True, use_saved=None):
        """
        Return the coordinates in Cartesian coordinate system. If the
        file 'grid.npz' exists, the method might load the coordinates
        from it instead of calculating them.

        Parameters
        ----------
        slices : tuple of slice objects
            The slices to apply on the coordinates.

        data_type : str, optional
            The data type of the returned coordinates. Default is 'float32'.

        try_save : bool, optional
            Whether to try to save the coordinates to a file for future use.
            Default is True.

        use_saved : bool, optional
            Whether to try to load the coordinates from a file. Default is True.
        """
        if use_saved is None:
            use_saved = self.use_saved
        if use_saved:
            xyz = self._try_load_xyz(slices)
            if xyz is not None:
                return xyz

        x, y, z = self.calc_xyz(slices, data_type)

        if self.coord.should_save and try_save and not self.dont_save:
            if slices == (slice(None), slice(None), slice(None)):
                self._save_xyz(data_type, (x, y, z))
            else:
                self._save_xyz(data_type, None)

        return x, y, z

    def _save_xyz(self, data_type, xyz=None):
        if not os.path.exists(os.path.join(self.path, 'grid.npz')):
            if xyz is None:
                x, y, z = self.calc_xyz((slice(None), slice(None), slice(None)), data_type)
            else:
                x, y, z = xyz
            np.savez_compressed(os.path.join(self.path, 'grid.npz'), x=x, y=y, z=z)

class Mesh:
    """
    A class for loading and storing a pyvista mesh from a netCDF file.
    """

    def __init__(
            self, conf,
            dont_save=None,
            ref_fn=None, recalc_grid=False,
            max_arrays=None, max_memory=4e9, data_type='float32',
            auto_load=False, slices=None):
        """
        Parameters
        ----------
        conf : str or Config
            The path to the config file folder or a Config object.

        dont_save : bool, optional
            True if not to save grid arrays.

        ref_fn : str, optional
            The path to a netCDF file containing the reference frame
            transformation matrix. If None, no transformation is applied.

        recalc_grid : bool, optional
            If True, recalculate grid even if it can be load from cache.

        max_arrays : int, optional
            The maximum number of arrays to store in the mesh. If None, the
            maximum number of arrays is calculated from the available memory.

        max_memory : number, optional
            The maximum amount of memory to use for storing arrays in bytes.

        data_type : str, optional
            Data type of netCDF dataset. Used to estimate memory requirement.
            Default to 'float32'.

        auto_load : bool, optional
            Whether to automatically load data from the netCDF file when
            accessing non-existent arrays. Only works if `lims' is a folder.

        slices : tuple of slice objects or ints, optional
            The slices to apply to the data when loading. If None, no slices
            are applied.
        """
        self.auto_load = False
        if slices is None:
            self.slices = tuple([slice(None)] * 3)
        else:
            self.slices = tuple(slices)
        self.slice_for_nc = tuple(self.slices[::-1])
        #print(self.slice_for_nc)

        if isinstance(conf, str):
            self.config = Config(conf, dont_save)
        elif isinstance(conf, Config):
            self.config = conf
        else:
            raise TypeError('conf must be a str or Config')

        self.auto_load = auto_load
        self.path = self.config.path
        self.nframes = self.config.nframes

        use_saved = None
        if recalc_grid:
            use_saved = False
        self.x, self.y, self.z = self.config.to_xyz(self.slices, data_type=data_type, use_saved=recalc_grid)

        self.xyz_for_mesh = [
                self.x.transpose(2, 1, 0),
                self.y.transpose(2, 1, 0),
                self.z.transpose(2, 1, 0)]

        self.nx, self.ny, self.nz = self.x.shape

        try_ref_fn = os.path.join(self.path, 'ref.nc')
        if ref_fn is None and os.path.exists(try_ref_fn):
            ref_fn = try_ref_fn
        if ref_fn is not None:
            ds = netCDF4.Dataset(ref_fn)
            concarEc = []
            for i in range(9):
                concarEc.append(ds.variables['onarE' + str(i+1)][self.slice_for_nc].transpose(2, 1, 0))
            concarEc = np.array(concarEc).reshape((3, 3) + concarEc[0].shape).transpose(2, 3, 4, 1, 0)
            self.concarEc = concarEc
        else:
            self.concarEc = None

        if max_arrays is None:
            npoints = self.nx * self.ny * self.nz
            match data_type:
                case 'float32' | 'float' | 'f' | 'single' | 'f4':
                    max_arrays = int(max_memory / 4 / npoints) - 3
                case 'float64' | 'double' | 'd' | 'f8':
                    max_arrays = int(max_memory / 8 / npoints) - 3
        if max_arrays < 1:
            raise ValueError('max_arrays must be greater than or equal to 1. Try increase `max_memory`.')

        self.max_arrays = max_arrays
        self.array_names = []
        self.protected_array_names = []
        self.arrays = {}

    def _convert_vec(self, vec):
        if self.concarEc is None:
            return vec
        #vec = pmatmul(vec, self.concarEc)
        vec = vec[..., np.newaxis, :] @ self.concarEc
        vec = vec[..., 0, :]
        return np.require(vec, requirements='F')

    def _calc_size(self):
        size = 0
        for k, v in self.arrays.items():
            size += 1 if v.ndim == 3 else 3
        return size

    def _free_space(self, num_required):
        """
        Delete old arrays to make room for new ones.
        """
        num_now = self._calc_size()
        num_free = self.max_arrays - num_now
        while num_free < num_required:
            # Delete the oldest array
            name = self.array_names.pop(0)
            free_num = 1 if self.arrays[name].ndim == 3 else 3
            self.__delkey(name)
            num_free += free_num

    def load(self, fn, var_names, suffix='', protected=False):
        """
        Load a variable from a netCDF file and store it in the mesh.
        If the variable is a vector, it is stored as a 3D array with
        the last dimension corresponding to the vector components.

        Parameters
        ----------
        fn : str or int
            The path to the netCDF file or the index of the file in the
            directory specified by the path.
        var_names : str or list of str
            The name of the variable(s) to load.
        suffix : str, optional
            A suffix to append to the variable name.
        """
        # Determine the file name
        if isinstance(fn, int):
            if self.path is None:
                raise ValueError('The Mesh instance is not initialized from a path, so full path of netCDF file must be given.')
            suffix = '_' + str(fn)
            fn = os.path.join(self.path, self.config.fn_prefix + '%05d.nc' % fn)
        if not os.path.exists(fn):
            if self.path is None:
                raise ValueError('{} does not exist.'.format(fn))
            fn_new = os.path.join(self.path, fn)
            if not os.path.exists(fn_new):
                raise ValueError('File {} or {} does not exist.'.format(fn, fn_new))
            fn = fn_new

        # Check the number of arrays, and delete the oldest arrays if necessary
        if isinstance(var_names, str):
            var_names = [var_names]

        store_names = []

        # Load the data
        ds = netCDF4.Dataset(fn)
        for var_name in var_names:
            store_name = var_name + str(suffix)
            if store_name in self.array_names or store_name in self.protected_array_names:
                continue
            if var_name in ds.variables:
                self._free_space(1)
                self.arrays[store_name] = ds.variables[var_name][self.slice_for_nc].transpose(2, 1, 0)
            elif var_name + 'x' in ds.variables:
                self._free_space(3)
                data = []
                axis_names = ['x', 'y', 'z']
                for i in range(3):
                    #data.append(ds.variables[var_name + axis_names[i]][self.slice_for_nc].transpose(2, 1, 0))
                    data.append(ds.variables[var_name + axis_names[i]][self.slice_for_nc])
                #data = np.stack(data).transpose(1, 2, 3, 0)
                data = np.stack(data).transpose(3, 2, 1, 0)
                self.arrays[store_name] = self._convert_vec(data)
            else:
                raise ValueError('Variable ' + var_name + ' not found in ' + fn)
            self.array_names.append(store_name)
            store_names.append(store_name)

        if not store_names[0] in self.array_names:
            raise ValueError('Memory limit exceeded. Try increasing max_memory.')

        if protected:
            self.protect(store_names)

    def protect(self, var_names):
        """
        Protect a variable from being deleted when loading new variables.
        """
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name in self.array_names:
                self.protected_array_names.append(var_name)
                self.array_names.remove(var_name)

    def __setitem__(self, name, val):
        if not isinstance(name, str):
            frame, name = name
            name = name + '_' + str(frame)
        if val.ndim == 3:
            self._free_space(1)
        else:
            assert(val.ndim == 4 and val.shape[3] == 3)
            self._free_space(3)
        self.arrays[name] = val
        self.array_names.append(name)

    def __getitem__(self, name) -> np.ndarray:
        if not isinstance(name, str):
            frame, name = name
            name = name + '_' + str(frame)
        if not name in self.arrays:
            if self.auto_load:
                name_sec = name.split('_')
                frame = int(name_sec[-1])
                _name = '_'.join(name_sec[:-1])
                self.load(frame, _name)
            else:
                raise KeyError('Variable ' + name + ' not found.')
        return self.arrays[name]

    def __delkey(self, key):
        """
        Delete a variable from the mesh.
        """
        self.arrays.pop(key)

        if key in self.array_names:
            self.array_names.remove(key)
        elif key in self.protected_array_names:
            self.protected_array_names.remove(key)
    
    def __delitem__(self, name):
        all_names = self.array_names + self.protected_array_names
        if isinstance(name, str):
            if name in all_names:
                self.__delkey(name)
            else:
                for key in all_names:
                    if key.startswith(name + '_'):
                        self.__delkey(key)
        elif isinstance(name, tuple):
            frame, name = name
            self.__delkey(name + '_' + str(frame))
        else:
            for key in all_names:
                if key.endswith('_' + str(name)):
                    self.__delkey(key)

    def __call__(self, frame, name) -> str:
        """
        Return the name of a variable with the given frame number.
        """
        return name + '_' + str(frame)
    
    def _add_to_mesh(self, mesh, name):
        array = self[name]
        if array.ndim == 3:
            mesh[name] = array.ravel()
        else:
            mesh[name] = array.reshape(-1, 3)

    def mesh(self, *names):
        """
        Return a PyVista mesh with the given variables.
        The variable names can be given as strings, or as tuples of
        (frame, name).
        """
        if not HAS_PYVISTA:
            raise Exception("PyVista is not available.")
        m = pv.StructuredGrid(*self.xyz_for_mesh)
        for name in names:
            if isinstance(name, str):
                self._add_to_mesh(m, name)
            else:
                self._add_to_mesh(m, self(*name))
        return m

    def remove(self, *args):
        """
        Remove a variable from the mesh.

        Usage
        -----
        mesh.remove(name) :
            Remove the variable with the given name, or all variables
            with the given prefix.
        mesh.remove(frame) :
            Remove all variables with the given frame number.
        mesh.remove(frame, name) :
            Remove the variable with the given name and frame number.
        """
        if len(args) == 1:
            name = args[0]
            del self[name]
        elif len(args) == 2:
            frame, name = args
            del self[frame, name]
        else:
            raise ValueError('Invalid number of arguments.')
