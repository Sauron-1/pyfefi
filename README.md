# PyFefi
Post processor for Prof. Wang's fefi code, mainly used for global simulations. This lib is meant to work with curvilinear netCDF outputs, especially the pnetcdf version.

## Dependencies
A C++20 compatible compiler, and the python libraries listed in `requirements.txt`.

## Install
Run `python3 setup.py install` in the project root directory.

## Usage
The main interface provided is the `Mesh` object for loading the data. For detail usage please refer to the python code.

Here is an example code:

```python
import numpy as np
import matplotlib.pyplot as plt

from pyfefi import Mesh, Slice, Config
from pyfefi import interp

t = 150
# The path to the folder where netCDF outputs, along
# with the simulation input file "fefi.input" can be
# found.
path = 'path/to/folder/'

# Initialize the configuration
conf = Config(path)

# Build data reader.
data = Mesh(conf, auto_load=True)

# When `auto_load` is enabled, the realder will load
# variable 'Ni' at `t` automaticly
Ni = data[t, 'Ni']

# Next are example using `interp` to convert data to
# cartesian coordinates
x = np.linspace(0, 18, 1600)
z = np.linspace(-15, 15, 1600)
X, Z = np.meshgrid(x, z, indexing='ij')

Y = np.zeros_like(X)

p, q, w = conf.pqw()
coords = conf.coord.from_cartesian(X, Y, Z)

result = interp((p, q, w), coords, Ni)
result = np.linalg.norm(result, axis=-1)

pcm = plt.pcolormesh(X, Y, result, vmin=0, vmax=5)
plt.gca().set_aspect(1)
plt.colorbar(pcm)
plt.show()
```

## Tested platform
Ubuntu 22.04 with GCC 11.3.0, Python 3.10.6.
