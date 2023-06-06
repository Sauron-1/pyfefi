from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

__version__ = "0.0.1"

# submodules:
#     cartesian_mod -> src/coords/cartesian_mod_core.cpp
#         install to: pyfefi.coords.lib.cartesian_mod
#     interp -> src/interp/interp.cpp
#         install to: pyfefi.interp
# main module:
#     pyfefi -> pyfefi (pure python)
#         install to: pyfefi
# cxx includes:
#     include
# cxx compile flags:
#     -std=c++20 -O3 -march=native -fopenmp

cxx_config = dict(
    include_dirs = ["include"],
    define_macros = [("VERSION_INFO", __version__)],
    extra_compile_args = ["-std=c++20", "-O3", "-march=native", "-fopenmp", "-Wno-sign-compare", "-DNDEBUG"],
    extra_link_args = ["-fopenmp"],
)

ext_modules = [
    Pybind11Extension(
        "pyfefi.coords.lib.cartesian_mod",
        ["src/coords/cartesian_mod_core.cpp"],
        **cxx_config
    ),
    Pybind11Extension(
        "pyfefi.interp",
        ["src/interp/interp.cpp"],
        **cxx_config
    ),
]

packages = [os.path.join('pyfefi', p) for p in find_packages('pyfefi')]
packages.append('pyfefi')

setup(
    name="pyfefi",
    version=__version__,
    author="JyRen",
    author_email="j.y.ren9871@gmail.com",
    url="",
    description="Python wrapper for FEFF interface",
    long_description="",
    ext_modules=ext_modules,
    install_requires=[
        "numpy", "scipy", "pybind11", "pyvista", "numba", "matplotlib", "netCDF4", "f90nml", "pandas"
    ],
    setup_requires=["pybind11>=2.5.0"],
    cmdclass={"build_ext": build_ext},
    packages=packages,
    zip_safe=False,
)
