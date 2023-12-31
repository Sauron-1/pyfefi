from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
import platform
import sys
import os

__version__ = "0.0.1"

headers = glob("include/*")

cxx_config = dict(
    include_dirs = ["include"],
    define_macros = [("VERSION_INFO", __version__)],
    extra_link_args = ["-fopenmp"],
)

# set up compiler flags based on platform
if platform.system() == "Windows":
    cxx_config['extra_compile_args'] = ["/std:c++20", "/O2", "/openmp", "/DNDEBUG", "/arch:AVX2"]
else:
    cxx_config['extra_compile_args'] = ["-std=c++20", "-O3", "-march=core-avx2", "-fopenmp", "-Wno-sign-compare", "-DNDEBUG", "-Wextra"]

#cxx_config['extra_compile_args'].append('-DBOUNDSCHECK')

ext_modules = [
    Pybind11Extension(
        "pyfefi.coords.lib.cartesian_mod",
        ["src/coords/cartesian_mod_core.cpp"],
        **cxx_config
    ),
    Pybind11Extension(
        "pyfefi.coords.lib.sphere_mod",
        ["src/coords/sphere_mod_core.cpp"],
        **cxx_config
    ),
    Pybind11Extension(
        "pyfefi.interp",
        ["src/interp/interp.cpp"],
        **cxx_config
    ),
    Pybind11Extension(
        "pyfefi.lic",
        ["src/lic.cpp"],
        **cxx_config
    ),
]

packages = [os.path.join('pyfefi', p) for p in find_packages('pyfefi')]
packages.append('pyfefi')

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

setup(
    name="pyfefi",
    version=__version__,
    author="JyRen",
    author_email="j.y.ren9871@gmail.com",
    url="",
    description="Python post processer for fefi code",
    long_description="",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0"],
    cmdclass={"build_ext": build_ext},
    packages=packages,
    zip_safe=False,
)
