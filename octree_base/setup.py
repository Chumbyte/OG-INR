from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# from https://stackoverflow.com/a/9740721/19834294
# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


# setup(
#     ext_modules = cythonize(["octree.pyx","octree_defs.pyx"], language_level = "3", annotate=True)
# )

ext_modules = [
    Extension(
        "octree_defs",
        ["octree_defs.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "octree_structure",
        ["octree_structure.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "octree_algo",
        ["octree_algo.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "octree",
        ["octree.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name='octree',
    ext_modules=cythonize(ext_modules, annotate=True),
    install_requires=["numpy"]
)

# Use with `python setup.py build_ext --inplace`