import os
from setuptools import setup, Extension
import geosconfig
import numpy

module_ufuncs = Extension(
    "pygeos.ufuncs",
    sources=["src/ufuncs.c"],
    include_dirs=geosconfig.include_dirs + [numpy.get_include()],
    library_dirs=geosconfig.library_dirs,
    libraries=geosconfig.libraries,
    extra_link_args=geosconfig.extra_link_args,
)

try:
    descr = open(os.path.join(os.path.dirname(__file__), "README.rst")).read()
except IOError:
    descr = ""


setup(
    name="pygeos",
    version="0.3.dev0",
    description="GEOS wrapped in numpy ufuncs",
    long_description=descr,
    url="https://github.com/caspervdw/pygeos",
    author="Casper van der Wel",
    license="BSD 3-Clause",
    packages=["pygeos"],
    install_requires=["numpy"],
    extras_require={"test": ["pytest"]},
    python_requires=">=3",
    include_package_data=True,
    ext_modules=[module_ufuncs],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: GIS",
        "Operating System :: Unix",
    ],
)
