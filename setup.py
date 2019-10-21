import os
import subprocess
import sys
from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import logging
import versioneer

log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.addHandler(ch)

MIN_GEOS_VERSION = "3.5"

if "all" in sys.warnoptions:
    # show GEOS messages in console with: python -W all
    log.setLevel(logging.DEBUG)


def get_geos_config(option):
    """Get configuration option from the `geos-config` development utility

    The PATH environment variable should include the path where geos-config is located.
    """
    try:
        stdout, stderr = subprocess.Popen(
            ["geos-config", option], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
    except OSError:
        return
    if stderr and not stdout:
        log.warning("geos-config %s returned '%s'", option, stderr.decode().strip())
        return
    result = stdout.decode().strip()
    log.debug("geos-config %s returned '%s'", option, result)
    return result


def get_geos_paths():
    """Obtain the paths for compiling and linking with the GEOS C-API

    First the presence of the GEOS_INCLUDE_PATH and GEOS_INCLUDE_PATH environment
    variables is checked. If they are both present, these are taken.

    If one of the two paths was not present, geos-config is called (it should be on the
    PATH variable). geos-config provides all the paths.

    If geos-config was not found, no additional paths are provided to the extension. It is
    still possible to compile in this case using custom arguments to setup.py.
    """
    include_dir = os.environ.get("GEOS_INCLUDE_PATH")
    library_dir = os.environ.get("GEOS_LIBRARY_PATH")
    if include_dir and library_dir:
        return {
            "include_dirs": [include_dir],
            "library_dirs": [library_dir],
            "libraries": ["geos_c"],
        }
    geos_version = get_geos_config("--version")
    if not geos_version:
        log.warning(
            "Could not find geos-config executable. Either append the path to geos-config"
            " to PATH or manually provide the include_dirs, library_dirs, libraries and "
            "other link args for compiling against a GEOS version >=%s.",
            MIN_GEOS_VERSION,
        )
        return {}
    if LooseVersion(geos_version) < LooseVersion(MIN_GEOS_VERSION):
        raise ImportError(
            "GEOS version should be >={}, found {}".format(
                MIN_GEOS_VERSION, geos_version
            )
        )
    libraries = []
    library_dirs = []
    include_dirs = []
    extra_link_args = []
    for item in get_geos_config("--cflags").split():
        if item.startswith("-I"):
            include_dirs.extend(item[2:].split(":"))
    for item in get_geos_config("--clibs").split():
        if item.startswith("-L"):
            library_dirs.extend(item[2:].split(":"))
        elif item.startswith("-l"):
            libraries.append(item[2:])
        else:
            extra_link_args.append(item)
    return {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "extra_link_args": extra_link_args,
    }


# Add numpy include dirs without importing numpy on module level.
# See https://stackoverflow.com/questions/19919905/
# how-to-bootstrap-numpy-installation-in-setup-py/21621689#21621689
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


module_lib = Extension(
    "pygeos.lib",
    sources=["src/lib.c", "src/geos.c", "src/pygeom.c", "src/ufuncs.c", "src/coords.c"],
    **get_geos_paths()
)


try:
    descr = open(os.path.join(os.path.dirname(__file__), "README.rst")).read()
except IOError:
    descr = ""


version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

setup(
    name="pygeos",
    version=version,
    description="GEOS wrapped in numpy ufuncs",
    long_description=descr,
    url="https://github.com/pygeos/pygeos",
    author="Casper van der Wel",
    license="BSD 3-Clause",
    packages=["pygeos"],
    setup_requires=["numpy"],
    install_requires=["numpy>=1.10"],
    extras_require={
        "test": ["pytest"],
        "docs": ["sphinx", "numpydoc"],
    },
    python_requires=">=3",
    include_package_data=True,
    ext_modules=[module_lib],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: GIS",
        "Operating System :: Unix",
    ],
    cmdclass=cmdclass,
)
