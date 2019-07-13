"""Code copied from shapely"""

from ctypes import CDLL, cdll, c_void_p, c_char_p
from ctypes.util import find_library
import os
import logging
import re
import subprocess
import sys


"""
Minimal proxy to a GEOS C dynamic library, which is system dependant

Two environment variables influence this module: GEOS_LIBRARY_PATH and/or
GEOS_CONFIG.

If GEOS_LIBRARY_PATH is set to a path to a GEOS C shared library, this is
used. Otherwise GEOS_CONFIG can be set to a path to `geos-config`. If
`geos-config` is already on the PATH environment variable, then it will
be used to help better guess the name for the GEOS C dynamic library.
"""


# Add message handler to this module's logger
log = logging.getLogger(__name__)
ch = logging.StreamHandler()
log.addHandler(ch)

if "all" in sys.warnoptions:
    # show GEOS messages in console with: python -W all
    log.setLevel(logging.DEBUG)


# The main point of this module is to load a dynamic library to this variable
lgeos = None

# First try: use GEOS_LIBRARY_PATH environment variable
if "GEOS_LIBRARY_PATH" in os.environ:
    geos_library_path = os.environ["GEOS_LIBRARY_PATH"]
    try:
        lgeos = CDLL(geos_library_path)
    except:
        log.warn(
            "cannot open shared object from GEOS_LIBRARY_PATH: %s", geos_library_path
        )
    if lgeos:
        if hasattr(lgeos, "GEOSversion"):
            log.debug("found GEOS C library using GEOS_LIBRARY_PATH")
        else:
            raise OSError(
                "shared object GEOS_LIBRARY_PATH is not a GEOS C library: "
                + str(geos_library_path)
            )

# Second try: use GEOS_CONFIG environment variable
if "GEOS_CONFIG" in os.environ:
    geos_config = os.environ["GEOS_CONFIG"]
    log.debug("geos_config: %s", geos_config)
else:
    geos_config = "geos-config"


def get_geos_config(option):
    """Get configuration option from the `geos-config` development utility

    Path to utility is set with a module-level `geos_config` variable, which
    can be changed or unset.
    """
    geos_config = globals().get("geos_config")
    if not geos_config or not isinstance(geos_config, str):
        raise OSError("Path to geos-config is not set")
    try:
        stdout, stderr = subprocess.Popen(
            [geos_config, option], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
    except OSError as ex:
        # e.g., [Errno 2] No such file or directory
        raise OSError("Could not find geos-config %r: %s" % (geos_config, ex))
    if stderr and not stdout:
        raise ValueError(stderr.strip())
    if sys.version_info[0] >= 3:
        result = stdout.decode("ascii").strip()
    else:
        result = stdout.strip()
    log.debug("%s %s: %r", geos_config, option, result)
    return result


# Now try and use the utility to load from `geos-config --clibs` with
# some magic smoke to guess the other parts of the library name
try:
    clibs = get_geos_config("--clibs")
except OSError:
    geos_config = None
if not lgeos and geos_config:
    base = ""
    name = "geos_c"
    for item in clibs.split():
        if item.startswith("-L"):
            base = item[2:]
        elif item.startswith("-l"):
            name = item[2:]
    # Now guess the actual library name using a list of possible formats
    if sys.platform == "win32":
        # Unlikely, since geos-config is a shell script, but you never know...
        fmts = ["{name}.dll"]
    elif sys.platform == "darwin":
        fmts = ["lib{name}.dylib", "{name}.dylib", "{name}.framework/{name}"]
    elif os.name == "posix":
        fmts = ["lib{name}.so", "lib{name}.so.1"]
    guesses = []
    for fmt in fmts:
        lib_name = fmt.format(name=name)
        geos_library_path = os.path.join(base, lib_name)
        try:
            lgeos = CDLL(geos_library_path)
            break
        except:
            guesses.append(geos_library_path)
    if lgeos:
        if hasattr(lgeos, "GEOSversion"):
            log.debug("found GEOS C library using geos-config")
        else:
            raise OSError(
                "shared object found by geos-config is not a GEOS C library: "
                + str(geos_library_path)
            )
    else:
        log.warn("cannot open shared object from '%s --clibs': %r", geos_config, clibs)
        log.warn(
            "there were %d guess(es) for this path:\n\t%s",
            len(guesses),
            "\n\t".join(guesses),
        )


# Platform-specific attempts, and build `free` object


def load_dll(libname, fallbacks=None):
    lib = find_library(libname)
    dll = None
    if lib is not None:
        try:
            log.debug("Trying `CDLL(%s)`", lib)
            dll = CDLL(lib)
        except OSError:
            log.warn("Failed `CDLL(%s)`", lib)
            pass

    if not dll and fallbacks is not None:
        for name in fallbacks:
            try:
                log.debug("Trying `CDLL(%s)`", name)
                dll = CDLL(name)
            except OSError:
                # move on to the next fallback
                log.warn("Failed `CDLL(%s)`", name)
                pass

    if dll:
        log.debug("Library path: %r", lib or name)
        log.debug("DLL: %r", dll)
        return dll
    else:
        # No shared library was loaded. Raise OSError.
        raise OSError(
            "Could not find library {} or load any of its variants {}".format(
                libname, fallbacks or []
            )
        )


if sys.platform.startswith("linux"):
    if not lgeos:
        lgeos = load_dll("geos_c", fallbacks=["libgeos_c.so.1", "libgeos_c.so"])
    free = load_dll("c").free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == "darwin":
    if not lgeos:
        if hasattr(sys, "frozen"):
            # .app file from py2app
            alt_paths = [
                os.path.join(
                    os.environ["RESOURCEPATH"], "..", "Frameworks", "libgeos_c.dylib"
                )
            ]
        else:
            alt_paths = [
                # The Framework build from Kyng Chaos
                "/Library/Frameworks/GEOS.framework/Versions/Current/GEOS",
                # macports
                "/opt/local/lib/libgeos_c.dylib",
            ]
        lgeos = load_dll("geos_c", fallbacks=alt_paths)

    free = load_dll("c", fallbacks=["/usr/lib/libc.dylib"]).free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == "win32":
    if not lgeos:
        try:
            egg_dlls = os.path.abspath(os.path.join(os.path.dirname(__file__), "DLLs"))
            wininst_dlls = os.path.abspath(os.__file__ + "../../../DLLs")
            original_path = os.environ["PATH"]
            os.environ["PATH"] = "%s;%s;%s" % (egg_dlls, wininst_dlls, original_path)
            lgeos = CDLL("geos_c.dll")
        except (ImportError, WindowsError, OSError):
            raise

    def free(m):
        try:
            cdll.msvcrt.free(m)
        except WindowsError:
            # TODO: http://web.archive.org/web/20070810024932/
            #     + http://trac.gispython.org/projects/PCL/ticket/149
            pass


elif sys.platform == "sunos5":
    if not lgeos:
        lgeos = load_dll("geos_c", fallbacks=["libgeos_c.so.1", "libgeos_c.so"])
    free = CDLL("libc.so.1").free
    free.argtypes = [c_void_p]
    free.restype = None

else:  # other *nix systems
    if not lgeos:
        lgeos = load_dll("geos_c", fallbacks=["libgeos_c.so.1", "libgeos_c.so"])
    free = load_dll("c", fallbacks=["libc.so.6"]).free
    free.argtypes = [c_void_p]
    free.restype = None


def _geos_version():
    # extern const char GEOS_DLL *GEOSversion();
    GEOSversion = lgeos.GEOSversion
    GEOSversion.restype = c_char_p
    GEOSversion.argtypes = []
    # #define GEOS_CAPI_VERSION "@VERSION@-CAPI-@CAPI_VERSION@"
    geos_version_string = GEOSversion()
    if sys.version_info[0] >= 3:
        geos_version_string = geos_version_string.decode("ascii")

    res = re.findall(r"(\d+)\.(\d+)\.(\d+)", geos_version_string)
    assert len(res) == 2, res
    geos_version = tuple(int(x) for x in res[0])
    capi_version = tuple(int(x) for x in res[1])

    if geos_version[0] < 3 or geos_version[1] < 5:
        raise ValueError(
            "PyGEOS requires GEOS version 3.5 or later, {} was found".format(
                geos_version_string
            )
        )

    return geos_version_string, geos_version, capi_version


geos_version_string, geos_version, geos_capi_version = _geos_version()


class GEOSConfig(object):
    """Interface to config options from the `geos-config` utility
    """

    def __init__(self, cmd):
        self.cmd = cmd

    def get(self, option):
        try:
            stdout, stderr = subprocess.Popen(
                [self.cmd, option], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ).communicate()
        except OSError as ex:
            # e.g., [Errno 2] No such file or directory
            raise OSError("Could not find geos-config script")
        if stderr and not stdout:
            raise ValueError(stderr.strip())
        if sys.version_info[0] >= 3:
            result = stdout.decode("ascii").strip()
        else:
            result = stdout.strip()
        log.debug("%s %s: %r", self.cmd, option, result)
        return result

    def version(self):
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", self.get("--version").strip())
        return tuple(map(int, match.groups()))


# Prepare build opts and args for the speedups extension module.
include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []

geos_config = GEOSConfig(os.environ.get("GEOS_CONFIG", "geos-config"))
if geos_version and geos_config:
    # Collect other options from GEOS configuration.
    for item in geos_config.get("--cflags").split():
        if item.startswith("-I"):
            include_dirs.extend(item[2:].split(":"))
    for item in geos_config.get("--clibs").split():
        if item.startswith("-L"):
            library_dirs.extend(item[2:].split(":"))
        elif item.startswith("-l"):
            libraries.append(item[2:])
        else:
            # e.g. -framework GEOS
            extra_link_args.append(item)
