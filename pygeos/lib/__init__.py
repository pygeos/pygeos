"""This module provides the top-level internal API exposed from
C and Cython extensions.  The top-level functions and attributes
are promoted to the pygeos.lib namespace for ease of use within
the Python API.

These are intended to be wrapped within the PyGEOS C API and not
used directly.
"""

from pygeos.lib.core import *

from pygeos.lib.geom import get_parts
