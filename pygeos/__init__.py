from ._version import get_versions  # NOQA
from .lib import geos_capi_version  # NOQA
from .lib import geos_capi_version_string  # NOQA
from .lib import geos_version  # NOQA
from .lib import geos_version_string  # NOQA


from .constructive import *  # NOQA
from .coordinates import *  # NOQA
from .creation import *  # NOQA
from .decorators import UnsupportedGEOSOperation  # NOQA
from .geometry import *  # NOQA
from .io import *  # NOQA
from .lib import Geometry  # NOQA
from .lib import GEOSException  # NOQA
from .linear import *  # NOQA
from .measurement import *  # NOQA
from .predicates import *  # NOQA
from .set_operations import *  # NOQA
from .strtree import *  # NOQA

__version__ = get_versions()["version"]
del get_versions
