Installation
============

Installation from PyPI
----------------------

PyGEOS is available as a binary distribution (wheel) for Linux, OSX and Windows platforms.
We strongly recommand installation as follows::

    $ pip install pygeos


Installation using conda
------------------------

PyGEOS is available on the conda-forage channel. Install as follows::

    $ conda install pygeos --channel conda-forge


Installation with custom GEOS libary
------------------------------------

On Linux::

    $ sudo apt install libgeos-dev
    $ pip install pygeos --no-binary

On OSX::

    $ brew install geos
    $ pip install pygeos --no-binary

We do not have a recipe for Windows platforms. The following steps should enable you
to build PyGEOS yourself:

- Get a C compiler applicable to your Python version (https://wiki.python.org/moin/WindowsCompilers)
- Download and install a GEOS binary (https://trac.osgeo.org/osgeo4w/)
- Set GEOS_INCLUDE_PATH and GEOS_LIBRARY_PATH environment variables (see below for notes on GEOS discovery)
- Run ``pip install pygeos --no-binary``
- Make sure the GEOS dlls are available on the PATH

Installation from source
------------------------

The same as above, but then instead of installing pygeos with pip, you clone the
package from Github::

    $ git clone git@github.com:pygeos/pygeos.git

Install it in development mode using `pip`::

    $ pip install -e .[test]

Run the unittests::

    $ pytest --pyargs pygeos.tests


Notes on GEOS discovery
-----------------------

If GEOS is installed on Linux or OSX, normally the ``geos-config`` command line utility
will be available and ``pip install`` will find GEOS automatically.
If the correct ``geos-config`` is not on the PATH, you can add it as follows:

    $ export PATH=/path/to/geos/bin:$PATH

Alternatively, you can specify where PyGEOS should look for the GEOS library and header
files before installation (Linux/OSX)::

    $ export GEOS_INCLUDE_PATH=$CONDA_PREFIX/Library/include
    $ export GEOS_LIBRARY_PATH=$CONDA_PREFIX/Library/lib

On Windows, there is no ``geos-config`` and the include and lib folders need to be
specified manually in any case::

    $ set GEOS_INCLUDE_PATH=%CONDA_PREFIX%\Library\include
    $ set GEOS_LIBRARY_PATH=%CONDA_PREFIX%\Library\lib

For all platforms, the GEOS library version that was used to compile PyGEOS with will
need to remain available on your system. Make sure your that the dynamic linker paths are
set such that the libraries can be found. If you are doing this from the default Conda paths
as in above example, this will already be the case and you don't need to do anything. For
custom usage on Linux::

    $ export LD_LIBRARY_PATH=/path/to/geos/lib:$LD_LIBRARY_PATH
    $ sudo ldconfig  # refresh dynamic linker cache

On OSX::

    $ export DYLD_LIBRARY_PATH=/path/to/geos/lib:$DYLD_LIBRARY_PATH

On Windows::

    $ export PATH=%PATH%;C:\path\to\geos\bin
