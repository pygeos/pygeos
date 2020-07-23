Changelog
=========

Version 0.8 (unreleased)
------------------------

**Highlights of this release**

* Release the GIL to allow for multithreading in most functions (#113, #156)
* Renamed lib.haussdorf_distance_densify to lib.hausdorff_distance_densify (#151)
* Addition of a ``frechet_distance()`` function for GEOS >= 3.7 (#144)
* Fixed segfaults when adding empty geometries to the STRtree (#147)
* Addition of a ``build_area()`` function for GEOS >= 3.8 (#141)
* Addition of a ``normalize()`` function (#136)
* Addition of a ``make_valid()`` function for GEOS >= 3.8 (#107)

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward
* Casper van der Wel
* Joris Van den Bossche
* Krishna Chaitanya +
* Martin Fleischmann +
* Tom Clancy +


Version 0.7 (2020-03-18)
------------------------

**Highlights of this release**

* STRtree improvements for spatial indexing:
  * Directly include predicate evaluation in ``STRtree.query()`` (#87)
  * Query multiple input geometries (spatial join style) with ``STRtree.query_bulk`` (#108)
* Addition of a ``total_bounds()`` function (#107)
* Geometries are now hashable, and can be compared with ``==`` or ``!=`` (#102)
* Fixed bug in ``create_collections()`` with wrong types (#86) 
* Fixed a reference counting bug in STRtree (#97, #100) 
* Start of a benchmarking suite using ASV (#96)
* This is the first release that will provide wheels!

**Acknowledgments**

Thanks to everyone who contributed to this release!
People with a "+" by their names contributed a patch for the first time.

* Brendan Ward +
* Casper van der Wel
* Joris Van den Bossche
* Mike Taves +


Version 0.6 (2020-01-31)
------------------------

Highlights of this release:

* Addition of the STRtree class for spatial indexing (#58)
* Addition of a ``bounds`` function (#69)
* A new ``from_shapely`` function to convert Shapely geometries to pygeos.Geometry (#61) 
* Reintroduction of the ``shared_paths`` function (#77) 

Contributors:

* Casper van der Wel
* Joris Van den Bossche
* mattijn +


Version 0.5 (2019-10-25)
------------------------

Highlights of this release:

* Moved to the pygeos GitHub organization.
* Addition of functionality to get and transform all coordinates (eg for reprojections or affine transformations) [#44]
* Ufuncs for converting to and from the WKT and WKB formats [#45]
* ``equals_exact`` has been added [PR #57]


Version 0.4 (2019-09-16)
------------------------

This is a major release of PyGEOS and the first one with actual release notes. Most important features of this release are:

* ``buffer`` and ``haussdorff_distance`` were completed  [#15] 
* ``voronoi_polygons`` and ``delaunay_triangles`` have been added [#17]
* The PyGEOS documentation is now mostly complete and available on http://pygeos.readthedocs.io .
* The concepts of "empty" and "missing" geometries have been separated. The ``pygeos.Empty`` and ``pygeos.NaG`` objects has been removed. Empty geometries are handled the same as normal geometries. Missing geometries are denoted by ``None`` and are handled by every pygeos function. ``NaN`` values cannot be used anymore to denote missing geometries. [PR #36]
* Added ``pygeos.__version__`` and ``pygeos.geos_version``. [PR #43]

