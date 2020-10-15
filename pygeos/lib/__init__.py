"""This module provides the top-level internal API exposed from
C and Cython extensions.  The top-level functions and attributes
are promoted to the pygeos.lib namespace for ease of use within
the Python API.

These are intended to be wrapped within the PyGEOS C API and not
used directly.
"""

from pygeos.lib.core import (
    GEOSException,
    Geometry,
    geos_version,
    geos_version_string,
    geos_capi_version,
    geos_capi_version_string,
    registry,
    # STRtree:
    STRtree,
    # ufuncs:
    area,
    boundary,
    bounds,
    buffer,
    centroid,
    contains,
    convex_hull,
    count_coordinates,
    covered_by,
    covers,
    create_collection,
    crosses,
    delaunay_triangles,
    difference,
    disjoint,
    distance,
    envelope,
    equals,
    equals_exact,
    extract_unique_points,
    from_shapely,
    from_wkb,
    from_wkt,
    get_coordinates,
    get_coordinate_dimension,
    get_dimensions,
    get_exterior_ring,
    get_geometry,
    get_interior_ring,
    get_num_coordinates,
    get_num_geometries,
    get_num_interior_rings,
    get_num_points,
    get_point,
    get_srid,
    get_type_id,
    get_x,
    get_y,
    has_z,
    hausdorff_distance,
    hausdorff_distance_densify,
    intersection,
    intersects,
    is_closed,
    is_empty,
    is_geometry,
    is_missing,
    is_ring,
    is_simple,
    is_valid,
    is_valid_input,
    is_valid_reason,
    length,
    linearrings,
    linestrings,
    line_interpolate_point,
    line_interpolate_point_normalized,
    line_locate_point,
    line_locate_point_normalized,
    line_merge,
    normalize,
    overlaps,
    points,
    point_on_surface,
    polygons_with_holes,
    polygons_without_holes,
    relate,
    set_coordinates,
    set_srid,
    shared_paths,
    simplify,
    simplify_preserve_topology,
    snap,
    symmetric_difference,
    touches,
    to_wkb,
    to_wkt,
    unary_union,
    union,
    voronoi_polygons,
    within,
)

# Requires GEOS 3.7.0
if geos_version >= (3, 7, 0):
    from pygeos.lib.core import (
        frechet_distance,
        frechet_distance_densify,
        get_z,
        is_ccw,
    )

# Requires GEOS 3.7.0
if geos_version >= (3, 8, 0):
    from pygeos.lib.core import build_area, coverage_union, make_valid

from pygeos.lib.geom import get_parts
