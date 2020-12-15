import numpy as np
import pygeos


class PointPolygonTimeSuite:
    """Benchmarks running on 100000 points and one polygon"""
    def setup(self):
        self.points = pygeos.points(np.random.random((100000, 2)))
        self.polygon = pygeos.polygons(np.random.random((3, 2)))

    def time_contains(self):
        pygeos.contains(self.points, self.polygon)

    def time_distance(self):
        pygeos.distance(self.points, self.polygon)

    def time_intersection(self):
        pygeos.intersection(self.points, self.polygon)


class IOSuite:
    """Benchmarks I/O operations (WKT and WKB) on a set of 10000 polygons"""
    def setup(self):
        self.to_write = pygeos.polygons(np.random.random((10000, 100, 2)))
        self.to_read_wkt = pygeos.to_wkt(self.to_write)
        self.to_read_wkb = pygeos.to_wkb(self.to_write)

    def time_write_to_wkt(self):
        pygeos.to_wkt(self.to_write)

    def time_write_to_wkb(self):
        pygeos.to_wkb(self.to_write)

    def time_read_from_wkt(self):
        pygeos.from_wkt(self.to_read_wkt)

    def time_read_from_wkb(self):
        pygeos.from_wkb(self.to_read_wkb)


class ConstructiveSuite:
    """Benchmarks constructive functions on a set of 10,000 points"""
    def setup(self):
        self.points = pygeos.points(np.random.random((10000, 2)))

    def time_voronoi_polygons(self):
        pygeos.voronoi_polygons(self.points)

    def time_envelope(self):
        pygeos.envelope(self.points)

    def time_convex_hull(self):
        pygeos.convex_hull(self.points)

    def time_delaunay_triangles(self):
        pygeos.delaunay_triangles(self.points)



class GetParts:
    """Benchmarks for getting individual parts from 100 multipolygons of 100 polygons each"""
    def setup(self):
        self.multipolygons = np.array([pygeos.multipolygons(pygeos.polygons(np.random.random((2, 100, 2)))) for i in range(10000)], dtype=object)

    def time_get_parts(self):
        """Cython implementation of get_parts"""
        pygeos.get_parts(self.multipolygons)

    def time_get_parts_python(self):
        """Python / ufuncs version of get_parts"""

        parts = []
        for i in range(len(self.multipolygons)):
            num_parts = pygeos.get_num_geometries(self.multipolygons[i])
            parts.append(pygeos.get_geometry(self.multipolygons[i], range(num_parts)))

        parts = np.concatenate(parts)


class STRtree:
    """Benchmarks queries against STRtree"""

    def setup(self):
        # create irregular polygons my merging overlapping point buffers
        self.polygons = pygeos.get_parts(
            pygeos.union_all(
                pygeos.buffer(pygeos.points(np.random.random((2000, 2)) * 500), 5)
            )
        )
        self.tree = pygeos.STRtree(self.polygons)
        # initialize the tree by making a tiny query first
        self.tree.query(pygeos.points(0, 0))

        # create points that extend beyond the domain of the above polygons to ensure
        # some don't overlap
        self.points = pygeos.points((np.random.random((2000, 2)) * 750) - 125)

        self.point_tree = pygeos.STRtree(pygeos.points(np.random.random((2000, 2)) * 750))
        self.point_tree.query(pygeos.points(0,0))

    def time_tree_nearest_points(self):
        self.point_tree.nearest(self.points)

    def time_tree_nearest_points_small_max_distance(self):
        # returns >300 results
        self.point_tree.nearest(self.points, max_distance=5)

    def time_tree_nearest_points_large_max_distance(self):
        # measures the overhead of using a distance that would encompass all tree points
        self.point_tree.nearest(self.points, max_distance=1000)

    def time_tree_nearest_poly(self):
        self.tree.nearest(self.points)

    def time_tree_nearest_poly_small_max_distance(self):
        # returns >300 results
        self.tree.nearest(self.points, max_distance=5)

    def time_tree_nearest_poly_python(self):
        # returns all input points

        # use an arbitrary search tolerance that seems appropriate for the density of
        # geometries
        tolerance = 200
        b = pygeos.buffer(self.points, tolerance, quadsegs=1)
        left, right = self.tree.query_bulk(b)
        dist = pygeos.distance(self.points.take(left), self.polygons.take(right))

        # sort by left, distance
        ix = np.lexsort((right, dist, left))
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

        run_start = np.r_[True, left[:-1] != left[1:]]
        run_counts = np.diff(np.r_[np.nonzero(run_start)[0], left.shape[0]])

        mins = dist[run_start]

        # spread to rest of array so we can extract out all within each group that match
        all_mins=np.repeat(mins, run_counts)
        ix = dist == all_mins
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

        # arrays are now roughly representative of what tree.nearest would provide, though
        # some nearest neighbors may be missed if they are outside tolerance
