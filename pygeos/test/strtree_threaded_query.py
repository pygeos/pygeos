import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pygeos

## Create data

polygons = pygeos.polygons(np.random.randn(1000, 3, 2))
# needs to be big enough to trigger the segfault
N = 100_000
points = pygeos.points(4 * np.random.random(N) - 2, 4 * np.random.random(N) - 2)


## Slice parts of the arrays -> 4x4 => 16 combinations

n = int(len(polygons) / 4)
polygons_parts = [
    polygons[:n],
    polygons[n : 2 * n],
    polygons[2 * n : 3 * n],
    polygons[3 * n :],
]

n = int(len(points) / 4)
points_parts = [points[:n], points[n : 2 * n], points[2 * n : 3 * n], points[3 * n :]]


## Creating the trees in advance

trees = []

for i in range(4):
    left = points_parts[i]
    tree = pygeos.STRtree(left)
    trees.append(tree)


## The function querying the trees in parallel


def thread_func(idxs):
    i, j = idxs
    tree = trees[i]
    right = polygons_parts[j]
    return tree.query_bulk(right, predicate="contains")


def main():
    with ThreadPoolExecutor() as pool:
        list(pool.map(thread_func, itertools.product(range(4), range(4))))


if __name__ == "__main__":
    main()
