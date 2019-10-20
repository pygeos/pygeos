import pygeos
import numpy as np
import sys
from contextlib import contextmanager
from .common import point


@contextmanager
def assert_increases_refcount(obj):
    before = sys.getrefcount(obj)
    yield
    assert sys.getrefcount(obj) == before + 1


@contextmanager
def assert_decreases_refcount(obj):
    before = sys.getrefcount(obj)
    yield
    assert sys.getrefcount(obj) == before - 1


def test_init_with_none():
    pygeos.lib.STRtree(np.array([None]), 5)


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = pygeos.lib.STRtree(arr, 5)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.lib.STRtree(arr, 5)
    with assert_decreases_refcount(point):
        del tree
