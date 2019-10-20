import pygeos
import pytest
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


@pytest.fixture
def tree():
    geoms = np.array([pygeos.box(0, 0, 1, 1), pygeos.box(10, 10, 11, 11)])
    yield pygeos.STRtree(geoms, 5)


def test_init_with_none():
    pygeos.STRtree(np.array([None]), 5)


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = pygeos.STRtree(arr, 5)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr, 5)
    with assert_decreases_refcount(point):
        del tree


def test_query_no_geom(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    with pytest.raises(TypeError):
        tree.query(None)


def test_query_increases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr, 5)
    with assert_increases_refcount(point):
        _ = tree.query(pygeos.box(0, 0, 10, 10))
