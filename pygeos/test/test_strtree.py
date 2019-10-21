import pygeos
from pygeos import box
import pytest
import numpy as np
import sys
from contextlib import contextmanager
from .common import point, empty


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
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


def test_init_with_none():
    pygeos.STRtree(np.array([None]))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = pygeos.STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_decreases_refcount(point):
        del tree


def test_query_no_geom(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    with pytest.raises(TypeError):
        tree.query(None)


def test_query_empty(tree):
    assert tree.query(empty).size == 0


def test_query_increases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_increases_refcount(point):
        _ = tree.query(box(0, 0, 10, 10))


@pytest.mark.parametrize("envelope,expected_n", [
    (pygeos.points(1, 1), 1),
    (box(0, 0, 1, 1), 2),
    (box(5, 5, 15, 15), 5),
    (pygeos.multipoints([[0, 10], [10, 0]]), 10),  # it queries by envelope
])
def test_query(tree, envelope, expected_n):
    assert tree.query(envelope).size == expected_n
