from concurrent.futures import ThreadPoolExecutor
import pygeos
import numpy as np
import pytest


@pytest.mark.parametrize("func", [pygeos.area, pygeos.length])
def test_not_writable_when_multithreading_1(func):
    a = pygeos.box(0, 0, np.arange(10000), np.arange(10000))

    def edit_arr(arr):
        for i in range(arr.shape[0]):
            arr[i] = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(func, a)
        with pytest.raises(ValueError):
            executor.submit(edit_arr, a).result()
        future.result()


@pytest.mark.parametrize("func", [pygeos.distance, pygeos.hausdorff_distance])
def test_no_editing_when_multithreading_2(func):
    a = pygeos.points(np.arange(1000), np.zeros(1000))
    b = pygeos.points(np.zeros(1000), np.arange(1000))

    def edit_arr(arr):
        for i in range(arr.shape[0]):
            arr[i] = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(func, a, b)
        with pytest.raises(ValueError):
            executor.submit(edit_arr, a).result()
        future.result()
