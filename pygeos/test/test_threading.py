from concurrent.futures import ThreadPoolExecutor
import pygeos
import numpy as np
import pytest


def test_no_editing_when_multithreading():
    a = pygeos.points(np.arange(1000), np.zeros(1000))
    b = pygeos.points(np.zeros(1000), np.arange(1000))

    def edit_arr(arr):
        for i in range(arr.shape[0]):
            arr[i] = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(pygeos.distance, a, b)
        with pytest.raises(ValueError):
            executor.submit(edit_arr, a).result()
        future.result()


def test_editing_ok_when_single_worker():
    a = pygeos.points(np.arange(1000), np.zeros(1000))
    b = pygeos.points(np.zeros(1000), np.arange(1000))

    def edit_arr(arr):
        for i in range(arr.shape[0]):
            arr[i] = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pygeos.distance, a, b)
        executor.submit(edit_arr, a).result()
        future.result()
