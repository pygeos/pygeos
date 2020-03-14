from concurrent.futures import ThreadPoolExecutor
import pygeos
import numpy as np

def test_nogil_while_editing():
    a = pygeos.points(np.arange(1000), np.zeros(1000))
    b = pygeos.points(np.zeros(1000), np.arange(1000))

    def edit_arr(arr):
        for i in range(arr.shape[0]):
            arr[i] = None

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(pygeos.distance, a, b)
        executor.submit(edit_arr, a)
        executor.submit(edit_arr, b)
