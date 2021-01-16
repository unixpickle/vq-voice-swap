import numpy as np
import pytest

from .smoothing import moving_average


@pytest.mark.parametrize("length", [9, 10, 11, 51])
def test_moving_average(length):
    data = np.random.normal(size=(length,))
    actual = moving_average(data, 10)
    expected = slow_moving_average(data, 10)
    assert np.allclose(actual, expected)


def slow_moving_average(data, window):
    res = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        res[i] = np.mean(data[start : i + 1])
    return res
