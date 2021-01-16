import numpy as np


def moving_average(xs: np.ndarray, window_size: int) -> np.ndarray:
    """
    :param xs: a 1-D array of floating points.
    :param window_size: the number of points to average over.
    :return: an array like xs, where every entry is the average of window_size
             points in xs. Thus, entry k is the average of [k, k-1, ...].
    """
    if len(xs) <= window_size:
        return np.cumsum(xs) / (np.arange(len(xs)) + 1)
    return np.concatenate(
        [
            np.cumsum(xs)[: window_size - 1] / (np.arange(window_size - 1) + 1),
            np.convolve(xs, np.ones([window_size]) / window_size, mode="valid"),
        ]
    )
