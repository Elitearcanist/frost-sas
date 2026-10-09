import numpy as np
from numba import njit, prange


@njit(parallel=True)
def match_filter(rx_data: np.ndarray, tx_data: np.ndarray):

    matched_data = np.empty(rx_data.shape, dtype=rx_data.dtype)

    # forced to use numpy correlate which doesn't use FFT method, however it can be parallelized
    for i in prange(rx_data.shape[0]):
        matched_data[i] = np.correlate(rx_data[i], tx_data, mode="same")

    return matched_data
