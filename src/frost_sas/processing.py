import logging
from typing import Any

import numpy as np
from numba import njit, prange
from scipy.signal import correlate

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    # print to both a file and to console
    handlers=[logging.FileHandler("processing.log", "w"), logging.StreamHandler()],
)


def match_filter(
    rx_data: np.ndarray[tuple[Any, Any], np.dtype[np.complex128]],
    tx_data: np.ndarray,
):
    if rx_data.ndim != 2:
        raise ValueError(f"Expected 2D rx_data, got {rx_data.ndim}D.")

    if tx_data.ndim == 1:
        tx_data = tx_data[:, np.newaxis]  # shape (N, 1)

    return correlate(rx_data, tx_data, mode="same", method="auto")


# TODO does not produce correct looking results, also slower
@njit(parallel=True)
def match_filter_numba(
    rx_data: np.ndarray[tuple[Any, Any], np.dtype[np.complex128]], tx_data: np.ndarray
):
    matched_data = np.empty_like(rx_data)
    start = (tx_data.size - 1) // 2

    # forced to use numpy correlate which doesn't use FFT method, however it can be parallelized
    for i in prange(rx_data.shape[0]):
        full = np.correlate(rx_data[i, :], tx_data, mode="full")
        matched_data[i, :] = full[start : start + rx_data.shape[1]]

    return matched_data
