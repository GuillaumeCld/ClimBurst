import xarray as xr
import pandas as pd
import numpy as np
from typing import List
from numba import njit, float64, int32




def interval_zscore(index, current_year, length, ts, n_year, len_year):
    """
    Compute the z-score of the interval starting at the given index and of given length.

    Parameters:
    - index: integer, starting index of the interval
    - current_year: integer, current year of the starting index
    - length: integer, length of the interval
    - ts: numpy array of floats, time series data
    - n_year: integer, number of years in the time series
    - len_year: integer, length of each year's data

    Returns:
    - float, z-score of the interval
    """
    interval_value = np.sum(ts[index : index + length])
    start_index = index - current_year * len_year
    brick = np.empty(n_year, dtype=np.float64)

    for year in range(n_year):
        start = start_index + year * len_year
        brick[year] = np.sum(ts[start : start + length])

    avr = np.mean(brick)
    std = np.std(brick)

    if np.isclose(std, 0.0):
        z_score = 0.0
    else:
        z_score = (interval_value - avr) / std

    return z_score



def find_consecutive_sequences(arr: List[int]) -> List[List[int]]:
    """
    Find consecutive sequences in the given list of index.

    Parameters:
    - list: list of integers

    Returns:
    - list: list of lists of integers
    """
    # Sort the array
    arr.sort()

    sequences = []
    current_sequence = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_sequence.append(arr[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [arr[i]]

    sequences.append(current_sequence)

    return sequences
