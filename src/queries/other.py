from src.utils.operators import overlapping, containment
import numpy as np
from typing import List, Tuple
from src.utils.operators import overlapping, containment
import numpy as np


def find_bursts_intersecting_with_query(
    query: Tuple[int, int],
    grid_bursts: np.ndarray,
    selection: List[Tuple[int, int]],
    min_overlap: int = 1,
) -> List[Tuple[int, int, Tuple[int, int, float]]]:
    """
    Find all bursts in the grid that intersect with a given query burst.

    Parameters:
    - query: tuple of integers
    - grid_bursts: numpy array of lists of tuples
    - selection: list of tuples with indices of the selected cells
    - min_overlap: integer, minimum number of days of overlap

    Returns:
    - list of tuples with indices of the bursts in the grid and the burst
    """

    list_bursts = []
    for i, j in selection:
        for burst in grid_bursts[i, j]:
            if overlapping(query, burst, min_overlap):
                list_bursts.append((i, j, burst))

    return list_bursts


def find_bursts_containing_query(query, grid_bursts, selection):
    """
    Find all bursts in the grid that contain a given query burst.

    Parameters:
    - query: tuple of integers
    - grid_bursts: numpy array of lists of tuples
    - selection: list of tuples with indices of the selected cells

    Returns:
    - list of tuples with indices of the bursts in the grid and the burst
    """

    list_bursts = []
    for i, j in selection:
        for burst in grid_bursts[i, j]:
            if containment(burst, query):
                list_bursts.append((i, j, burst))

    return list_bursts


def compute_selection_average(grid, selection):
    """
    Compute the average of the grid values for a given selection of cells.

    Parameters:
    - grid: numpy array of shape (nt, nx, ny)
    - selection: list of tuples with indices of the selected cells

    Returns:
    - numpy array of shape (nt,) with the average values
    """
    average = np.zeros(grid.shape[0])
    for i, j in selection:
        average += grid[:, i, j]
    average /= len(selection)

    return average
