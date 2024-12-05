from src.utils.operators import containment, threshold_containment
from numba import njit, float64, int64
import numpy as np
from numba.typed import List


def sort_bursts(all_bursts):
    all_bursts.sort(key=lambda x: (x[0], -x[1]))
    unique_bursts = []
    last = None

    for burst in all_bursts:
        if burst != last:
            unique_bursts.append(burst)
            last = burst

    return unique_bursts

def merge_bursts(sequences):
    """
    Compute the Merged bursts representation i.e. the union of the given sequences with z_score of the same sign.

    Parameters:
    - sequences: list of lists of integers (i_min, i_max, z_score)

    Returns:
    - list: list of tuples of integers (i_min, i_max, sign(z_score))
    """

    # Separate the sequences with positive and negative z_score
    positive_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] > 0
    ]
    negative_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] < 0
    ]
    # Merge the sequences with positive and negative z_score
    positive_merged_intervals = merge_intervals(positive_intervals, 1)
    negative_merged_intervals = merge_intervals(negative_intervals, -1)
    # Concatenate the positive and negative sequences
    merged_intervals = positive_merged_intervals + negative_merged_intervals

    return merged_intervals

def merge_intervals(intervals, sign):
    """
    Compute the union of the given intervals.

    Parameters:
    - intervals: list of tuples of integers (i_min, i_max)
    - sign: integer

    Returns:
    - list: list of tuples of integers (i_min, i_max, sign)
    """
    if not intervals:
        return []
    # Sort intervals by the starting index
    intervals.sort()
    n = len(intervals)
    # Create the union of the intervals
    merged_intervals = []
    i = 0
    while i < n:
        start, end = intervals[i]

        j = i + 1
        while j < n and intervals[j][0] <= end + 1:
            end = max(end, intervals[j][1])
            j += 1

        merged_intervals.append((start, end, sign))
        i = j

    return merged_intervals

def remove_treshold_contained_bursts(bursts):
    """
    Remove the bursts that are threshold contained in another burst i.e. same z-score sign and interval contained .

    Parameters:
    - bursts: list of tuples of integers (i_min, i_max, z_score)

    Returns:
    - list of tuples of integers (i_min, i_max, z_score)
    """
    # Separate the bursts with positive and negative z_score
    pos_intervals = [bursts for bursts in bursts if bursts[2] > 0]
    neg_intervals = [bursts for bursts in bursts if bursts[2] < 0]

    # Remove the threshold contained bursts
    if len(pos_intervals) > 0:
        pos_intervals = remove_treshold_contained_bursts_signed(pos_intervals)
    if len(neg_intervals) > 0:
        neg_intervals = remove_treshold_contained_bursts_signed(neg_intervals)

    # Concatenate the positive and negative filtered bursts
    all_intervals = pos_intervals + neg_intervals
    return all_intervals

def remove_treshold_contained_bursts_signed(intervals):
    """
    Remove the bursts that are included in another burst.

    Parameters:
    - intervals: list of tuples of integers (i_min, i_max, z_score) of same sign

    Returns:
    - list of tuples of integers (i_min, i_max, z_score)
    """
    # Sort the intervals by the starting index and decreasing ending index
    intervals.sort(key=lambda x: (x[0], -x[1]))
    processed = []
    n = len(intervals)
    i = 1
    current = intervals[0]
    while i < n:
        if intervals[i][1] > current[1]:
            processed.append(current)
            current = intervals[i]
        i += 1
    processed.append(current)
    return processed

