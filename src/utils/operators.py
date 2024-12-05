import numpy as np
from numba import njit


@njit(inline="always")
def overlapping(
    burst1: tuple[int, int, int],
    burst2: tuple[int, int, int],
    min_overlap: int = 1,
    max_overlap: int = -1,
) -> bool:
    """
    Check if two bursts are overlapping based on a minimum and a maximum number of days overlaping.

    Parameters:
    - burst1: tuple of integers (start, end, sign)
    - burst2: tuple of integers (start, end, sign)
    - min_overlap: integer, number of days of overlap
    - max_overlap: integer, maximum number of days of overlap

    Returns:
    - boolean
    """
    s1, e1, z1 = burst1
    s2, e2, z2 = burst2

    # Check if the signs are the same
    if np.sign(z1) != np.sign(z2):
        return False

    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)

    overlap_days = overlap_end - overlap_start + 1

    if max_overlap == -1:
        return overlap_days >= min_overlap
    
    return min_overlap <= overlap_days <= max_overlap


def intersection(
    interval1: tuple[int, int, int],
    interval2: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Compute the intersection of two intervals. There is an intersection if the intervals overlap

    Parameters:
    - interval1: tuple of integers (start, end, z_score)
    - interval2: tuple of integers (start, end, z_score)

    Returns:
    - tuple of integers (start, end, z_score) or None if the intervals do not intersect

    """
    s1, e1, z_score1 = interval1
    s2, e2, z_score2 = interval2

    start = max(s1, s2)
    end = min(e1, e2)
    if start > end and np.sign(z_score1) != np.sign(z_score2):
        return None

    z_score = min(np.abs(z_score1), np.abs(z_score2))
    return (start, end, z_score)


@njit(inline="always")
def containment(
    interval1: tuple[int, int, int],
    interval2: tuple[int, int, int],
) -> bool:
    """
    Check if interval1 contains interval2 i.e interval2 c interval1 and z_score1 >= z_score2

    Parameters:
    - interval1: tuple of integers (start, end, z_score)
    - interval2: tuple of integers (start, end, z_score)

    Returns:
    - boolean
    """
    s1, e1, z1 = interval1
    s2, e2, z2 = interval2

    return (
        (np.sign(z1) == np.sign(z2))
        and (np.abs(z1) >= np.abs(z2))
        and (s1 <= s2)
        and (e1 >= e2)
    )


@njit(inline="always")
def threshold_containment(
    interval1: tuple[int, int, int],
    interval2: tuple[int, int, int],
) -> bool:
    """
    Check if interval1 contains interval2 i.e interval2 c interval1

    Parameters:
    - interval1: tuple of integers (start, end, z_score)
    - interval2: tuple of integers (start, end, z_score)

    Returns:
    - boolean
    """
    s1, e1, z1 = interval1
    s2, e2, z2 = interval2

    return (
        (np.sign(z1) == np.sign(z2))
        and (s1 <= s2)
        and (e1 >= e2)
    )