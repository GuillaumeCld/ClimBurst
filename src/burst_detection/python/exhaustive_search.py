"""
"""
import numpy as np


def compute_all_bursts(ts, n_year, threshold, len_year, min_length=1, max_length=365, abs=True):
    """
    """
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = []

    if abs:
        z_score_filter_fn = lambda z_score, threshold: np.where(np.abs(z_score) >= threshold)[0]
    elif threshold > 0:
        z_score_filter_fn = lambda z_score, threshold: np.where(z_score >= threshold)[0]
    else:
        z_score_filter_fn = lambda z_score, threshold: np.where(z_score <= threshold)[0]
        print(f"Threshold: {threshold}")

    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        compute_length_1(v, n_year, threshold, len_year, all_bursts, z_score_filter_fn)
    else:
        compute_initial_length(
            ts, v, min_length, threshold, len_year, all_bursts, z_score_filter_fn)

    end_index = n + 1 - min_length
    for length in range(min_length+1, max_length+1):
        end_index -= 1
        incremental_compute(ts, v, length, threshold,
                            len_year, all_bursts, end_index, z_score_filter_fn)

    return all_bursts

def compute_z_score(v, threshold, all_bursts, n_valid_years, max_sequence, len_year, length, filter_fn):
    # Z-score computation
    # First loop for the condition: interval < max_sequence
    # for those intervals the last valid year is included
    for interval in range(max_sequence):
        column = v[:n_valid_years, interval]
        mean = np.add.reduce(column) / n_valid_years
        squared_mean = np.add.reduce(np.square(column)) / n_valid_years
        std = np.sqrt(squared_mean - mean*mean)
        z_score = (column - mean) / std
        flagged_years = filter_fn(z_score, threshold)
        for y in flagged_years:
            start = interval + y * len_year
            all_bursts.append(
                (start, start+length-1, z_score[y]))

    # Second loop for the condition: interval >= max_sequence
    # for those intervals the last valid year is excluded
    n_valid_years -= 1
    for interval in range(max_sequence, len_year):
        column = v[:n_valid_years, interval]
        mean = np.add.reduce(column) / n_valid_years
        squared_mean = np.add.reduce(np.square(column)) / n_valid_years
        std = np.sqrt(squared_mean - mean*mean)
        z_score = (column - mean) / std
        flagged_years = filter_fn(z_score, threshold)
        for y in flagged_years:
            start = interval + y * len_year
            all_bursts.append(
                (start, start+length-1, z_score[y]))

def compute_length_1(v, n_year, threshold, len_year, all_bursts, filter_fn):
    for interval in range(len_year):
        column = v[:, interval]
        mean = np.add.reduce(column) / n_year
        squared_mean = np.add.reduce(np.square(column)) / n_year
        std = np.sqrt(squared_mean - mean*mean)
        z_score = (column - mean) / std
        flagged_years = filter_fn(z_score, threshold)
        for y in flagged_years:
            start = interval + y * len_year
            all_bursts.append(
                (start, start, z_score[y]))


def compute_initial_length(ts, v, length, threshold, len_year, all_bursts, filter_fn):
    end_index = len(ts) - length + 1
    end_year = end_index // len_year
    # Compute aggreagates, vectorized by year
    v[0, 0] = np.add.reduce(ts[:length])
    for j in range(1, len_year):
        v[0, j] = v[0, j-1] + ts[j+length-1] - ts[j-1]

    for year in range(1, end_year):
        start = year * len_year
        v[year, 0] = v[year-1, len_year-1] + ts[start+length-1] - ts[start-1]
        for j in range(1, len_year):
            v[year, j] = v[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

    max_sequence = len_year - length + 1
    start = end_year * len_year
    v[end_year, 0] = v[end_year-1, len_year-1] + ts[start+length-1] - ts[start-1]
    for j in range(1, max_sequence):
        v[end_year, j] = v[end_year, j-1] + \
            ts[start+j+length-1] - ts[start+j-1]

    # Z-score computation
    n_valid_years = end_year + 1
    compute_z_score(v, threshold, all_bursts, n_valid_years,
                    max_sequence, len_year, length, filter_fn)

def incremental_compute(ts, v, length, threshold, len_year, all_bursts, end_index, filter_fn):

    span = len_year
    e_span = end_index % len_year
    end_year = end_index // len_year
    # Compute aggreagates, vectorized by year
    for year in range(end_year):
        start = year * len_year + length - 1
        end = start + len_year
        span = len_year
        v[year, :span] += ts[start:end]

    # Last year special case to handle interval out of bounds
    start = end_year * len_year + length - 1
    end = start + e_span
    v[end_year, :e_span] += ts[start:end]

    n_valid_years = end_year + 1
    compute_z_score(v, threshold, all_bursts,
                    n_valid_years, e_span, len_year, length, filter_fn)