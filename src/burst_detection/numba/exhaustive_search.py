"""
"""
import numpy as np
from numba import njit, float64, int32, int64, boolean
from numba.typed import List




def compute_all_bursts(ts, n_year, threshold, len_year, min_length=1, max_length=365, abs=True):
    """
    """
    if abs:
        all_bursts = compute_all_bursts_numba_abs(
            ts, n_year, threshold, len_year, min_length, max_length)
    elif threshold > 0:
        all_bursts = compute_all_bursts_numba_pos(
            ts, n_year, threshold, len_year, min_length, max_length)
    else:
        all_bursts = compute_all_bursts_numba_neg(
            ts, n_year, threshold, len_year, min_length, max_length)

    return all_bursts


@njit((float64[:], int64, float64, int64, int64, int64), cache=True, fastmath=True)
def compute_all_bursts_numba_abs(ts, n_year, threshold, len_year, min_length, max_length):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = List()
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        for i in range(len_year):
            column = v[:n_year, i]
            mean = np.sum(column) / n_year
            squared_mean = np.sum(np.square(column)) / n_year
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y*len_year
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        length = min_length
        end_index = n - length + 1
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        v[0, 0] = np.sum(ts[:length])
        for j in range(1, len_year):
            v[0, j] = v[0, j-1] + ts[j+length-1] - ts[j-1]

        for year in range(1, end_year):
            start = year * len_year
            v[year, 0] = v[year-1, len_year-1] + \
                ts[start+length-1] - ts[start-1]
            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

        max_sequence = len_year - length + 1
        start = end_year * len_year
        v[end_year, 0] = v[end_year-1, len_year-1] + \
            ts[start+length-1] - ts[start-1]
        for j in range(1, max_sequence):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+length-1] - ts[start+j-1]

        n_valid_years = end_year + 1
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

        # Second loop for the condition: interval >= max_sequence
        # for those intervals the last valid year is excluded
        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

    # Z-score computation
    n_valid_years = end_year + 1
    # Finding burst of length 2 <= l <= 256
    for size in range(2, max_length+1):
        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        for j in range(end_year):
            start = j * len_year + size - 1
            end = start + len_year
            v[j, :len_year] += ts[start:end]
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # Z-score computation
        # First loop for the condition: interval < e_span
        # for those intervals the last valid year is included
        n_valid_years = end_year + 1
        for i in range(e_span):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
        # Second loop for the condition: interval >= e_span
        # for those intervals the last valid year is excluded
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
    return all_bursts


@njit((float64[:], int64, float64, int64, int64, int64), cache=True, fastmath=True)
def compute_all_bursts_numba_pos(ts, n_year, threshold, len_year, min_length, max_length):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = List()
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        for i in range(len_year):
            column = v[:n_year, i]
            mean = np.sum(column) / n_year
            squared_mean = np.sum(np.square(column)) / n_year
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score >= threshold)[0]
            for y in flagged_years:
                start = i + y*len_year
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        length = min_length
        end_index = n - length + 1
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        v[0, 0] = np.sum(ts[:length])
        for j in range(1, len_year):
            v[0, j] = v[0, j-1] + ts[j+length-1] - ts[j-1]

        for year in range(1, end_year):
            start = year * len_year
            v[year, 0] = v[year-1, len_year-1] + \
                ts[start+length-1] - ts[start-1]
            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

        max_sequence = len_year - length + 1
        start = end_year * len_year
        v[end_year, 0] = v[end_year-1, len_year-1] + \
            ts[start+length-1] - ts[start-1]
        for j in range(1, max_sequence):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+length-1] - ts[start+j-1]

        n_valid_years = end_year + 1
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

        # Second loop for the condition: interval >= max_sequence
        # for those intervals the last valid year is excluded
        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score >= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

    # Z-score computation
    n_valid_years = end_year + 1
    # Finding burst of length 2 <= l <= 256
    for size in range(2, max_length+1):
        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        for j in range(end_year):
            start = j * len_year + size - 1
            end = start + len_year
            v[j, :len_year] += ts[start:end]
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # Z-score computation
        # First loop for the condition: interval < e_span
        # for those intervals the last valid year is included
        n_valid_years = end_year + 1
        for i in range(e_span):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score >= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
        # Second loop for the condition: interval >= e_span
        # for those intervals the last valid year is excluded
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score >= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
    return all_bursts


@njit((float64[:], int64, float64, int64, int64, int64), cache=True, fastmath=True)
def compute_all_bursts_numba_neg(ts, n_year, threshold, len_year, min_length, max_length):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    all_bursts = List()
    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        for i in range(len_year):
            column = v[:n_year, i]
            mean = np.sum(column) / n_year
            squared_mean = np.sum(np.square(column)) / n_year
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score <= threshold)[0]
            for y in flagged_years:
                start = i + y*len_year
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        length = min_length
        end_index = n - length + 1
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        v[0, 0] = np.sum(ts[:length])
        for j in range(1, len_year):
            v[0, j] = v[0, j-1] + ts[j+length-1] - ts[j-1]

        for year in range(1, end_year):
            start = year * len_year
            v[year, 0] = v[year-1, len_year-1] + \
                ts[start+length-1] - ts[start-1]
            for j in range(1, len_year):
                v[year, j] = v[year, j-1] + ts[start+j+length-1] - ts[start+j-1]

        max_sequence = len_year - length + 1
        start = end_year * len_year
        v[end_year, 0] = v[end_year-1, len_year-1] + \
            ts[start+length-1] - ts[start-1]
        for j in range(1, max_sequence):
            v[end_year, j] = v[end_year, j-1] + \
                ts[start+j+length-1] - ts[start+j-1]

        n_valid_years = end_year + 1
        for interval in range(max_sequence):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score <= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

        # Second loop for the condition: interval >= max_sequence
        # for those intervals the last valid year is excluded
        n_valid_years -= 1
        for interval in range(max_sequence, len_year):
            column = v[:n_valid_years, interval]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score <= threshold)[0]
            for y in flagged_years:
                start = interval + y * len_year
                all_bursts.append(
                    (start, start+length-1, z_score[y]))

    # Z-score computation
    n_valid_years = end_year + 1
    # Finding burst of length 2 <= l <= 256
    for size in range(2, max_length+1):
        end_index -= 1
        e_span = end_index % len_year
        end_year = end_index // len_year
        # Compute aggreagates, vectorized by year
        for j in range(end_year):
            start = j * len_year + size - 1
            end = start + len_year
            v[j, :len_year] += ts[start:end]
        # Last year special case to handle interval out of bounds
        start = end_year * len_year + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]
        # Z-score computation
        # First loop for the condition: interval < e_span
        # for those intervals the last valid year is included
        n_valid_years = end_year + 1
        for i in range(e_span):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score <= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
        # Second loop for the condition: interval >= e_span
        # for those intervals the last valid year is excluded
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            flagged_years = np.where(z_score <= threshold)[0]
            for y in flagged_years:
                start = i + y * len_year
                all_bursts.append(
                    (start, start+size-1, z_score[y]))
    return all_bursts
