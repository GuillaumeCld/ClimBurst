import numpy as np


def zscore(x):
    """
    Compute the z-score of an array x
    """
    size = x.size
    sum = np.add.reduce(x)
    mean = sum / size
    squared_sum = np.add.reduce(x * x)
    std = np.sqrt(squared_sum /size - mean * mean)

    return (x - mean) / std


def zscore_unbiased(x):
    """
    Compute the z-score of an array x with unbiased variance
    """
    size = x.size
    sum = np.add.reduce(x)
    mean = sum / size
    squared_sum = np.add.reduce(x * x)
    var = (squared_sum - (sum * sum) / size) / (size - 1)
    std = np.sqrt(var)

    return (x - mean) / std


def modified_zscore(x):
    """
    Compute the modified z-score of an array x
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return 0.6745 * (x - median) / mad



def box_outlier(x):
    """
    Detect outliers using the box method
    """
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (x < lower_bound) | (x > upper_bound)


def z_score_outliers(x, threshold):
    """
    Detect outliers using the z-score method
    """
    z = zscore(x)
    return np.where(np.abs(z) > threshold)[0]

def zscore_unbiased_outliers(x, threshold):
    """
    Detect outliers using the unbiased z-score method
    """
    z = zscore_unbiased(x)
    return np.where(np.abs(z) > threshold)[0]

def modified_z_score_outliers(x, threshold):
    """
    Detect outliers using the modified z-score method
    """
    z = modified_zscore(x)
    return np.where(np.abs(z) > threshold)[0]

def box_outliers(x, threshold):
    """
    Detect outliers using the box method
    """
    return np.where(box_outlier(x))[0]


def compute_all_bursts(ts, n_year, threshold, len_year, score_filter):

    n = len(ts)
    v = np.copy(ts)
    all_bursts = []
    for i in range(len_year):
        column = []
        for j in range(n_year):
            index = i + j*len_year
            column.append(v[index])

        flagged_years = score_filter(np.array(column), threshold)
        for y in flagged_years:
            start = i + y*len_year
            all_bursts.append(
                (start, start, 1))

    for length in range(2, 90+1):
        n_sequence = n - length + 1
        # Compute the level aggregates
        for i in range(n_sequence):
            v[i] += ts[i+length-1]

        v[n_sequence:] = np.nan
        for interval in range(len_year):
            column = []
            for year in range(n_year):
                index = interval + year*len_year
                column.append(v[index])

            flagged_years = score_filter(np.array(column), threshold)
            for y in flagged_years:
                start = interval + y*len_year
                all_bursts.append(
                    (start, start+length-1, 1))

    return all_bursts


