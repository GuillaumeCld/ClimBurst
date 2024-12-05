import numpy as np
import pandas as pd


def remove_grid_leap_days(grid, ts_date):
    """
    Remove leap days for all time series in a grid of time series.

    Parameters:
    - grid: numpy array of shape (nt, nx, ny)
    - ts_date: numpy array of shape (nt,) with the dates

    Returns:
    - numpy array of shape (nt, nx, ny) without leap days
    """
    nt, nx, ny = grid.shape
    n_year = nt // 365
    new_grid = np.empty((365 * n_year, nx, ny))

    for i in range(nx):
        for j in range(ny):
            ts = grid[:, i, j]
            new_grid[:, i, j] = remove_leap_days(ts, ts_date)

    return new_grid


def remove_leap_days(ts, ts_date):
    data = {"Date": ts_date, "Value": ts}
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))]
    
    return df["Value"].values