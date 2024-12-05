import os
import matplotlib
import numpy as np
from netCDF4 import Dataset
from pyspark import SparkContext, SparkConf
from src.utils.post_process import merge_bursts
from src.burst_detection.python.exhaustive_search import compute_all_bursts
from typing import Callable

import time


def split_grid_into_chunks(grid_file_path, parameter, n_worker):
    # Load the NetCDF file
    data = Dataset(grid_file_path)

    # Get indices of non-masked time series
    non_nan_indices = np.argwhere(data[parameter][0, :, :])

    # Determine the number of non-masked time series to be assigned to each worker
    num_non_nan = len(non_nan_indices)
    chunk_sizes = [num_non_nan // n_worker +
                   (1 if x < num_non_nan % n_worker else 0) for x in range(n_worker)]

    # Split the indices into chunks
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(non_nan_indices[start:end])
        start = end

    # Convert chunks to lists of coordinates
    chunked_coords = [chunk for chunk in chunks]

    return chunked_coords


def compute_bursts_for_chunk(
    chunk_coords,
    datafile_path,
    parameter,
    n_year,
    len_year,
    threshold,
    method,
    min_length,
    max_length,
    absolute_threshold
):
    data = Dataset(datafile_path)

    rows = np.unique(chunk_coords[:, 0])
    rows = np.arange(rows[0], rows[-1]+1)
    cols = np.unique(chunk_coords[:, 1])
    cols = np.arange(cols[0], cols[-1]+1)


    chunk_data = data[parameter][:, rows, cols].data.astype(np.float64)

    results = []
    for (i, j) in chunk_coords:
        ts = chunk_data[:, i-rows[0], j-cols[0]]
        ts_bursts = merge_bursts(method(
            ts,
            n_year,
            threshold,
            len_year,
            min_length,
            max_length,
            absolute_threshold))
        results.append(((i, j), ts_bursts))


    return results


def grid_bursts_distributed(
    grid_file_path: str,
    n_job: int,
    n_year: int,
    len_year: int,
    threshold: float,
    parameter: str,
    method: Callable = compute_all_bursts,
    min_length: int = 1,
    max_length: int = 365,
    absolute_threshold: bool = True,
) -> np.ndarray:
    start_time = time.time()
    # Initialize SparkContext
    conf = (
        SparkConf()
        .setAppName("Grid Chunk Processing")
        .set("spark.driver.memory", "10g")
        .set("spark.driver.maxResultSize", "0")
        .set('spark.ui.showConsoleProgress', False)
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    assert os.path.exists(grid_file_path), "Data file does not exist"

    data = Dataset(grid_file_path)

    if "latitude" in data.variables:
        n_lat = data["latitude"].size
        n_lon = data["longitude"].size

    elif "lat" in data.variables:
        n_lat = data["lat"].size
        n_lon = data["lon"].size

    else:
        raise KeyError("No latitude key found in data")

    # Split the grid into chunks
    chunk_coords = split_grid_into_chunks(grid_file_path, parameter, n_job*4)

    # print("Creating grid_chunks_rdd...")
    grid_chunks_rdd = sc.parallelize(chunk_coords, n_job)
    del chunk_coords
    # Apply the function to each grid chunk and collect results
    results_rdd = grid_chunks_rdd.map(
        lambda chunk_indices: compute_bursts_for_chunk(
            chunk_indices,
            grid_file_path,
            parameter,
            n_year,
            len_year,
            threshold,
            method,
            min_length,
            max_length,
            absolute_threshold
        )
    )
    del grid_chunks_rdd
    collected_results = results_rdd.flatMap(lambda x: x).collect()

    # Reconstruct the full grid with ts_bursts
    ts_bursts_grid = np.empty((n_lat, n_lon), dtype=object)
    ts_bursts_grid[:] = [[[] for _ in range(n_lon)] for _ in range(n_lat)]
    for (lat, lon), bursts in collected_results:
        ts_bursts_grid[lat, lon] = bursts

    del collected_results
    sc.stop()
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")
    return ts_bursts_grid



