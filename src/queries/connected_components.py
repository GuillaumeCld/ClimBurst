import numpy as np
from src.utils.operators import overlapping
# from src.test_cases import show_bursts
from collections import defaultdict, deque
import time
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr


def neighbors(i, j, nx, ny):
    """
    Find the indices of all neighbors of a given cell in a grid.

    Parameters:
    - i: integer, row index
    - j: integer, column index
    - nx: integer, number of rows
    - ny: integer, number of columns

    Returns:
    - list of tuples with neighbors' indices
    """
    neighbors = []
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            if 0 <= x < nx and 0 <= y < ny and (x != i or y != j):
                neighbors.append((x, y))

    return neighbors


def find_connected_components(vertices, edges, directed=False):
    # Step 1: Build the adjacency list
    graph = defaultdict(list)
    for v1, v2 in edges:
        graph[v1].append(v2)
        # We do not add v2 -> v1 for directed graphs
        if not directed:
            graph[v2].append(v1)

    # Step 2: Initialize data structures
    visited = set()
    components = []

    # Step 3: Define a BFS to explore each component
    def bfs(start_vertex):
        component = []
        queue = deque([start_vertex])
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                component.append(vertex)
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return component

    # Step 4: Find all connected components
    for vertex in vertices:
        if vertex not in visited:
            component = bfs(vertex)
            components.append(component)

    return components


def compute_connected_components_series(grid_bursts, min_overlap=1, max_overlap=-1):
    nx, ny = grid_bursts.shape
    grid_mark = np.full((nx, ny), False, dtype=bool)

    edges = []
    vertices = []

    for i in range(nx):
        for j in range(ny):
            for burst in grid_bursts[i, j]:
                for neighbor in neighbors(i, j, nx, ny):
                    for neighbor_burst in grid_bursts[neighbor]:
                        if grid_mark[neighbor]:
                            continue
                        if overlapping(burst, neighbor_burst, min_overlap, max_overlap):
                            e1 = (i, j)
                            e2 = neighbor
                            edges.append((e1, e2))
                            break
                vertices.append((i, j))
            grid_mark[i, j] = True

    connected_components = find_connected_components(vertices, edges)

    return connected_components


def compute_connected_components_bursts(grid_bursts, min_overlap=1, directed=False):
    nx, ny = grid_bursts.shape
    grid_mark = np.full((nx, ny), False, dtype=bool)

    edges = []
    vertices = []

    for i in range(nx):
        for j in range(ny):
            for i_burst, burst in enumerate(grid_bursts[i, j]):
                for neighbor in neighbors(i, j, nx, ny):
                    for i_nn_burst, neighbor_burst in enumerate(grid_bursts[neighbor]):
                        if grid_mark[neighbor]:
                            continue

                        if overlapping(burst, neighbor_burst, min_overlap):
                            e1 = (i, j, i_burst)
                            e2 = (*neighbor, i_nn_burst)
                            if directed :
                                # case e2 <-> e1
                                if burst[0] == neighbor_burst[0]:
                                    edges.append((e1, e2))
                                    edges.append((e2, e1))
                                # case:  e1 -> e2
                                elif burst[0] < neighbor_burst[0]:
                                    edges.append((e1, e2))
                                    
                                # case e2 -> e1
                                else:
                                    edges.append((e2, e1))

                            else:
                                edges.append((e1, e2))
                vertices.append((i, j, i_burst))
            grid_mark[i, j] = True

    connected_components = find_connected_components(vertices, edges, directed)

    return connected_components


def main():
    ds = xr.open_dataset("Data/SST/merged_data_processed.nc")
    grid = ds["sst"].values
    ts_date = ds["time"].values.astype("datetime64[D]")
    grid_brut = ds["sst"].values.astype(np.double)
    threshold = 3
    # np.save("Data/SST/test_g.npy", grid_bursts)
    grid_bursts = np.load(f"Data/SST/ts_bursts_grid_{threshold}.npy", allow_pickle=True)
    print("Computing connected components...")
    start_time = time.time()
    connected_components = compute_connected_components_series(grid_bursts, 1)
    end_time = time.time()
    print(
        "Time taken for computing connected components: {:.4f} seconds".format(
            end_time - start_time
        )
    )

    burst_count = np.zeros((grid.shape[1], grid.shape[2]))
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            burst_count[i, j] = len(grid_bursts[i, j])

    plt.figure()
    plt.imshow(burst_count, cmap="hot", interpolation="nearest")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

    max_length = 0
    i_max = 0
    sizes = []
    for ind, cc in enumerate(connected_components):
        sizes.append(len(cc))
        if len(cc) > max_length:
            max_length = len(cc)
            i_max = ind

    plt.figure(figsize=(15, 6))
    plt.hist(sizes, bins=100, color="skyblue", edgecolor="black")
    plt.yscale("log")
    plt.xlabel("Size", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    # Save the plot as an SVG file with tight layout
    plt.savefig(
        "size_distribution_log_scale.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    # Show the plot
    plt.show()

    print("Number of connected components:", len(connected_components))
    print("Largest connected component:", len(connected_components[i_max]), i_max)

    grid_cc = np.zeros((grid.shape[1], grid.shape[2]))
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            if np.isnan(grid[0, i, j]):
                grid_cc[i, j] = np.nan
    for id, cc in enumerate(connected_components):
        for i, j in connected_components[id]:
            grid_cc[i, j] = id

    def create_custom_colormap(num_colors):
        colormaps = ["tab20", "tab20b", "tab20c", "Set3", "Accent", "Paired"]
        colors = []
        for cmap_name in colormaps:
            cmap = plt.get_cmap(cmap_name)
            colors.extend(cmap(np.linspace(0, 1, cmap.N)))
            if len(colors) >= num_colors:
                break
        return ListedColormap(colors[:num_colors])

    num_components = len(connected_components)
    custom_cmap = create_custom_colormap(num_components)

    plt.figure()
    plt.imshow(grid_cc, cmap=custom_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

    random_elements = random.sample(connected_components[i_max], 3)
    for i, j in random_elements:
        show_bursts(grid_brut[:, i, j], ts_date, threshold, 0)


if __name__ == "__main__":
    main()
