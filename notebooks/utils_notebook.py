# Computation
import pandas as pd
import numpy as np
from numba import njit, float64, int64, int32
# Plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cf
# Misc
import copy


def plot_table(bursts_df):
    colors = {
    'Hot': '#FFDDC1',  # Light red for Hot bursts
    'Cold': '#C1D3FF'  # Light blue for Cold bursts
    }
    cell_colors = [[colors.get(value, "white") for value in bursts_df["Type"]] for _ in bursts_df.columns]
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Start Date</b>", "<b>End Date</b>", "<b>Length</b>", "<b>Type</b>"],
            fill_color="lightgrey",
            align="center",
            font=dict(color="black", size=14)
        ),
        cells=dict(
            values=[bursts_df[col] for col in bursts_df.columns],
            fill_color=cell_colors,
            align="center",
            font=dict(color="black", size=12)
        )
    )])
    return fig


def plot_bars(yearly_sum):
    fig = go.Figure()
    # Add hot days as a bar trace
    fig.add_trace(go.Bar(
        x=yearly_sum['year'],
        y=yearly_sum['hot'],
        name='Hot',
        marker_color='red'
    ))
    # Add cold days as a bar trace
    fig.add_trace(go.Bar(
        x=yearly_sum['year'],
        y=yearly_sum['cold'],
        name='Cold',
        marker_color='blue'
    ))
    return fig


def compute_statistics(ts, ts_date, union_bursts, len_year):
    burst_years = list(set([burst[0] // len_year for burst in union_bursts]))
    bursts_data = []
    hot_bursts = [burst for burst in union_bursts if burst[-1] > 0]
    cold_bursts = [burst for burst in union_bursts if burst[-1] < 0]
    for (start, end, _) in hot_bursts:
        length = end - start + 1
        bursts_data.append([ts_date[start].strftime("%d %b %Y"), ts_date[end].strftime("%d %b %Y"), length, 'Hot'])
    for (start, end, _) in cold_bursts:
        length = end - start + 1
        bursts_data.append([ts_date[start].strftime("%d %b %Y"), ts_date[end].strftime("%d %b %Y"), length, 'Cold'])

    bursts_df = pd.DataFrame(bursts_data, columns=['Start Date', 'End Date', 'length', 'Type'])
    daylies_hot = np.zeros_like(ts)
    daylies_cold = np.zeros_like(ts)
    for (start, end, score) in hot_bursts:
        daylies_hot[start:end+1] = 1
    for (start, end, score) in cold_bursts:
        daylies_cold[start:end+1] = 1
    data = {'date': ts_date,
            'hot': daylies_hot,
            'cold': daylies_cold}

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    yearly_sum = df.groupby('year')[['hot', 'cold']].sum().reset_index()

    return bursts_df, yearly_sum



def show_connected_component(cc, n_year, ts_date, lat, lon, grid_burst, values, lon_min, lon_max, lat_min, lat_max):
    min_s = n_year*365
    max_e = -1

    grid = np.zeros((lat,lon))
    for (i,j,burst_id) in cc:
        grid[i,j] = 1
        burst = grid_burst[i,j][burst_id]
        s, e = burst[:2]
        min_s = min(min_s, s)
        max_e = max(max_e, e)
    if burst[-1] < 0:
        cmap = plt.cm.YlGnBu
    else:
        cmap = plt.cm.YlOrRd
    start_date = ts_date[min_s].strftime("%d-%m-%Y")
    end_date = ts_date[max_e].strftime("%d-%m-%Y")
    plt.figure(figsize=(12, 6))
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    img = ax.imshow(grid, extent=[lon_min, lon_max, lat_min, lat_max], 
                transform=ccrs.PlateCarree(), origin='upper', aspect='auto', cmap=cmap)
    ax.coastlines(resolution='50m', linewidth=1.5)
    ax.add_feature(cf.BORDERS, linestyle=':', linewidth=1.2)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    #cbar = plt.colorbar(img, orientation='vertical', pad=0.02, aspect=50, boundaries=bounds, ticks=bounds)
    plt.title(f"Connected Components ranging from {start_date} to {end_date}")
    plt.show()

    if max_e - min_s == 0:
        return grid
    
    min_s = n_year*365
    max_e = -1
    min_i, min_j = lat, lon
    max_i, max_j = 0, 0

    unique_series = []
    for (i, j, burst_id) in cc:
        burst = grid_burst[i, j][burst_id]
        s, e = burst[:2]
        min_s = min(min_s, s)
        max_e = max(max_e, e)
        min_i = min(min_i, i)
        min_j = min(min_j, j)
        max_i = max(max_i, i)
        max_j = max(max_j, j)
        if not (i, j) in unique_series:
            unique_series.append((i, j))
    
    length = max_e - min_s + 1
    densities = np.zeros(length)
    h = max_i - min_i + 1
    l = max_j - min_j + 1
    grid = np.zeros((h, l))
    grid_start = np.zeros((h, l)) + n_year * 365
    grid_end = np.zeros((h, l)) - 1
    
    for (i, j, burst_id) in cc:
        burst = grid_burst[i, j][burst_id]
        s, e = burst[:2]
        
        length = e - s + 1
        densities[s-min_s: e-min_s+1] += 1
        grid[i-min_i, j-min_j] += length
        grid_start[i-min_i, j-min_j] = min(s, grid_start[i-min_i, j-min_j])
        grid_end[i-min_i, j-min_j] = max(e, grid_end[i-min_i, j-min_j])
    
    densities /= len(unique_series)
    
    for i in range(min_i, max_i+1):
        for j in range(min_j, max_j+1):
            if np.isnan(values[0, i, j]):
                grid[i-min_i, j-min_j] = np.nan
    
    grid = np.where(grid == 0, np.nan, grid)
    grid_start = np.where(np.isnan(grid), np.nan, grid_start) - min_s
    grid_end = np.where(np.isnan(grid), np.nan, grid_end) - min_s
    
    # Discretize colormaps for grid
    cmap = plt.cm.viridis
    step = max(1,  int(np.nanmax(grid) - np.nanmin(grid[np.where(grid != 0)])+1)//10)
    bounds = np.arange(np.nanmin(grid[np.where(grid != 0)]), np.nanmax(grid) + 1, step)        
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    first_start_date, last_start_date = np.nanmin(grid_start), np.nanmax(grid_start)
    start_length = last_start_date - first_start_date +1
    date_bounds_start = np.linspace(0, start_length+1, num=11) 
    date_labels_start = ts_date[(date_bounds_start+min_s).astype(int)].strftime('%d-%b')
    
    first_end_date, last_end_date = np.nanmin(grid_end), np.nanmax(grid_end)
    end_length = last_end_date - first_end_date + 1
    date_bounds_end = np.linspace(0, end_length+1, num=11) 
    date_labels_end = ts_date[(date_bounds_end+min_s).astype(int)].strftime('%d-%b')
    
    # Create a figure and 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot the density line plot
    axes[0, 0].plot(ts_date[min_s: max_e+1], densities, color='blue')
    axes[0, 0].grid(True)
    axes[0, 0].set_title(f"Density Plot with {len(unique_series)} series")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Density")
    
    # Plot the heatmap

    heatmap = axes[0, 1].imshow(grid, aspect='auto', cmap=cmap, norm=norm)
    fig.colorbar(heatmap, ax=axes[0, 1], boundaries=bounds, ticks=bounds)
    axes[0, 1].set_title("Number of days per series")
    axes[0, 1].set_xlabel("Longitude Index")
    axes[0, 1].set_ylabel("Latitude Index")
    
    # Plot the grid_start - min_s heatmap with date labels

    start_heatmap = axes[1, 0].imshow(grid_start, aspect='auto', cmap='plasma', norm=mcolors.BoundaryNorm(date_bounds_start, cmap.N))
    cbar = fig.colorbar(start_heatmap, ax=axes[1, 0], boundaries=date_bounds_start, ticks=date_bounds_start)
    cbar.ax.set_yticklabels(date_labels_start)
    axes[1, 0].set_title("Start Day per Series")
    axes[1, 0].set_xlabel("Longitude Index")
    axes[1, 0].set_ylabel("Latitude Index")
    
    
    # Plot the grid_end - min_s heatmap with date labels
    end_heatmap = axes[1, 1].imshow(grid_end, aspect='auto', cmap='plasma', norm=mcolors.BoundaryNorm(date_bounds_end, cmap.N))
    cbar = fig.colorbar(end_heatmap, ax=axes[1, 1], boundaries=date_bounds_end, ticks=date_bounds_end)
    cbar.ax.set_yticklabels(date_labels_end)
    axes[1, 1].set_title("End day per series")
    axes[1, 1].set_xlabel("Longitude Index")
    axes[1, 1].set_ylabel("Latitude Index")
    
    
    # Enhance layout
    plt.tight_layout()
    plt.show()



def compute_grid_counts(grid_burst, lat, lon, values):
    grid_count = np.zeros((lat,lon))
    grid_count[:] = np.nan
    grid_count_hot = np.copy(grid_count)
    grid_count_cold = np.copy(grid_count)
    grid_count_days_hot = np.copy(grid_count)
    grid_count_days_cold = np.copy(grid_count)

    coords = [(i,j) for j in range(lon) for i in range(lat) if not np.isnan(values[0, i, j])]

    for (i,j) in coords:
        grid_count[i,j] = len(grid_burst[i,j])
        hots = [burst for burst in grid_burst[i,j] if burst[2]>0]
        colds = [burst for burst in grid_burst[i,j] if burst[2]<0]
        
        grid_count_hot[i,j] = len(hots)
        grid_count_cold[i,j] = len(colds)
        for hot_burst in hots:
            s,e,_ = hot_burst
            if not np.isnan(grid_count_days_hot[i,j]):
                grid_count_days_hot[i,j] += e-s+1 
            else:
                grid_count_days_hot[i,j] = e-s+1 
        for cold_burst in colds:
            s,e,_ = cold_burst
            if not np.isnan(grid_count_days_cold[i,j]):
                grid_count_days_cold[i,j] += e-s+1
            else:
                grid_count_days_cold[i,j] = e-s+1

    return grid_count, grid_count_hot, grid_count_cold, grid_count_days_hot, grid_count_days_cold, coords

def hide_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def compute_grid_bars(ts_date, grid_burst, coords):
    daylies_hot = np.zeros(len(ts_date))
    daylies_cold = np.zeros(len(ts_date))
    series_hot = [[] for _ in range(len(ts_date)//365)]
    series_cold = [[] for _ in range(len(ts_date)//365)]

    for (i, j) in coords:
        for burst in grid_burst[i, j]:
            s, e,z = burst
            if z > 0:
                daylies_hot[s:e + 1] += 1
                year = s//365
                series_hot[year].append((i,j))
            else:
                daylies_cold[s:e + 1] += 1
                year = s//365
                series_cold[year].append((i,j))

    series_hot = np.array([len(set(l)) for l in series_hot])
    series_cold = np.array([len(set(l)) for l in series_cold])

    return daylies_hot, daylies_cold, series_hot, series_cold


def compute_cc_stats(grid_burst, ccs, n_year, first_year, lat, lon, coords):
    cc_stats= []
    grid_start_5 = np.zeros((lat,lon))
    grid_start_5[:] = np.nan
    grid_start_10 = copy.deepcopy(grid_start_5)
    grid_start_15 = copy.deepcopy(grid_start_5)
    grid_start_20 = copy.deepcopy(grid_start_5)
    for (i,j) in coords:
        grid_start_5[i,j] = 0
        grid_start_10[i,j] = 0
        grid_start_15[i,j] = 0
        grid_start_20[i,j] = 0
        
    for cc in ccs:
        min_s = n_year*365
        max_e = -1
        unique_series = []
        for (i,j,burst_id) in cc:
            if not (i,j) in unique_series:
                unique_series.append((i,j))
            burst = grid_burst[i,j][burst_id]
            s, e = burst[:2]

            min_s = min(min_s, s)
            max_e = max(max_e, e)

        length = max_e-min_s+1
        densities = np.zeros(length)
        for (i,j,burst_id) in cc:
            s, e, z = burst
            if z > 0:
                if s <= min_s + max(0,5*length/100):
                    grid_start_5[i,j] += 1
                if s <= min_s + max(0,10*length/100):
                    grid_start_10[i,j] += 1
                if s <= min_s + max(1,15*length/100):
                    grid_start_15[i,j] += 1
                if s <= min_s + max(1,20*length/100):
                    grid_start_20[i,j] += 1

                densities[s-min_s: e-min_s+1] += 1
                
        nb_series = len(unique_series)
        start_year = first_year + min_s // 365    
        cc_stats.append([length, nb_series, start_year, np.sign(burst[-1])])

    return cc_stats, grid_start_5, grid_start_10, grid_start_15, grid_start_20



def cc_time_travel(cc, ts_date, grid_burst, lat, lon, values):
    min_s = ts_date.shape[0]
    max_e = 0
    for vertex in cc:
        i, j, burst_id = vertex
        burst = grid_burst[i,j][burst_id]
        s, e, sign = burst
        min_s = min(min_s, s)
        max_e = max(max_e, e)


    len_cc = max_e - min_s
    cc_bursts_grid = np.zeros((len_cc, lat, lon)) 
    for i in range(lat):
        for j in range(lon):
            if np.isnan(values[0, i, j]):
                cc_bursts_grid[:, i, j] = np.nan
    for i, j, id_burst in cc:
        s, e, z  = grid_burst[i, j][id_burst]
        cc_bursts_grid[s-min_s:e-min_s, i, j ] = 1

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = plt.colormaps.get_cmap("Paired")
    cmap.set_bad(color="darkgrey")
    # Initialize the image plot
    img = ax.imshow(cc_bursts_grid[0], cmap=cmap)

    date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    formatted_dates = [f'Date: {date.strftime("%Y-%m-%d")}' for date in ts_date[min_s:max_e+1]]
   
    plt.close()
    # Update function for animation
    def update(frame):
        img.set_array(cc_bursts_grid[frame])
        date_text.set_text(formatted_dates[frame])
        return img,
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(cc_bursts_grid), interval=200)

    return anim



@njit((float64[:], int64, int64))
def compute_daily_zscore(ts, n_year, len_year):
    n = len(ts)
    v = np.copy(ts).reshape(n_year, len_year)
    z_per_day = np.zeros((len(ts)))

    # Finding burst of length 1 (day) only the z-score are needed
    for i in range(len_year):
        column = v[:n_year, i]
        mean = np.sum(column) / n_year
        squared_mean = np.sum(np.square(column)) / n_year
        std = np.sqrt(squared_mean - mean*mean)
        z_score = (column - mean) / std
        for y, z in enumerate(z_score):
            start = i + y*len_year
            z_per_day[start] = z

    end_index = n
    # Finding burst of length 2 <= l <= 256
    for size in range(2, 366):
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
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z
        # Second loop for the condition: interval >= e_span
        # for those intervals the last valid year is excluded
        n_valid_years = end_year
        for i in range(e_span, len_year):
            column = v[:n_valid_years, i]
            mean = np.sum(column) / n_valid_years
            squared_mean = np.sum(np.square(column)) / n_valid_years
            std = np.sqrt(squared_mean - mean*mean)
            z_score = (column - mean) / std
            for y, z in enumerate(z_score):
                start = i + y*len_year
                for day in range(size):
                    abs_val = np.abs(z_per_day[start+day])
                    if abs_val < np.abs(z):
                        z_per_day[start+day] = z
    return z_per_day


    

@njit((float64[:], int32, int32))
def SWT_per_year(ts: np.ndarray, n_year: int, len_year: int):
    """
    Calculate the Shifted Wavelet Tree (SWT) for each year of a time series.

    Parameters:
    - ts The time series data.
    - n_year The number of years in the time series.
    - len_year The length of each year in the time series.

    Returns:
    list: A list of SWTs for each year, where each SWT is a list of levels.
    """
    n_level = 9
    levels_matrices = []

    brick_sizes = np.empty(n_level, dtype=np.int32)
    ns_bricks = np.empty(n_level, dtype=np.int32)
    for i in range(n_level):
        size = 2 << i
        brick_sizes[i] = size
        n_bricks_wavelet = len_year // size
        n_bricks = n_bricks_wavelet * 2

        if size // 2 + n_bricks_wavelet * size < len_year:
            n_bricks += 2
        else:
            n_bricks += 1
        ns_bricks[i] = n_bricks

    # For each level of the SWT
    for level in range(n_level):
        brick_size = brick_sizes[level]
        n_bricks = ns_bricks[level]
        level_matrice = np.empty(
            (n_year, n_bricks), dtype=np.float64)

        for year in range(n_year):
            i_year = year * len_year

            if level == 0:
                m = 1
                previous_aggregate = ts[i_year: i_year + len_year]
            else:
                m = 2
                previous_aggregate = levels_matrices[level - 1][year, :]

            for j in range(n_bricks - 2):
                level_matrice[year, j] = (
                    previous_aggregate[j * m] + previous_aggregate[(j + 1) * m]
                )

            if year != n_year - 1 or brick_size == 2:
                level_matrice[year, n_bricks - 2] = (
                    previous_aggregate[(n_bricks - 2) * m]
                    + previous_aggregate[(n_bricks - 1) * m]
                )
            else:
                level_matrice[year, n_bricks - 2] = np.nan
            if year != n_year - 1 and not (year == n_year - 2 and brick_size == 512):
                start = (
                    i_year + (n_bricks - 1) * brick_size // 2 + brick_size // 2
                )
                level_matrice[year, n_bricks - 1] = previous_aggregate[
                    (n_bricks - 1) * m
                ] + np.sum(ts[start: start + brick_size // 2])
            else:
                level_matrice[year, n_bricks - 1] = np.nan

        levels_matrices.append(level_matrice)
    return levels_matrices



@njit()
def columns_z_scores(column):
    mean = np.nanmean(column)
    std = np.nanstd(column)
    z_scores = (column - mean) / std

    return z_scores




def z_score_SWT(
    levels_matrices: list[np.ndarray], n_year: int, len_year: int, threshold: float, ts: np.ndarray
):
    """
    Compute the z-score of the yearly shifted wavelet transform (SWT) data and return the flagged columns and their flagged bricks (years).

    Parameters:
    - yearly_SWT (Array): The yearly SWT data.
    - n_year (int): The number of years.
    - len_year (int): The length of each year's data.
    - threshold (float): The threshold value for flagging bricks.

    Returns:
    - flagged_columns (List): A list of flagged columns, where each item is a tuple containing the level, brick, and flagged years.
    """

    n_level = 9
    swt = copy.deepcopy(levels_matrices)
    for level in range(n_level):
        level_matrix = levels_matrices[level]
        n_brick = level_matrix.shape[1]

        for brick in range(n_brick):

            column = level_matrix[:, brick]

            column_z_scores = columns_z_scores(column)

            swt[level][:, brick] = column_z_scores

    return swt

def plot_bricks_with_z_score(z_score_SWT, year,  abs_max, ax):


    levels = np.arange(-abs_max, abs_max+0.5, 1)

    # norm = plt.Normalize(minv, maxv)
    norm = BoundaryNorm(boundaries=levels, ncolors=256, extend='both')
    cmap = plt.cm.RdYlBu_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Z-score', ticks=levels)

    a = int(np.ceil(np.log2(512)))-1

    to_draw = []
    for i in range(a):

        size = 2**(i+1)
        n_bricks_wavelet = 365 // size
        n_bricks = n_bricks_wavelet * 2

        if (size//2 + n_bricks_wavelet * size < 365):
            n_bricks += 2
        else:
            n_bricks += 1

        for j in range(0, n_bricks, 1):
            val = z_score_SWT[i][year, j]

            xy = (j * size//2, i+1+(j % 2)*0.4)
            if (j % 2 == 1):
                ls = "--"
                color = "grey"
                label = "Shifted brick"
            else:
                ls = "-"
                color = "black"
                label = "Brick"

            if (np.abs(val) >= 3):
                color = "black"
                rect = patches.Rectangle(
                    xy, size, 0.4, edgecolor=color, facecolor=cmap(norm(val)), label=label, lw=1)
                to_draw.append(rect)

            else:
                rect = patches.Rectangle(
                    xy, size, 0.4, edgecolor="lightgrey", facecolor=cmap(norm(val)), lw=0.5 , label=label)
                ax.add_patch(rect)

    for rect in to_draw:
        ax.add_patch(rect)
