import xarray as xr
import numpy as np

def remove_leap_days(ds):
    """
    Remove leap days from a dataset.

    Parameters:
    ds (xarray.Dataset): Dataset with a 'time' coordinate.

    Returns:
    xarray.Dataset: Dataset with leap days removed.
    """
    
    # Convert time to pandas datetime to easily filter leap days
    time_index = ds['time'].to_index()
    is_leap_day = (time_index.month == 2) & (time_index.day == 29)
    
    # Filter out leap days
    ds_filtered = ds.sel(time=~is_leap_day)
    
    # Save to a new NetCDF file
    return ds_filtered
    

def reverse_lat(ds):
    """
    Reverse the latitude of a dataset.
    
    Parameters:
    ds (xarray.Dataset): Dataset with a 'lat' coordinate.

    Returns:
    xarray.Dataset: Dataset with the 'lat' coordinate reversed.
    """
    ds = ds.reindex(lat=ds['lat'][::-1])

    return ds




def propagate_and_collect(grid, i, j):
    """
    Find all connected coordinates in a grid starting from a given coordinate.

    Parameters:
    grid (numpy.ndarray): 2D grid with 0s and 1s.
    i (int): Row index of the starting coordinate.
    j (int): Column index of the starting coordinate.

    Returns:
    list: List of coordinates in the connected component
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    stack = [(i, j)]
    component_coords = []

    while stack:
        x, y = stack.pop()
        if grid[x, y] == 0:  
            grid[x, y] = 1 
            component_coords.append((x, y))
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < lat and 0 <= ny < lon:
                    if grid[nx, ny] == 0:
                        stack.append((nx, ny))

    return component_coords

all_components = []



# Example usage
input_file = 'medi_1960_2021.nc'
output_file = 'medi_1960_2021_processed_focused.nc'
ds = xr.open_dataset(input_file)


# Remove leap days
ds_filtered = remove_leap_days(ds)
# Reverse latitude
ds_filtered = reverse_lat(ds_filtered)



# Find the largest connected component which is the Mediterranean Sea
# It allows to remove part of the Atlantic Ocean and the Black Sea
lat = ds.lat.size
lon = ds.lon.size
grid_cc = np.zeros((lat, lon))
values = ds_filtered['sst'].values
for i in range(lat):
    for j in range(lon):
        if np.isnan(values[0,i,j]):
            grid_cc[i,j] = np.nan
starting_value = 1

for i in range(lat):
    for j in range(lon):
        if grid_cc[i, j] == 0:  
            component_coords = propagate_and_collect(grid_cc, i, j)
            if component_coords:  
                all_components.append(component_coords)

# Find the largest connected component
largest_component = max(all_components, key=len)


# Fill the other connected components with nan
for i in range(lat):
    for j in range(lon):
        if (i,j) not in largest_component:
            ds_filtered['sst'][:,i,j] = np.nan

# Save to a new NetCDF file
ds_filtered.to_netcdf(output_file)

