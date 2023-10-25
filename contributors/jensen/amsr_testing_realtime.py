"""
Script for downloading AMSR snow data, converting it to DEM format, and saving as a CSV file.

This script downloads AMSR snow data, converts it to a format compatible with DEM, and saves it as a CSV file.
It utilizes the h5py library to read HDF5 files, pandas for data manipulation, and scipy.spatial.KDTree
for finding the nearest grid points. The script also checks if the target CSV file already exists to avoid redundant
downloads and processing.

Usage:
    Run this script to download and convert AMSR snow data for a specific date. It depends on the test_start_date from snowcast_utils to specify which date to download. You can overwrite that.

"""

import os
import h5py
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from snowcast_utils import work_dir, test_start_date
from scipy.spatial import KDTree
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

western_us_coords = '/home/chetana/gridmet_test_run/dem_file.tif.csv'

latlontree = None

def find_closest_index_numpy(target_latitude, target_longitude, lat_grid, lon_grid):
    # Calculate the squared Euclidean distance between the target point and all grid points
    distance_squared = (lat_grid - target_latitude)**2 + (lon_grid - target_longitude)**2
    
    # Find the indices of the minimum distance
    lat_idx, lon_idx = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
    
    return lat_idx, lon_idx, lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

def find_closest_index_tree(target_latitude, target_longitude, lat_grid, lon_grid):
    """
    Find the closest grid point indices for a target latitude and longitude using KDTree.

    Parameters:
        target_latitude (float): Target latitude.
        target_longitude (float): Target longitude.
        lat_grid (numpy.ndarray): Array of latitude values.
        lon_grid (numpy.ndarray): Array of longitude values.

    Returns:
        int: Latitude index.
        int: Longitude index.
        float: Closest latitude value.
        float: Closest longitude value.
    """
    global latlontree
    
    if latlontree is None:
        # Create a KD-Tree from lat_grid and lon_grid
        lat_grid_cleaned = np.nan_to_num(lat_grid, nan=0.0)  # Replace NaN with 0
        lon_grid_cleaned = np.nan_to_num(lon_grid, nan=0.0)  # Replace NaN with 0
        latlontree = KDTree(list(zip(lat_grid_cleaned.ravel(), lon_grid_cleaned.ravel())))
      
    # Query the KD-Tree to find the nearest point
    distance, index = latlontree.query([target_latitude, target_longitude])

    # Convert the 1D index to 2D grid indices
    lat_idx, lon_idx = np.unravel_index(index, lat_grid.shape)

    return lat_idx, lon_idx, lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

def find_closest_index(target_latitude, target_longitude, lat_grid, lon_grid):
    """
    Find the closest grid point indices for a target latitude and longitude.

    Parameters:
        target_latitude (float): Target latitude.
        target_longitude (float): Target longitude.
        lat_grid (numpy.ndarray): Array of latitude values.
        lon_grid (numpy.ndarray): Array of longitude values.

    Returns:
        int: Latitude index.
        int: Longitude index.
        float: Closest latitude value.
        float: Closest longitude value.
    """
    lat_diff = np.float64(np.abs(lat_grid - target_latitude))
    lon_diff = np.float64(np.abs(lon_grid - target_longitude))

    # Find the indices corresponding to the minimum differences
    lat_idx, lon_idx = np.unravel_index(np.argmin(lat_diff + lon_diff), lat_grid.shape)

    return lat_idx, lon_idx, lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

  
def prepare_amsr_grid_mapper():
    df = pd.DataFrame(columns=['amsr_lat', 'amsr_lon', 
                               'amsr_lat_idx', 'amsr_lon_idx',
                               'gridmet_lat', 'gridmet_lon'])
    date = test_start_date
    date = date.replace("-", ".")
    he5_date = date.replace(".", "")
    
    # Check if the CSV already exists
    target_csv_path = f'{work_dir}/amsr_to_gridmet_mapper.csv'
    if os.path.exists(target_csv_path):
        print(f"File {target_csv_path} already exists, skipping..")
        return
    
    target_amsr_hdf_path = f"{work_dir}/amsr_testing/testing_amsr_{date}.he5"
    if os.path.exists(target_amsr_hdf_path):
        print(f"File {target_amsr_hdf_path} already exists, skip downloading..")
    else:
        cmd = f"curl --output {target_amsr_hdf_path} -b ~/.urs_cookies -c ~/.urs_cookies -L -n -O https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/{date}/AMSR_U2_L3_DailySnow_B02_{he5_date}.he5"
        print(f'Running command: {cmd}')
        subprocess.run(cmd, shell=True)
    
    # Read the HDF
    file = h5py.File(target_amsr_hdf_path, 'r')
    hem_group = file['HDFEOS/GRIDS/Northern Hemisphere']
    lat = hem_group['lat'][:]
    lon = hem_group['lon'][:]
    
    # Replace NaN values with 0
    lat = np.nan_to_num(lat, nan=0.0)
    lon = np.nan_to_num(lon, nan=0.0)
    
    # Convert the AMSR grid into our gridMET 1km grid
    western_us_df = pd.read_csv(western_us_coords)
    for idx, row in western_us_df.iterrows():
        target_lat = row['Latitude']
        target_lon = row['Longitude']
        
        # compare the performance and find the fastest way to search nearest point
        closest_lat_idx, closest_lon_idx, closest_lat, closest_lon = find_closest_index(target_lat, target_lon, lat, lon)
        df.loc[len(df.index)] = [closest_lat, 
                                 closest_lon,
                                 closest_lat_idx,
                                 closest_lon_idx,
                                 target_lat, 
                                 target_lon]
    
    # Save the new converted AMSR to CSV file
    df.to_csv(target_csv_path, index=False)
  
    print('AMSR mapper csv is created.')

def download_amsr_and_convert_grid():
    """
    Download AMSR snow data, convert it to DEM format, and save as a CSV file.
    """
    
    prepare_amsr_grid_mapper()
    
    # the mapper
    target_mapper_csv_path = f'{work_dir}/amsr_to_gridmet_mapper.csv'
    mapper_df = pd.read_csv(target_mapper_csv_path)
    print(mapper_df.head())
    
    df = pd.DataFrame(columns=['date', 'lat', 
                               'lon', 'AMSR_SWE', 
                               'AMSR_Flag'])
    date = test_start_date
    date = date.replace("-", ".")
    he5_date = date.replace(".", "")
    
    # Check if the CSV already exists
    target_csv_path = f'{work_dir}/testing_ready_amsr_{date}.csv'
    if os.path.exists(target_csv_path):
        print(f"File {target_csv_path} already exists, skipping..")
        return
    
    target_amsr_hdf_path = f"{work_dir}/amsr_testing/testing_amsr_{date}.he5"
    if os.path.exists(target_amsr_hdf_path):
        print(f"File {target_amsr_hdf_path} already exists, skip downloading..")
    else:
        cmd = f"curl --output {target_amsr_hdf_path} -b ~/.urs_cookies -c ~/.urs_cookies -L -n -O https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/{date}/AMSR_U2_L3_DailySnow_B02_{he5_date}.he5"
        print(f'Running command: {cmd}')
        subprocess.run(cmd, shell=True)
    
    # Read the HDF
    file = h5py.File(target_amsr_hdf_path, 'r')
    hem_group = file['HDFEOS/GRIDS/Northern Hemisphere']
    lat = hem_group['lat'][:]
    lon = hem_group['lon'][:]
    
    # Replace NaN values with 0
    lat = np.nan_to_num(lat, nan=0.0)
    lon = np.nan_to_num(lon, nan=0.0)
    
    swe = hem_group['Data Fields/SWE_NorthernDaily'][:]
    flag = hem_group['Data Fields/Flags_NorthernDaily'][:]
    date = datetime.strptime(date, '%Y.%m.%d')
    
    # Convert the AMSR grid into our DEM 1km grid
    
    def get_swe(row):
        # Perform your custom calculation here
        closest_lat_idx = int(row['amsr_lat_idx'])
        closest_lon_idx = int(row['amsr_lon_idx'])
        closest_swe = swe[closest_lat_idx, closest_lon_idx]
        return closest_swe
    
    def get_swe_flag(row):
        # Perform your custom calculation here
        closest_lat_idx = int(row['amsr_lat_idx'])
        closest_lon_idx = int(row['amsr_lon_idx'])
        closest_flag = flag[closest_lat_idx, closest_lon_idx]
        return closest_flag
    
    # Use the apply function to apply the custom function to each row
    mapper_df['AMSR_SWE'] = mapper_df.apply(get_swe, axis=1)
    mapper_df['AMSR_Flag'] = mapper_df.apply(get_swe_flag, axis=1)
    mapper_df['date'] = date
    mapper_df.rename(columns={'dem_lat': 'lat'}, inplace=True)
    mapper_df.rename(columns={'dem_lon': 'lon'}, inplace=True)
    mapper_df = mapper_df.drop(columns=['amsr_lat',
                                        'amsr_lon',
                                        'amsr_lat_idx',
                                        'amsr_lon_idx'])
    
    print("result df: ", mapper_df.head())
    # Save the new converted AMSR to CSV file
    print(f"saving the new AMSR SWE to csv: {target_csv_path}")
    mapper_df.to_csv(target_csv_path, index=False)
    
    print('Completed AMSR testing data collection.')

    

# Run the download and conversion function
#prepare_amsr_grid_mapper()
download_amsr_and_convert_grid()

