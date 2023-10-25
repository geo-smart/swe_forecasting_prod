import os
import csv
import h5py
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr

def copy_he5_files(source_dir, destination_dir):
    '''
    Copy .he5 files from the source directory to the destination directory.

    Args:
        source_dir (str): The source directory containing .he5 files to copy.
        destination_dir (str): The destination directory where .he5 files will be copied.

    Returns:
        None
    '''
    # Get a list of all subdirectories and files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.he5'):
                # Get the absolute path of the source file
                source_file_path = os.path.join(root, file)
                # Copy the file to the destination directory
                shutil.copy(source_file_path, destination_dir)

def find_closest_index(target_latitude, target_longitude, lat_grid, lon_grid):
    '''
    Find the index of the grid cell with the closest coordinates to the target latitude and longitude.

    Args:
        target_latitude (float): The target latitude.
        target_longitude (float): The target longitude.
        lat_grid (numpy.ndarray): An array of latitude values.
        lon_grid (numpy.ndarray): An array of longitude values.

    Returns:
        Tuple[int, int, float, float]: A tuple containing the row index, column index, closest latitude, and closest longitude.
    '''
    # Compute the absolute differences between target and grid coordinates
    lat_diff = np.abs(lat_grid - target_latitude)
    lon_diff = np.abs(lon_grid - target_longitude)

    # Find the indices corresponding to the minimum differences
    lat_idx, lon_idx = np.unravel_index(np.argmin(lat_diff + lon_diff), lat_grid.shape)

    return lat_idx, lon_idx, lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

def extract_amsr_values_save_to_csv(amsr_data_dir, output_csv_file):
    '''
    Extract AMSR data values and save them to a CSV file.

    Args:
        amsr_data_dir (str): The directory containing AMSR .he5 files.
        output_csv_file (str): The path to the output CSV file.

    Returns:
        None
    '''
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    
    # Open the output CSV file in append mode using the csv module
    with open(output_csv_file, 'a', newline='') as csvfile:
        # Create a csv writer object
        writer = csv.writer(csvfile)

        # If the file is empty, write the header row
        if os.path.getsize(output_csv_file) == 0:
            writer.writerow(["date", "lat", "lon", "amsr_swe"])

        # Loop through all the .he5 files in the directory
        for filename in os.listdir(amsr_data_dir):
            if filename.endswith('.he5'):
                file_path = os.path.join(amsr_data_dir, filename)
                print(file_path)

                data_field_ds = xr.open_dataset(file_path, group='/HDFEOS/GRIDS/Northern Hemisphere/Data Fields')
                swe_df = data_field_ds["SWE_NorthernDaily"].values

                latlon_ds = xr.open_dataset(file_path, group='/HDFEOS/GRIDS/Northern Hemisphere')
                lat_df = latlon_ds["lat"].values
                lon_df = latlon_ds["lon"].values

                swe_variable = data_field_ds['SWE_NorthernDaily'].assign_coords(
                    lat=latlon_ds['lat'],
                    lon=latlon_ds['lon']
                )

                date_str = filename.split('_')[-1].split('.')[0]
                date = datetime.strptime(date_str, '%Y%m%d')
                df_val = swe_variable.to_dataframe()
                
                swe_variable = swe_variable.to_dataframe().reset_index()
                for idx, row in station_data.iterrows():
                    desired_lat = row['lat']
                    desired_lon = row['lon']
                    swe_variable['distance'] = np.sqrt((swe_variable['lat'] - desired_lat)**2 + (swe_variable['lon'] - desired_lon)**2)
                    closest_row = swe_variable.loc[swe_variable['distance'].idxmin()]
                    writer.writerow([date, desired_lat, desired_lon, closest_row['SWE_NorthernDaily']])

amsr_data_dir = '/home/chetana/gridmet_test_run/amsr'
output_csv_file = '/home/chetana/gridmet_test_run/training_amsr_data_tmp.csv'

station_cell_mapping = pd.read_csv('/home/chetana/gridmet_test_run/training_test_cords.csv')

extract_amsr_values_save_to_csv(amsr_data_dir, output_csv_file)

