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
from datetime import datetime, timedelta, date
import warnings
import sys
from convert_results_to_images import plot_all_variables_in_one_csv

warnings.filterwarnings("ignore", category=RuntimeWarning)

western_us_coords = f'{work_dir}/dem_file.tif.csv'

latlontree = None

def find_closest_index_numpy(target_latitude, target_longitude, lat_grid, lon_grid):
    # Calculate the squared Euclidean distance between the target point and all grid points
    distance_squared = (lat_grid - target_latitude)**2 + (lon_grid - target_longitude)**2
    
    # Find the indices of the minimum distance
    lat_idx, lon_idx = np.unravel_index(np.argmin(distance_squared), distance_squared.shape)
    
    return lat_idx, lon_idx, lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

def is_binary(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read a chunk of bytes from the file
            chunk = file.read(1024)

            # Check for null bytes, a common indicator of binary data
            if b'\x00' in chunk:
                return True

            # Check for a high percentage of non-printable ASCII characters
            text_characters = "".join(chr(byte) for byte in chunk if 32 <= byte <= 126)
            if not text_characters:
                return True

            # If none of the binary indicators are found, assume it's a text file
            return False

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
  
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
    parent_directory = os.path.dirname(target_amsr_hdf_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print(f"Parent directory '{parent_directory}' created successfully.")
    
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

def download_amsr_and_convert_grid(target_date = test_start_date):
    """
    Download AMSR snow data, convert it to DEM format, and save as a CSV file.
    """
    
    
    
    # the mapper
    target_mapper_csv_path = f'{work_dir}/amsr_to_gridmet_mapper.csv'
    mapper_df = pd.read_csv(target_mapper_csv_path)
    #print(mapper_df.head())
    
    df = pd.DataFrame(columns=['date', 'lat', 
                               'lon', 'AMSR_SWE', 
                               'AMSR_Flag'])
    date = target_date
    date = date.replace("-", ".")
    he5_date = date.replace(".", "")
    
    # Check if the CSV already exists
    target_csv_path = f'{work_dir}/testing_ready_amsr_{date}.csv'
    if os.path.exists(target_csv_path):
        print(f"File {target_csv_path} already exists, skipping..")
        return target_csv_path
    
    target_amsr_hdf_path = f"{work_dir}/amsr_testing/testing_amsr_{date}.he5"
    if os.path.exists(target_amsr_hdf_path) and is_binary(target_amsr_hdf_path):
        print(f"File {target_amsr_hdf_path} already exists, skip downloading..")
    else:
        cmd = f"curl --output {target_amsr_hdf_path} -b ~/.urs_cookies -c ~/.urs_cookies -L -n -O https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/{date}/AMSR_U2_L3_DailySnow_B02_{he5_date}.he5"
        print(f'Running command: {cmd}')
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # Check the exit code
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}.")
            if os.path.exists(target_amsr_hdf_path):
              os.remove(target_amsr_hdf_path)
              print(f"Wrong {target_amsr_hdf_path} removed successfully.")
            raise Exception(f"Failed to download {target_amsr_hdf_path} - {result.stderr}")
    
    # Read the HDF
    print(f"Reading {target_amsr_hdf_path}")
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
    return target_csv_path

def add_cumulative_column(df, column_name):
    df[f'cumulative_{column_name}'] = df[column_name].sum()
    return df

# Function to perform polynomial interpolation and fill in missing values
def interpolate_missing_and_add_cumulative_inplace(row, column_name, degree=1):
  """
  Interpolate missing values in a Pandas Series using polynomial interpolation
  and add a cumulative column.

  Parameters:
    - row (pd.Series): The input row containing the data to be interpolated.
    - column_name (str): The name of the column to be interpolated.
    - degree (int, optional): The degree of the polynomial fit. Default is 1 (linear).

  Returns:
    - pd.Series: The row with interpolated values and a cumulative column.

  Raises:
    - ValueError: If there are unexpected null values after interpolation.

  Note:
    - For 'SWE' column, values above 240 are treated as gaps and set to 240.
    - For 'fsca' column, values above 100 are treated as gaps and set to 100.

  Examples:
    ```python
    # Example usage:
    interpolated_row = interpolate_missing_and_add_cumulative_inplace(my_row, 'fsca', degree=2)
    ```

  """
  
  # Extract X series (column names)
  x_all_key = row.index
  
  x_subset_key = x_all_key[x_all_key.str.startswith(column_name)]
  #print("x_subset_key = ", x_subset_key)
#   x = np.arange(len(x_subset_key))

#   # Extract Y series (values from the first row)
#   y = row[x_subset_key]
# #   print("start row: ", y)
  
#   # Create a mask for missing values
#   if column_name == "AMSR_SWE":
#     mask = (y > 240) | y.isnull()
#   elif column_name == "fsca":
#     mask = (y > 100) | y.isnull() 
#   else:
#     mask = y.isnull()

#   # Check if all elements in the mask array are True
#   all_true = np.all(mask)

#   if all_true or len(np.where(~mask)[0]) == 1:
#     row[x_subset_key] = 0
# #     print("Final all columns: ", row)
#   else:
#     # Perform interpolation
#     #new_y = np.interp(x, x[~mask], y[~mask])
#     # Replace missing values with interpolated values
#     #df[column_name] = new_y
    
#     try:
#       # Coefficients of the polynomial fit
#       #coefficients = np.polyfit(x[~mask], y[~mask], deg=degree)

#       # Perform polynomial interpolation
#       #interpolated_values = np.polyval(coefficients, x)

#       # Merge using np.where
#       #merged_array = np.where(mask, interpolated_values, y)

#       #row.loc[x_subset_key] = merged_array
# #       print("after assign: ", row)
#       #print("don't interpolate and check the original data")
#       pass
#     except Exception as e:
#       # Print the error message and traceback
#       import traceback
#       traceback.print_exc()
#       print("x:", x)
#       print("y:", y)
#       print("mask:", mask)
#       print(f"Error: {e}")
#       raise e
      
#     if column_name == "AMSR_SWE":
#       row[x_subset_key] = row[x_subset_key].clip(upper=240, lower=0)
#     elif column_name == "fsca":
#       row[x_subset_key] = row[x_subset_key].clip(upper=100, lower=0)
#     else:
#       row[x_subset_key] = row[x_subset_key].clip(upper=240, lower=0)
      
#     print("after clip: ", row)
      
#     if row[x_subset_key].isnull().any():
#       print("x:", x)
#       print("y:", y)
#       print("mask:", mask)
#       print("why row still has values > 100", row)
#       raise ValueError("Single group: shouldn't have null values here")

  are_all_values_between_0_and_240 = row[x_subset_key].between(1, 239).all()
  if are_all_values_between_0_and_240:
    print("row[x_subset_key] = ", row[x_subset_key])
    print("row[x_subset_key].sum() = ", row[x_subset_key].sum())
  # create the cumulative column after interpolation
  row[f"cumulative_{column_name}"] = row[x_subset_key].sum()
  return row
    
    
def get_cumulative_amsr_data(target_date = test_start_date, force=False):
    
    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    print(selected_date)
    if selected_date.month < 10:
      past_october_1 = datetime(selected_date.year - 1, 10, 1)
    else:
      past_october_1 = datetime(selected_date.year, 10, 1)

    # Traverse and print every day from past October 1 to the specific date
    current_date = past_october_1
    target_csv_path = f'{work_dir}/testing_ready_amsr_{target_date}_cumulative.csv'

    columns_to_be_cumulated = ["AMSR_SWE"]
    
    gap_filled_csv = f"{target_csv_path}_gap_filled.csv"
    if os.path.exists(gap_filled_csv) and not force:
      print(f"{gap_filled_csv} already exists, skipping..")
      df = pd.read_csv(gap_filled_csv)
      print(df["AMSR_SWE"].describe())
    else:
      date_keyed_objects = {}
      data_dict = {}
      new_df = None
      while current_date <= selected_date:
        print(current_date.strftime('%Y-%m-%d'))
        current_date_str = current_date.strftime('%Y-%m-%d')

        data_dict[current_date_str] = download_amsr_and_convert_grid(current_date_str)
        current_df = pd.read_csv(data_dict[current_date_str])
        current_df.drop(columns=["date"], inplace=True)

        if current_date != selected_date:
          current_df.rename(columns={
            "AMSR_SWE": f"AMSR_SWE_{current_date_str}",
            "AMSR_Flag": f"AMSR_Flag_{current_date_str}",
          }, inplace=True)
        #print(current_df.head())

        if new_df is None:
          new_df = current_df
        else:
          new_df = pd.merge(new_df, current_df, on=['gridmet_lat', 'gridmet_lon'])
          #new_df = new_df.append(current_df, ignore_index=True)

        current_date += timedelta(days=1)

      print("new_df.columns = ", new_df.columns)
      print("new_df.head = ", new_df.head())
      df = new_df

      #df.sort_values(by=['gridmet_lat', 'gridmet_lon', 'date'], inplace=True)
      print("All current head: ", df.head())
      print("the new_df.shape: ", df.shape)

      print("Start to fill in the missing values")
      #grouped = df.groupby(['gridmet_lat', 'gridmet_lon'])
      filled_data = pd.DataFrame()

      # Apply the function to each group
      for column_name in columns_to_be_cumulated:
        start_time = time.time()
        #filled_data = df.apply(lambda row: interpolate_missing_and_add_cumulative_inplace(row, column_name), axis=1)
        #alike_columns = filled_data.filter(like=column_name)
        #filled_data[f'cumulative_{column_name}'] = alike_columns.sum(axis=1)
        print("filled_data.columns = ", filled_data.columns)
        filtered_columns = df.filter(like=column_name)
        print(filtered_columns.columns)
        filtered_columns = filtered_columns.mask(filtered_columns > 240)
        filtered_columns.interpolate(axis=1, method='linear', inplace=True)
        filtered_columns.fillna(0, inplace=True)
        
        sum_column = filtered_columns.sum(axis=1)
        # Define a specific name for the new column
        df[f'cumulative_{column_name}'] = sum_column
        df[filtered_columns.columns] = filtered_columns
        
        if filtered_columns.isnull().any().any():
          print("filtered_columns :", filtered_columns)
          raise ValueError("Single group: shouldn't have null values here")
        
        
        

        # Concatenate the original DataFrame with the Series containing the sum
        #df = pd.concat([df, sum_column.rename(new_column_name)], axis=1)
#         cumulative_column = filled_data.filter(like=column_name).sum(axis=1)
#         filled_data[f'cumulative_{column_name}'] = cumulative_column
        #filled_data = pd.concat([filled_data, cumulative_column], axis=1)
        print("filled_data.columns: ", filled_data.columns)
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"calculate column {column_name} elapsed time: {elapsed_time} seconds")

#       if any(filled_data['AMSR_SWE'] > 240):
#         raise ValueError("Error: shouldn't have AMSR_SWE > 240 at this point")
      filled_data = df
      filled_data["date"] = target_date
      print("Finished correctly ", filled_data.head())
      filled_data.to_csv(gap_filled_csv, index=False)
      print(f"New filled values csv is saved to {gap_filled_csv}")
      df = filled_data
    
    result = df
    print("result.head = ", result.head())
    # fill in the rest NA as 0
    if result.isnull().any().any():
      print("result :", result)
      raise ValueError("Single group: shouldn't have null values here")
    
    # only retain the rows of the target date
    print(result['date'].unique())
    print(result.shape)
    print(result[["AMSR_SWE", "AMSR_Flag"]].describe())
    result.to_csv(target_csv_path, index=False)
    print(f"New data is saved to {target_csv_path}")
    
      
    
if __name__ == "__main__":
    # Run the download and conversion function
    #prepare_amsr_grid_mapper()
    prepare_amsr_grid_mapper()
#     download_amsr_and_convert_grid()
    
    get_cumulative_amsr_data(force=False)
    input_time_series_file = f'{work_dir}/testing_ready_amsr_{test_start_date}_cumulative.csv_gap_filled.csv'

    #plot_all_variables_in_one_csv(input_time_series_file, f"{input_time_series_file}.png")

