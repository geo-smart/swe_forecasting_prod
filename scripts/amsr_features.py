import os
import csv
import h5py
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import dask
import dask.dataframe as dd
import dask.delayed as delayed
import dask.bag as db
import xarray as xr
from snowcast_utils import work_dir, train_start_date, train_end_date
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message="overflow encountered in add")


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


def create_snotel_ghcnd_station_to_amsr_mapper(
  new_base_station_list_file, 
  target_csv_path
):
    station_data = pd.read_csv(new_base_station_list_file)
    
    
    date = "2022-10-01"
    date = date.replace("-", ".")
    he5_date = date.replace(".", "")
    
    # Check if the CSV already exists
    
    if os.path.exists(target_csv_path):
        print(f"File {target_csv_path} already exists, skipping..")
        df = pd.read_csv(target_csv_path)
        return df
    
    target_amsr_hdf_path = f"{work_dir}/amsr_testing/testing_amsr_{date}.he5"
    if os.path.exists(target_amsr_hdf_path):
        print(f"File {target_amsr_hdf_path} already exists, skip downloading..")
    else:
        cmd = f"curl --output {target_amsr_hdf_path} -b ~/.urs_cookies -c ~/.urs_cookies -L -n -O https://n5eil01u.ecs.nsidc.org/AMSA/AU_DySno.001/{date}/AMSR_U2_L3_DailySnow_B02_{he5_date}.he5"
        print(f'Running command: {cmd}')
        subprocess.run(cmd, shell=True)
    
    df = pd.DataFrame(columns=['amsr_lat', 'amsr_lon', 
                               'amsr_lat_idx', 'amsr_lon_idx',
                               'station_lat', 'station_lon'])
    # Read the HDF
    file = h5py.File(target_amsr_hdf_path, 'r')
    hem_group = file['HDFEOS/GRIDS/Northern Hemisphere']
    lat = hem_group['lat'][:]
    lon = hem_group['lon'][:]
    
    # Replace NaN values with 0
    lat = np.nan_to_num(lat, nan=0.0)
    lon = np.nan_to_num(lon, nan=0.0)
    
    # Convert the AMSR grid into our gridMET 1km grid
    for idx, row in station_data.iterrows():
        target_lat = row['latitude']
        target_lon = row['longitude']
        
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
    return df
  
  
def extract_amsr_values_save_to_csv(amsr_data_dir, output_csv_file, new_base_station_list_file, start_date, end_date):
    if os.path.exists(output_csv_file):
        os.remove(output_csv_file)
    
    target_csv_path = f'{work_dir}/training_snotel_ghcnd_station_to_amsr_mapper_all_training_points.csv'
    mapper_df = create_snotel_ghcnd_station_to_amsr_mapper(new_base_station_list_file, 
                                         target_csv_path)
        
    # station_data = pd.read_csv(new_base_station_list_file)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Create a Dask DataFrame
    dask_station_data = dd.from_pandas(mapper_df, npartitions=1)

    # Function to process each file
    def process_file(filename):
        file_path = os.path.join(amsr_data_dir, filename)
        print(file_path)
        
        file = h5py.File(file_path, 'r')
        hem_group = file['HDFEOS/GRIDS/Northern Hemisphere']

        date_str = filename.split('_')[-1].split('.')[0]
        date = datetime.strptime(date_str, '%Y%m%d')

        if not (start_date <= date <= end_date):
            print(f"{date} is not in the training period, skipping..")
            return None

        new_date_str = date.strftime("%Y-%m-%d")
        swe = hem_group['Data Fields/SWE_NorthernDaily'][:]
        flag = hem_group['Data Fields/Flags_NorthernDaily'][:]
        # Create an empty Pandas DataFrame with the desired columns
        result_df = pd.DataFrame(columns=['date', 'lat', 'lon', 'AMSR_SWE'])

        # Sample loop to add rows to the Pandas DataFrame using dask.delayed
        @delayed
        def process_row(row, swe, new_date_str):
          closest_lat_idx = int(row['amsr_lat_idx'])
          closest_lon_idx = int(row['amsr_lon_idx'])
          closest_swe = swe[closest_lat_idx, closest_lon_idx]
          
          return pd.DataFrame([[
            new_date_str, 
            row['station_lat'],
            row['station_lon'],
            closest_swe]], 
            columns=result_df.columns
          )


        # List of delayed computations
        delayed_results = [process_row(row, swe, new_date_str) for _, row in mapper_df.iterrows()]

        # Compute the delayed results and concatenate them into a Pandas DataFrame
        result_df = dask.compute(*delayed_results)
        result_df = pd.concat(result_df, ignore_index=True)

        # Print the final Pandas DataFrame
        #print(result_df)
          
        return result_df

    # Get the list of files
    files = [f for f in os.listdir(amsr_data_dir) if f.endswith('.he5')]

    # Create a Dask Bag from the files
    dask_bag = db.from_sequence(files, npartitions=2)

    # Process files in parallel
    processed_data = dask_bag.map(process_file).filter(lambda x: x is not None).compute()

    # Concatenate the processed data
    combined_df = pd.concat(processed_data, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_csv_file, index=False)

    print(f"Merged data saved to {output_csv_file}")

                    
if __name__ == "__main__":
    amsr_data_dir = '/home/chetana/gridmet_test_run/amsr'
    # new_base_station_list_file = f"{work_dir}/all_snotel_cdec_stations_active_in_westus.csv"
    all_training_points_with_snotel_ghcnd_file = f"{work_dir}/all_training_points_snotel_ghcnd_in_westus.csv"
    new_base_df = pd.read_csv(all_training_points_with_snotel_ghcnd_file)
    print(new_base_df.head())
    output_csv_file = f"{all_training_points_with_snotel_ghcnd_file}_amsr_dask_all_training_ponits_with_ghcnd.csv"
    
    start_date = train_start_date
    end_date = train_end_date

    extract_amsr_values_save_to_csv(amsr_data_dir, output_csv_file, all_training_points_with_snotel_ghcnd_file, start_date, end_date)

