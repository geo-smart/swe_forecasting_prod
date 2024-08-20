"""
This script performs the following operations:
1. Reads multiple CSV files into Dask DataFrames with specified chunk sizes and compression.
2. Repartitions the DataFrames for optimized processing.
3. Merges the DataFrames based on specified columns.
4. Saves the merged DataFrame to a CSV file in chunks.
5. Reads the merged DataFrame, removes duplicate rows, and saves the cleaned DataFrame to a new CSV file.

Attributes:
    working_dir (str): The directory where the CSV files are located.
    chunk_size (str): The chunk size used for reading and processing the CSV files.

Functions:
    main(): The main function that executes the data processing operations and saves the results.
"""

import dask.dataframe as dd
import os
from snowcast_utils import work_dir, homedir
import pandas as pd
import time

working_dir = work_dir
final_output_name = "final_merged_data_4yrs_snotel_and_ghcnd_stations.csv"
chunk_size = '10MB'  # You can adjust this chunk size based on your hardware and data size
amsr_file = f'{working_dir}/all_training_points_in_westus.csv_amsr_dask_all_training_ponits.csv'
snotel_file = f'{working_dir}/all_snotel_cdec_stations_active_in_westus.csv_swe_restored_dask_all_vars.csv'
ghcnd_file = f'{working_dir}/active_station_only_list.csv_all_vars_masked_non_snow.csv'
all_station_obs_file = f'{working_dir}/snotel_ghcnd_all_obs.csv'
gridmet_file = f'{working_dir}/training_all_point_gridmet_with_snotel_ghcnd.csv'
terrain_file = f'{working_dir}/all_training_points_with_ghcnd_in_westus.csv_terrain_4km_grid_shift.csv'
fsca_file = f'{homedir}/fsca/fsca_final_training_all.csv'
final_final_output_file = f'{work_dir}/{final_output_name}'


def merge_snotel_ghcnd_together():
    snotel_df = pd.read_csv(snotel_file)
    ghcnd_df = pd.read_csv(ghcnd_file)
    print(snotel_df.columns)
    print(ghcnd_df.columns)
    ghcnd_df = ghcnd_df.rename(columns={'STATION': 'station_name',
                                       'DATE': 'date',
                                       'LATITUDE': 'lat',
                                       'LONGITUDE': 'lon',
                                       'SNWD': 'snow_depth',})
    df_combined = pd.concat([snotel_df, ghcnd_df], axis=0, ignore_index=True)
    df_combined.to_csv(all_station_obs_file, index=False)
    print(f"All snotel ang ghcnd are saved to {all_station_obs_file}")
    

def merge_all_data_together():
    
    if os.path.exists(final_final_output_file):
      print(f"The file '{final_final_output_file}' exists. Skipping")
      return final_final_output_file
    
    # merge the snotel and ghcnd together first
    
      
    # Read the CSV files with a smaller chunk size and compression
    amsr = dd.read_csv(amsr_file, blocksize=chunk_size)
    print("amsr.columns = ", amsr.columns)
    ground_truth = dd.read_csv(all_station_obs_file, blocksize=chunk_size)
    print("ground_truth.columns = ", ground_truth.columns)
#     gridmet = dd.read_csv(f'{working_dir}/gridmet_climatology/training_ready_gridmet.csv', blocksize=chunk_size)
    gridmet = dd.read_csv(gridmet_file, blocksize=chunk_size)
    gridmet = gridmet.drop(columns=["Unnamed: 0"])
    print("gridmet.columns = ", gridmet.columns)
    terrain = dd.read_csv(terrain_file, blocksize=chunk_size)
    terrain = terrain.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    terrain = terrain[["lat", "lon", 'Elevation', 'Slope', 'Aspect', 'Curvature', 'Northness', 'Eastness']]
    print("terrain.columns = ", terrain.columns)
    snowcover = dd.read_csv(fsca_file, blocksize=chunk_size)
    snowcover = snowcover.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    print("snowcover.columns = ", snowcover.columns)

    # Repartition DataFrames for optimized processing
    amsr = amsr.repartition(partition_size=chunk_size)
    ground_truth = ground_truth.repartition(partition_size=chunk_size)
    gridmet = gridmet.repartition(partition_size=chunk_size)
    gridmet = gridmet.rename(columns={'day': 'date'})
    terrain = terrain.repartition(partition_size=chunk_size)
    snow_cover = snowcover.repartition(partition_size=chunk_size)
    print("all the dataframes are partitioned")

    # Merge DataFrames based on specified columns
    print("start to merge amsr and ground_truth")
    merged_df = dd.merge(amsr, ground_truth, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_ground_truth.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge gridmet")
    merged_df = dd.merge(merged_df, gridmet, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_gridmet.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge terrain")
    merged_df = dd.merge(merged_df, terrain, on=['lat', 'lon'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_terrain.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge snowcover")
    merged_df = dd.merge(merged_df, snow_cover, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_snow_cover.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    # Save the merged DataFrame to a CSV file in chunks
    output_file = os.path.join(working_dir, final_output_name)
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f'Merge completed. {output_file}')

def cleanup_dataframe():
    """
    Read the merged DataFrame, remove duplicate rows, and save the cleaned DataFrame to a new CSV file
    """
    final_final_output_file = f'{work_dir}/{final_output_name}'
    dtype = {'station_name': 'object'}  # 'object' dtype represents strings
    df = dd.read_csv(final_final_output_file, dtype=dtype)
    df = df.drop_duplicates(keep='first')
    df.to_csv(final_final_output_file, single_file=True, index=False)
    print('Data cleaning completed.')
    return final_final_output_file

  
def sort_training_data(input_training_csv, sorted_training_csv):
    # Read Dask DataFrame from CSV with increased blocksize and assuming missing data
    dtype = {'station_name': 'object'}  # 'object' dtype represents strings
    ddf = dd.read_csv(input_training_csv, assume_missing=True, blocksize='10MB', dtype=dtype)

    # Persist the Dask DataFrame in memory
    ddf = ddf.persist()

    # Sort Dask DataFrame by three columns: date, lat, and Lon
    sorted_ddf = ddf.sort_values(by=['date', 'lat', 'lon'])

    # Save the sorted Dask DataFrame to a new CSV file
    sorted_ddf.to_csv(sorted_training_csv, index=False, single_file=True)
    print(f"sorted training data is saved to {sorted_training_csv}")
  
if __name__ == "__main__":
  
#     merge_snotel_ghcnd_together()
  
#     merge_all_data_together()
#     cleanup_dataframe()
#     final_final_output_file = f'{work_dir}/{final_output_name}'
#     sort_training_data(final_final_output_file, f'{work_dir}/{final_output_name}_sorted.csv')
    
    start_time = time.time()
    
    merge_all_data_together()
    print(f"Time taken for merge_all_data_together: {time.time() - start_time} seconds")
    
    start_time = time.time()
    cleanup_dataframe()
    print(f"Time taken for cleanup_dataframe: {time.time() - start_time} seconds")
    
    final_final_output_file = f'{work_dir}/{final_output_name}'
    sorted_output_file = f'{work_dir}/{final_output_name}_sorted.csv'
    
    start_time = time.time()
    sort_training_data(final_final_output_file, sorted_output_file)
    print(f"Time taken for sort_training_data: {time.time() - start_time} seconds")
    
