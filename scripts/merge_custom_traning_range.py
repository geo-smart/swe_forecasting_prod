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
from snowcast_utils import work_dir

working_dir = work_dir
chunk_size = '32MB'  # You can adjust this chunk size based on your hardware and data size

def main():
    # Read the CSV files with a smaller chunk size and compression
    amsr = dd.read_csv(f'{working_dir}/training_ready_amsr_3_yrs.csv', blocksize=chunk_size)
    snotel = dd.read_csv(f'{working_dir}/training_data_ready_snotel_3_yrs.csv', blocksize=chunk_size)
    gridmet = dd.read_csv(f'{working_dir}/gridmet_climatology/training_ready_gridmet.csv', blocksize=chunk_size)
    terrain = dd.read_csv(f'{working_dir}/training_ready_terrain.csv', blocksize=chunk_size)

    # Repartition DataFrames for optimized processing
    amsr = amsr.repartition(partition_size=chunk_size)
    snotel = snotel.repartition(partition_size=chunk_size)
    gridmet = gridmet.repartition(partition_size=chunk_size)
    gridmet = gridmet.rename(columns={'day': 'date'})
    terrain = terrain.repartition(partition_size=chunk_size)

    # Merge DataFrames based on specified columns
    merged_df = dd.merge(amsr, snotel, on=['lat', 'lon', 'date'], how='outer')
    merged_df = dd.merge(merged_df, gridmet, on=['lat', 'lon', 'date'], how='outer')
    merged_df = dd.merge(merged_df, terrain, on=['lat', 'lon'], how='outer')

    # Save the merged DataFrame to a CSV file in chunks
    output_file = os.path.join(working_dir, 'final_merged_data_3_yrs.csv')
    merged_df.to_csv(output_file, single_file=True, index=False)
    print('Merge completed.')

    # Read the merged DataFrame, remove duplicate rows, and save the cleaned DataFrame to a new CSV file
    df = dd.read_csv(f'{work_dir}/final_merged_data_3_yrs.csv')
    df = df.drop_duplicates(keep='first')
    df.to_csv(f'{work_dir}/final_merged_data_3yrs_cleaned_v2.csv', single_file=True, index=False)
    print('Data cleaning completed.')

if __name__ == "__main__":
    main()

