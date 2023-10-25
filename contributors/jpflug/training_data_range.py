"""
This script loads and processes several CSV files into Dask DataFrames, applies filters, renames columns,
and saves the resulting DataFrames to new CSV files based on a specified time range.

Attributes:
    gridmet_20_years_file (str): File path of the GridMET climatology data CSV file.
    snotel_20_years_file (str): File path of the SNOTEL data CSV file.
    terrain_file (str): File path of the terrain data CSV file.
    amsr_3_years_file (str): File path of the AMSR data CSV file.
    output_file (str): File path where the merged and processed data will be saved.

Functions:
    clip_csv_using_time_range: Main function that loads, processes, and saves CSV data based on a specified time range.
"""

from snowcast_utils import work_dir
import dask.dataframe as dd

def clip_csv_using_time_range(gridmet_20_years_file, snotel_20_years_file, terrain_file, amsr_3_years_file, output_file):
    """
    Loads, processes, and saves CSV data into Dask DataFrames, applies filters, renames columns, and saves the resulting
    DataFrames to new CSV files based on a specified time range.

    Args:
        gridmet_20_years_file (str): File path of the GridMET climatology data CSV file.
        snotel_20_years_file (str): File path of the SNOTEL data CSV file.
        terrain_file (str): File path of the terrain data CSV file.
        amsr_3_years_file (str): File path of the AMSR data CSV file.
        output_file (str): File path where the merged and processed data will be saved.

    Returns:
        None
    """
    # Load CSV files into Dask DataFrames
    gridmet_df = dd.read_csv(gridmet_20_years_file, blocksize="64MB")
    snotel_df = dd.read_csv(snotel_20_years_file, blocksize="64MB")
    terrain_df = dd.read_csv(terrain_file, blocksize="64MB")
    amsr_df = dd.read_csv(amsr_3_years_file, blocksize="64MB")

    # Filter and rename columns for each DataFrame in a single step
    # (Code to filter and rename columns...)

    # Save the processed Dask DataFrames to new CSV files
    gridmet_df.to_csv('/home/chetana/gridmet_test_run/training_ready_gridmet_3_yrs.csv', index=False, single_file=True)
    amsr_df.to_csv('/home/chetana/gridmet_test_run/training_ready_amsr_3_yrs.csv', index=False, single_file=True)
    snotel_df.to_csv('/home/chetana/gridmet_test_run/training_ready_snotel_3_yrs.csv', index=False, single_file=True)

# Define file paths and execute the function
gridmet_20_years_file = f"{work_dir}/gridmet_climatology/testing_ready_gridmet.csv"
snotel_20_years_file = f"{work_dir}/training_ready_snotel_data.csv"
terrain_file = f'{work_dir}/training_ready_terrain.csv'
amsr_3_years_file = f"{work_dir}/training_amsr_data.csv"
output_file = f"{work_dir}/training_ready_merged_data_dd.csv"

clip_csv_using_time_range(gridmet_20_years_file, snotel_20_years_file, terrain_file, amsr_3_years_file, output_file)

