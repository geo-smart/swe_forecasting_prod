import os
import pandas as pd
from datetime import datetime
from snowcast_utils import work_dir, test_start_date

def merge_all_gridmet_amsr_csv_into_one(gridmet_csv_folder, dem_all_csv, testing_all_csv):
    """
    Merge all GridMET and AMSR CSV files into one combined CSV file.

    Args:
        gridmet_csv_folder (str): The folder containing GridMET CSV files.
        dem_all_csv (str): Path to the DEM (Digital Elevation Model) CSV file.
        testing_all_csv (str): Path to save the merged CSV file.

    Returns:
        None
    """
    # List of file paths for the CSV files
    csv_files = []
    selected_date = datetime.strptime(test_start_date, "%Y-%m-%d")
    for file in os.listdir(gridmet_csv_folder):
        if file.endswith('.csv') and test_start_date in file:
            csv_files.append(os.path.join(gridmet_csv_folder, file))

    # Initialize an empty list to store all dataframes
    dfs = []

    # Read each CSV file into separate dataframes
    for file in csv_files:
        df = pd.read_csv(file, encoding='utf-8', index_col=False)
        dfs.append(df)

    dem_df = pd.read_csv(f"{work_dir}/dem_all.csv", encoding='utf-8', index_col=False)
    dfs.append(dem_df)

    date = test_start_date
    date = date.replace("-", ".")
    amsr_df = pd.read_csv(f'{work_dir}/testing_ready_amsr_{date}.csv', index_col=False)
    amsr_df.rename(columns={'gridmet_lat': 'Latitude', 'gridmet_lon': 'Longitude'}, inplace=True)
    dfs.append(amsr_df)

    # Merge the dataframes based on the latitude and longitude columns
    merged_df = dfs[0]  # Start with the first dataframe
    for i in range(1, len(dfs)):
        print(dfs[i].shape)
        merged_df = pd.merge(merged_df, dfs[i], on=['Latitude', 'Longitude'])

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(testing_all_csv, index=False)
    print(f"All input CSV files are merged to {testing_all_csv}")
    print(merged_df.columns)
    print(merged_df["AMSR_SWE"].describe())
    print(merged_df["vpd"].describe())
    print(merged_df["pr"].describe())
    print(merged_df["tmmx"].describe())

if __name__ == "__main__":
    # Replace with the actual path to your folder
    gridmet_csv_folder = f"{work_dir}/gridmet_climatology/"
    merge_all_gridmet_amsr_csv_into_one(f"{work_dir}/testing_output/",
                                        f"{work_dir}/dem_all.csv",
                                        f"{work_dir}/testing_all_ready.csv")

