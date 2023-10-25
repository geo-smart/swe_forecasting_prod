import pandas as pd
import os
from snowcast_utils import work_dir

ready_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned.csv'
dem_slope_csv_path = f"{work_dir}/slope_file.tif.csv"

# Read the cleaned ready CSV and DEM slope CSV
train_ready_df = pd.read_csv(ready_csv_path)
dem_slope_df = pd.read_csv(dem_slope_csv_path)

print(train_ready_df.head())
print(dem_slope_df.head())

print("all train.csv columns: ", train_ready_df.columns)
print("all dem slope columns: ", dem_slope_df.columns)

def replace_slope(row):
    '''
    Replace the 'slope' column in the input DataFrame row with the closest slope value from the DEM data.

    Args:
        row (pandas.Series): A row of data containing 'lat' and 'lon' columns.

    Returns:
        float: The closest slope value from the DEM data for the given latitude and longitude.
    '''
    target_lat = row["lat"]
    target_lon = row["lon"]
    # Calculate the squared distance to find the closest DEM point
    dem_slope_df['Distance'] = (dem_slope_df['Latitude'] - target_lat) ** 2 + (dem_slope_df['Longitude'] - target_lon) ** 2
    closest_row = dem_slope_df.loc[dem_slope_df['Distance'].idxmin()]
    return closest_row["Slope"]

# Apply the 'replace_slope' function to calculate and replace slope values in the DataFrame
train_ready_df['slope'] = train_ready_df.apply(lambda row: replace_slope(row), axis=1)

print(train_ready_df.head())
print(train_ready_df.columns)

new_result_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3.csv'

# Save the modified DataFrame to a new CSV file
train_ready_df.to_csv(new_result_csv_path, index=False)

