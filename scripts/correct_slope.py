import pandas as pd
import os
from snowcast_utils import work_dir
from scipy.spatial import KDTree
import dask.dataframe as dd
from dask.distributed import Client

ready_csv_path = f'{work_dir}/final_merged_data_4yrs_snotel_and_ghcnd_stations.csv_sorted.csv'
dem_slope_csv_path = f"{work_dir}/slope_file.tif.csv"
print(f"ready_csv_path = {ready_csv_path}")
new_result_csv_path = f'{work_dir}/final_merged_data_4yrs_snotel_ghcnd.csv_sorted_slope_corrected.csv'


def replace_slope(row, tree, dem_df):
    '''
    Replace the 'slope' column in the input DataFrame row with the closest slope value from the DEM data.

    Args:
        row (pandas.Series): A row of data containing 'lat' and 'lon' columns.
        tree (scipy.spatial.KDTree): KDTree built from DEM data.
        dem_df (pandas.DataFrame): DataFrame containing DEM data.

    Returns:
        float: The closest slope value from the DEM data for the given latitude and longitude.
    '''
    # print("row = ", row)
    target_lat = row["lat"]
    target_lon = row["lon"]
    _, idx = tree.query([(target_lat, target_lon)])
    closest_row = dem_df.iloc[idx[0]]
    return closest_row["Slope"]

def parallelize_slope_correction():
    # Start Dask client
#     client = Client()

    # Scatter DEM data
#     dem_future = client.scatter(dem_slope_df)
    # Read the cleaned ready CSV and DEM slope CSV
    train_ready_df = pd.read_csv(ready_csv_path)
    dem_slope_df = pd.read_csv(dem_slope_csv_path)

    print(train_ready_df.head())
    print(dem_slope_df.head())

    print("all train.csv columns: ", train_ready_df.columns)
    print("all dem slope columns: ", dem_slope_df.columns)
    
    # Build KDTree for DEM data
    tree = KDTree(dem_slope_df[['Latitude', 'Longitude']].values)
    
    print("deduplicate the training point lat/lon")
    print("train_ready_df.shape = ", train_ready_df.shape)
    lat_lon_df = train_ready_df[['lat', 'lon']].drop_duplicates()

    print("lat_lon_df.shape", lat_lon_df.shape)
    # Apply the 'replace_slope' function to calculate and replace slope values in the DataFrame
    print("start to correct slope")
    #train_ready_df['corrected_slope'] = train_ready_df.apply(replace_slope, args=(tree, dem_slope_df), axis=1)
    
    # Apply the function with scattered data and log progress
#     train_ready_ddf['corrected_slope'] = train_ready_ddf.map_partitions(
#         lambda df: df.apply(replace_slope, args=(tree, dem_future)), 
#         meta=('slope', 'float64')
#     )
#     train_ready_ddf['corrected_slope'].compute(progress_callback=progress)

    lat_lon_df['corrected_slope'] = lat_lon_df.apply(replace_slope, args=(tree, dem_slope_df), axis=1)
  
    train_ready_df = train_ready_df.merge(lat_lon_df, on=['lat', 'lon'], how='left')
    
    print("The new train_ready_df.shape with corrected slope is ", train_ready_df.shape)

    print("finished correcting slope")
    print(train_ready_df.head())
    print(train_ready_df.columns)

    
    print(f"saving the correct data into {new_result_csv_path}")
    # Save the modified DataFrame to a new CSV file
    train_ready_df.to_csv(new_result_csv_path, index=False)
    print(f"The new slope corrected dataframe is saved to {new_result_csv_path}")
    
  
  
if __name__ == "__main__":
    parallelize_slope_correction()

