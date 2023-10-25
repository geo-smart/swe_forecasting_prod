from datetime import date, datetime, timedelta
import json
import math
import numpy as np
import os
import pandas as pd
# import ee
#import seaborn as sns
import matplotlib.pyplot as plt
#import geopandas as gpd
#import geojson

today = date.today()

# dd/mm/YY
d1 = today.strftime("%Y-%m-%d")
print("today date =", d1)

train_start_date = "2019-01-01"
train_end_date = "2022-12-31"

test_start_date = "2023-05-17"
test_end_date = d1
print("test start date: ", test_start_date)
print("test end date: ", test_end_date)

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

work_dir = f"{homedir}/gridmet_test_run"

# Define a function to convert the month to season
def month_to_season(month):
    if 3 <= month <= 5:
        return 1
    elif 6 <= month <= 8:
        return 2
    elif 9 <= month <= 11:
        return 3
    else:
        return 4

def calculateDistance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance (Euclidean) between two sets of coordinates (lat1, lon1) and (lat2, lon2).
    
    Parameters:
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.
    
    Returns:
    - float: The Euclidean distance between the two points.
    """
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def create_cell_location_csv():
    """
    Create a CSV file containing cell locations from a GeoJSON file.
    """
    # read grid cell
    gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells_eval.geojson"
    all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
    if os.path.exists(all_cell_coords_file):
        os.remove(all_cell_coords_file)

    grid_coords_df = pd.DataFrame(columns=["cell_id", "lat", "lon"])
    print(grid_coords_df.head())
    gridcells = geojson.load(open(gridcells_file))
    for idx, cell in enumerate(gridcells['features']):
        current_cell_id = cell['properties']['cell_id']
        cell_lon = np.unique(np.ravel(cell['geometry']['coordinates'])[0::2]).mean()
        cell_lat = np.unique(np.ravel(cell['geometry']['coordinates'])[1::2]).mean()
        grid_coords_df.loc[len(grid_coords_df.index)] = [current_cell_id, cell_lat, cell_lon]

    # grid_coords_np = grid_coords_df.to_numpy()
    # print(grid_coords_np.shape)
    grid_coords_df.to_csv(all_cell_coords_file, index=False)
    # np.savetxt(all_cell_coords_file, grid_coords_np[:, 1:], delimiter=",")
    # print(grid_coords_np.shape)

def get_latest_date_from_an_array(arr, date_format):
    """
    Get the latest date from an array of date strings.
    
    Parameters:
    - arr (list): List of date strings.
    - date_format (str): Date format for parsing the date strings.
    
    Returns:
    - str: The latest date string.
    """
    return max(arr, key=lambda x: datetime.strptime(x, date_format))

def findLastStopDate(target_testing_dir, data_format):
    """
    Find the last stop date from CSV files in a directory.
    
    Parameters:
    - target_testing_dir (str): Directory containing CSV files.
    - data_format (str): Date format for parsing the date strings.
    
    Returns:
    - str: The latest stop date.
    """
    date_list = []
    for filename in os.listdir(target_testing_dir):
        f = os.path.join(target_testing_dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and ".csv" in f:
            pdf = pd.read_csv(f, header=0, index_col=0)
            date_list = np.concatenate((date_list, pdf.index.unique()))
    latest_date = get_latest_date_from_an_array(date_list, data_format)
    print(latest_date)
    date_time_obj = datetime.strptime(latest_date, data_format)
    return date_time_obj.strftime("%Y-%m-%d")

def convert_date_from_1900(day_value):
    """
    Convert a day value since 1900 to a date string in the format "YYYY-MM-DD".
    
    Parameters:
    - day_value (int): Number of days since January 1, 1900.
    
    Returns:
    - str: Date string in "YYYY-MM-DD" format.
    """
    reference_date = datetime(1900, 1, 1)
    result_date = reference_date + timedelta(days=day_value)
    return result_date.strftime("%Y-%m-%d")

def convert_date_to_1900(date_string):
    """
    Convert a date string in the format "YYYY-MM-DD" to a day value since 1900.
    
    Parameters:
    - date_string (str): Date string in "YYYY-MM-DD" format.
    
    Returns:
    - int: Number of days since January 1, 1900.
    """
    input_date = datetime.strptime(date_string, "%Y-%m-%d")
    reference_date = datetime(1900, 1, 1)
    delta = input_date - reference_date
    day_value = delta.days
    return day_value

day_index = convert_date_to_1900(test_start_date)
#create_cell_location_csv()
#findLastStopDate(f"{github_dir}/data/sim_testing/gridmet/", "%Y-%m-%d %H:%M:%S")
#findLastStopDate(f"{github_dir}/data/sat_testing/sentinel1/", "%Y-%m-%d %H:%M:%S")
#findLastStopDate(f"{github_dir}/data/sat_testing/modis/", "%Y-%m-%d")

