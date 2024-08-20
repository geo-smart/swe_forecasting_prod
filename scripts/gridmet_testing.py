"""
Script for downloading specific variables of GridMET climatology data.

This script downloads specific meteorological variables from the GridMET climatology dataset
for a specified year. It uses the netCDF4 library for handling NetCDF files, urllib for downloading files,
and pandas for data manipulation. The script also removes existing files in the target folder before downloading.


Usage:
    Run this script to download specific meteorological variables for a specified year from the GridMET dataset.

"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import urllib.request
from datetime import datetime, timedelta, date
from snowcast_utils import test_start_date, work_dir
import matplotlib.pyplot as plt

# Define the folder to store downloaded files
gridmet_folder_name = f'{work_dir}/gridmet_climatology'

western_us_coords = f'{work_dir}/dem_file.tif.csv'


gridmet_var_mapping = {
  "etr": "potential_evapotranspiration",
  "pr":"precipitation_amount",
  "rmax":"relative_humidity",
  "rmin":"relative_humidity",
  "tmmn":"air_temperature",
  "tmmx":"air_temperature",
  "vpd":"mean_vapor_pressure_deficit",
  "vs":"wind_speed",
}
# Define the custom colormap with specified colors and ranges
colors = [
    (0.8627, 0.8627, 0.8627),  # #DCDCDC - 0 - 1
    (0.8627, 1.0000, 1.0000),  # #DCFFFF - 1 - 2
    (0.6000, 1.0000, 1.0000),  # #99FFFF - 2 - 4
    (0.5569, 0.8235, 1.0000),  # #8ED2FF - 4 - 6
    (0.4509, 0.6196, 0.8745),  # #739EDF - 6 - 8
    (0.4157, 0.4706, 1.0000),  # #6A78FF - 8 - 10
    (0.4235, 0.2784, 1.0000),  # #6C47FF - 10 - 12
    (0.5529, 0.0980, 1.0000),  # #8D19FF - 12 - 14
    (0.7333, 0.0000, 0.9176),  # #BB00EA - 14 - 16
    (0.8392, 0.0000, 0.7490),  # #D600BF - 16 - 18
    (0.7569, 0.0039, 0.4549),  # #C10074 - 18 - 20
    (0.6784, 0.0000, 0.1961),  # #AD0032 - 20 - 30
    (0.5020, 0.0000, 0.0000)   # #800000 - > 30
]

# Define your value ranges for color mapping
#value_ranges = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30]
#value_ranges = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2, 2.5, 3]

def create_color_maps_with_value_range(df_col, value_ranges=None):
  if value_ranges == None:
    max_value = df_col.max()
    min_value = df_col.min()
    if min_value < 0:
      min_value = 0
    step_size = (max_value - min_value) / 12

    # Create 10 periods
    new_value_ranges = [min_value + i * step_size for i in range(12)]
  # Define your custom function to map data values to colors
  def map_value_to_color(value):
    # Iterate through the value ranges to find the appropriate color index
    for i, range_max in enumerate(new_value_ranges):
      if value <= range_max:
        return colors[i]

      # If the value is greater than the largest range, return the last color
      return colors[-1]

    # Map predicted_swe values to colors using the custom function
  color_mapping = [map_value_to_color(value) for value in df_col.values]
  return color_mapping, new_value_ranges

def get_current_year():
    """
    Get the current year.

    Returns:
        int: The current year.
    """
    now = datetime.now()
    current_year = now.year
    return current_year

def remove_files_in_folder(folder_path, current_year):
    """
    Remove all files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder to remove files from.
    """
    # Get a list of files in the folder
    files = os.listdir(folder_path)

    # Loop through the files and remove them
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and str(current_year) in file_path and file_path.endswith(".nc"):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def download_file(url, target_file_path, variable):
    """
    Download a file from a URL and save it to a specified location.

    Parameters:
        url (str): URL of the file to download.
        target_file_path (str): Path where the downloaded file should be saved.
        variable (str): Name of the meteorological variable being downloaded.
    """
    try:
        with urllib.request.urlopen(url) as response:
            print(f"Downloading {url}")
            file_content = response.read()
        save_path = target_file_path
        with open(save_path, 'wb') as file:
            file.write(file_content)
        print(f"File downloaded successfully and saved as: {save_path}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")

def download_gridmet_of_specific_variables(year_list):
    """
    Download specific meteorological variables from the GridMET climatology dataset.
    """
    # Make a directory to store the downloaded files
    

    base_metadata_url = "http://www.northwestknowledge.net/metdata/data/"
    variables_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'etr', 'rmax', 'rmin', 'vs']

    for var in variables_list:
        for y in year_list:
            download_link = base_metadata_url + var + '_' + '%s' % y + '.nc'
            target_file_path = os.path.join(gridmet_folder_name, var + '_' + '%s' % y + '.nc')
            if not os.path.exists(target_file_path):
                download_file(download_link, target_file_path, var)
            else:
                print(f"File {target_file_path} exists")


def get_current_year():
    now = datetime.now()
    current_year = now.year
    return current_year


def get_file_name_from_path(file_path):
    # Get the file name from the file path
    file_name = os.path.basename(file_path)
    return file_name

def get_var_from_file_name(file_name):
    # Assuming the file name format is "tmmm_year.csv"
    var_name = str(file_name.split('_')[0])
    return var_name

def get_coordinates_of_template_tif():
  	# Load the CSV file and extract coordinates
    coordinates = []
    df = pd.read_csv(dem_csv)
    for index, row in df.iterrows():
        # Process each row here
        lon, lat = float(row["Latitude"]), float(row["Longitude"])
        coordinates.append((lon, lat))
    return coordinates

def find_nearest_index(array, value):
    # Find the index of the element in the array that is closest to the given value
    return (abs(array - value)).argmin()

def create_gridmet_to_dem_mapper(nc_file):
    western_us_dem_df = pd.read_csv(western_us_coords)
    # Check if the CSV already exists
    target_csv_path = f'{work_dir}/gridmet_to_dem_mapper.csv'
    if os.path.exists(target_csv_path):
        print(f"File {target_csv_path} already exists, skipping..")
        return
    
    # get the netcdf file and generate the csv file for every coordinate in the dem_template.csv
    selected_date = datetime.strptime(test_start_date, "%Y-%m-%d")
    # Read the NetCDF file
    with nc.Dataset(nc_file) as nc_file:
      
      # Get the values at each coordinate using rasterio's sample function
      latitudes = nc_file.variables['lat'][:]
      longitudes = nc_file.variables['lon'][:]
      
      def get_gridmet_var_value(row):
        # Perform your custom calculation here
        gridmet_lat_index = find_nearest_index(latitudes, float(row["Latitude"]))
        gridmet_lon_index = find_nearest_index(longitudes, float(row["Longitude"]))
        return latitudes[gridmet_lat_index], longitudes[gridmet_lon_index], gridmet_lat_index, gridmet_lon_index
    
      # Use the apply function to apply the custom function to each row
      western_us_dem_df[['gridmet_lat', 'gridmet_lon', 
                         'gridmet_lat_idx', 'gridmet_lon_idx',]] = western_us_dem_df.apply(lambda row: pd.Series(get_gridmet_var_value(row)), axis=1)
      western_us_dem_df.rename(columns={"Latitude": "dem_lat", 
                                        "Longitude": "dem_lon"}, inplace=True)
      
    # print(western_us_dem_df.head())
    
    # Save the new converted AMSR to CSV file
    western_us_dem_df.to_csv(target_csv_path, index=False)
    
    return western_us_dem_df
  
  
def get_nc_csv_by_coords_and_variable(nc_file,
                                      var_name,
                                      target_date=test_start_date):
    
    create_gridmet_to_dem_mapper(nc_file)
  	
    mapper_df = pd.read_csv(f'{work_dir}/gridmet_to_dem_mapper.csv')
    
    # get the netcdf file and generate the csv file for every coordinate in the dem_template.csv
    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    # Read the NetCDF file
    with nc.Dataset(nc_file) as nc_file:
      # Get a list of all variables in the NetCDF file
      variables = nc_file.variables.keys()
      
      # Get the values at each coordinate using rasterio's sample function
      latitudes = nc_file.variables['lat'][:]
      longitudes = nc_file.variables['lon'][:]
      day = nc_file.variables['day'][:]
      long_var_name = gridmet_var_mapping[var_name]
      var_col = nc_file.variables[long_var_name][:]
      #print("val_col.shape: ", var_col.shape)
      
      # Calculate the day of the year
      day_of_year = selected_date.timetuple().tm_yday
      day_index = day_of_year - 1
      #print('day_index:', day_index)
      
      def get_gridmet_var_value(row):
        # Perform your custom calculation here
        lat_index = int(row["gridmet_lat_idx"])
        lon_index = int(row["gridmet_lon_idx"])
        var_value = var_col[day_index, lat_index, lon_index]
        
        return var_value
    
      # Use the apply function to apply the custom function to each row
      # print(mapper_df.columns)
      # print(mapper_df.head())
      mapper_df[var_name] = mapper_df.apply(get_gridmet_var_value, axis=1)
      
      # print("mapper_df[var_name]: ", mapper_df[var_name].describe())
      
      # drop useless columns
      mapper_df = mapper_df[["dem_lat", "dem_lon", var_name]]
      mapper_df.rename(columns={"dem_lat": "Latitude",
                               "dem_lon": "Longitude"}, inplace=True)

      
    # print(mapper_df.head())
    return mapper_df


def turn_gridmet_nc_to_csv(target_date=test_start_date):
    
    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    generated_csvs = []
    for root, dirs, files in os.walk(gridmet_folder_name):
        for file_name in files:
            
            if str(selected_date.year) in file_name and file_name.endswith(".nc"):
                print(f"Checking file: {file_name}")
                var_name = get_var_from_file_name(file_name)
                # print("Variable name:", var_name)
                res_csv = f"{work_dir}/testing_output/{str(selected_date.year)}_{var_name}_{target_date}.csv"

                if os.path.exists(res_csv):
                    #os.remove(res_csv)
                    print(f"{res_csv} already exists. Skipping..")
                    generated_csvs.append(res_csv)
                    continue

                # Perform operations on each file here
                netcdf_file_path = os.path.join(root, file_name)
                print("Processing file:", netcdf_file_path)
                file_name = get_file_name_from_path(netcdf_file_path)

                df = get_nc_csv_by_coords_and_variable(netcdf_file_path, 
                                                       var_name, target_date)
                df.replace('--', pd.NA, inplace=True)
                df.to_csv(res_csv, index=False)
                print("gridmet var saved: ", res_csv)
                generated_csvs.append(res_csv)
    return generated_csvs   

def plot_gridmet(target_date=test_start_date):
  selected_date = datetime.strptime(target_date, "%Y-%m-%d")
  var_name = "pr"
  test_csv = f"{work_dir}/testing_output/{str(selected_date.year)}_{var_name}_{target_date}.csv"
  gridmet_var_df = pd.read_csv(test_csv)
  gridmet_var_df.replace('--', pd.NA, inplace=True)
  gridmet_var_df.dropna(inplace=True)
  gridmet_var_df['pr'] = pd.to_numeric(gridmet_var_df['pr'], errors='coerce')
  #print(gridmet_var_df.head())
  #print(gridmet_var_df["Latitude"].describe())
  #print(gridmet_var_df["Longitude"].describe())
  #print(gridmet_var_df["pr"].describe())
  
  colormaplist, value_ranges = create_color_maps_with_value_range(gridmet_var_df[var_name])
  
  # Create a scatter plot
  plt.scatter(gridmet_var_df["Longitude"].values, 
              gridmet_var_df["Latitude"].values, 
              label='Pressure', 
              color=colormaplist, 
              marker='o')

  # Add labels and a legend
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('Scatter Plot Example')
  plt.legend()
  
  res_png_path = f"{work_dir}/testing_output/{str(selected_date.year)}_{var_name}_{target_date}.png"
  plt.savefig(res_png_path)
  print(f"test image is saved at {res_png_path}")
                
def prepare_folder_and_get_year_list(target_date=test_start_date):
  # Check if the folder exists, if not, create it
  if not os.path.exists(gridmet_folder_name):
      os.makedirs(gridmet_folder_name)

  selected_date = datetime.strptime(target_date, "%Y-%m-%d")
  if selected_date.month < 10:
    past_october_1 = datetime(selected_date.year - 1, 10, 1)
  else:
    past_october_1 = datetime(selected_date.year, 10, 1)
  year_list = [selected_date.year, past_october_1.year]

  # Remove any existing files in the folder
  if selected_date.year == datetime.now().year:
    # check if the current year's netcdf contains the selected date
    # get etr netcdf and read
    nc_file = f"{gridmet_folder_name}/tmmx_{selected_date.year}.nc"
    ifremove = False
    if os.path.exists(nc_file):
      with nc.Dataset(nc_file) as ncd:
        day = ncd.variables['day'][:]
        # Calculate the day of the year
        day_of_year = selected_date.timetuple().tm_yday
        day_index = day_of_year - 1
        if len(day) <= day_index:
          ifremove = True
    
    if ifremove:
      print("The current year netcdf has new data. Redownloading..")
      remove_files_in_folder(gridmet_folder_name, selected_date.year)  # only redownload when the year is the current year
    else:
      print("The existing netcdf already covers the selected date. Avoid downloading..")
  return year_list

def add_cumulative_column(df, column_name):
  df[f'cumulative_{column_name}'] = df[column_name].sum()
  return df
    

def prepare_cumulative_history_csvs(target_date=test_start_date, force=False):
  """
    Prepare cumulative history CSVs for a specified target date.

    Parameters:
    - target_date (str, optional): The target date in the format 'YYYY-MM-DD'. Default is 'test_start_date'.
    - force (bool, optional): If True, forcefully regenerate cumulative CSVs even if they already exist. Default is False.

    Returns:
    None

    This function generates cumulative history CSVs for a specified target date. It traverses the date range from the past
    October 1 to the target date, downloads gridmet data, converts it to CSV, and merges it into a big DataFrame.
    The cumulative values are calculated and saved in new CSV files.

    Example:
    ```python
    prepare_cumulative_history_csvs(target_date='2023-01-01', force=True)
    ```

    Note: This function assumes the existence of the following helper functions:
    - download_gridmet_of_specific_variables
    - prepare_folder_and_get_year_list
    - turn_gridmet_nc_to_csv
    - add_cumulative_column
    - process_group_value_filling
    ```

    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    print(selected_date)
    if selected_date.month < 10:
        past_october_1 = datetime(selected_date.year - 1, 10, 1)
    else:
        past_october_1 = datetime(selected_date.year, 10, 1)

    # Rest of the function logic...

    filled_data = filled_data.loc[:, ['Latitude', 'Longitude', var_name, f'cumulative_{var_name}']]
    print("new_df final shape: ", filled_data.head())
    filled_data.to_csv(cumulative_target_path, index=False)
    print(f"new df is saved to {cumulative_target_path}")
    print(filled_data.describe())
    ```
Note: This docstring includes placeholders such as "download_gridmet_of_specific_variables" and "prepare_folder_and_get_year_list" for the assumed existence of related helper functions. You should replace these placeholders with actual documentation for those functions.
  """
  selected_date = datetime.strptime(target_date, "%Y-%m-%d")
  print(selected_date)
  if selected_date.month < 10:
    past_october_1 = datetime(selected_date.year - 1, 10, 1)
  else:
    past_october_1 = datetime(selected_date.year, 10, 1)

  # Traverse and print every day from past October 1 to the specific date
  current_date = past_october_1
  
  date_keyed_objects = {}
  
  download_gridmet_of_specific_variables(
    prepare_folder_and_get_year_list(target_date=target_date)
  )
  
  while current_date <= selected_date:
    print(current_date.strftime('%Y-%m-%d'))
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    
    generated_csvs = turn_gridmet_nc_to_csv(target_date=current_date_str)
    
    # read the csv into dataframe and merge to the big dataframe
    date_keyed_objects[current_date_str] = generated_csvs
    
    current_date += timedelta(days=1)
    
  print("date_keyed_objects: ", date_keyed_objects)
  target_generated_csvs = date_keyed_objects[target_date]
  for index, single_csv in enumerate(target_generated_csvs):
    # traverse the variables of gridmet here
    # each variable is a loop
    print(f"creating cumulative for {single_csv}")
    
    cumulative_target_path = f"{single_csv}_cumulative.csv"
    print("cumulative_target_path = ", cumulative_target_path)
    
    if os.path.exists(cumulative_target_path) and not force:
      print(f"{cumulative_target_path} already exists, skipping..")
      continue
    
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(single_csv))[0]
    gap_filled_csv = f"{cumulative_target_path}_gap_filled.csv"

	# Split the file name using underscores
    var_name = file_name.split('_')[1]
    print(f"Found variable name {var_name}")
    current_date = past_october_1
    new_df = pd.read_csv(single_csv)
    print(new_df.head())
    
    all_df = pd.read_csv(f"{work_dir}/testing_output/{str(selected_date.year)}_{var_name}_{target_date}.csv")
    all_df["date"] = target_date
    all_df[var_name] = pd.to_numeric(all_df[var_name], errors='coerce')
    
#     while current_date <= selected_date:
#       print(current_date.strftime('%Y-%m-%d'))
#       current_date_str = current_date.strftime('%Y-%m-%d')
#       #current_generated_csv = date_keyed_objects[current_date_str][index]
#       current_generated_csv = f"{work_dir}/testing_output/{str(current_date.year)}_{var_name}_{current_date_str}.csv"
#       print(f"reading file: {current_generated_csv}")
#       previous_df = pd.read_csv(current_generated_csv)
#       previous_df["date"] = current_date_str
#       previous_df[var_name] = pd.to_numeric(previous_df[var_name], errors='coerce')
      
#       if all_df is None:
#         all_df = previous_df
#       else:
#         all_df = pd.concat([all_df, previous_df], ignore_index=True)
      
#       current_date += timedelta(days=1)
    
    
#     print("all df head: ", all_df.shape, all_df[var_name].describe())
    
    # add all the columns together and save to new csv
    # Adding all columns except latitude and longitude
    
#     def process_group_value_filling(group, var_name, target_date):
#       # Sort the group by 'date'
# #       group = group.sort_values(by='date')
#       # no need to interpolate for gridmet, only add the cumulative columns
#       cumvalue = group[var_name].sum()
#       group = group[(group['date'] == target_date)]
#       group[f'cumulative_{var_name}'] = cumvalue
#       return group

#     grouped = all_df.groupby(['Latitude', 'Longitude'])
#     # Apply the function to each group
#     filled_data = grouped.apply(lambda group: process_group_value_filling(group, var_name, target_date)).reset_index(drop=True)
    filled_data = all_df
    filled_data = filled_data[(filled_data['date'] == target_date)]
    
    filled_data.fillna(0, inplace=True)
    
#     print("filled_data.shape = ", filled_data.shape)
#     print("filled_data.head = ", filled_data.head())
#     print(f"filled_data[{var_name}].unique() = {filled_data[var_name].describe()}")
    # only retain the rows of the target date
    
	
    print("Finished correctly ", filled_data.head())
    #filled_data.to_csv(gap_filled_csv, index=False)
    #print(f"New filled values csv is saved to {gap_filled_csv}_gap_filled.csv")
    
    filled_data = filled_data[['Latitude', 'Longitude', 
                               var_name, 
#                                f'cumulative_{var_name}'
                              ]]
    print(filled_data.shape)
    filled_data.to_csv(cumulative_target_path, index=False)
    print(f"new df is saved to {cumulative_target_path}")
    print(filled_data.describe())


if __name__ == "__main__":
  # Run the download function
#   download_gridmet_of_specific_variables(prepare_folder_and_get_year_list())
#   turn_gridmet_nc_to_csv()
#   plot_gridmet()

  # prepare testing data with cumulative variables
  prepare_cumulative_history_csvs(force=True)


