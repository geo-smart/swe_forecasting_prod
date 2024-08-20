# This process is to merge the land cover data into the training.csv
import glob
import pandas as pd
from snowcast_utils import work_dir
import re
from datetime import datetime
import os
import shutil
import numpy as np
from gridmet_testing import download_gridmet_of_specific_variables, turn_gridmet_nc_to_csv, gridmet_var_mapping
from datetime import datetime, timedelta

# list all the csvs to merge here
dem_data = f"{work_dir}/dem_all.csv"

# all the testing files are here
original_testing_points = f"{work_dir}/testing_points.csv"
landcover_testing = f"{work_dir}/lc_data_test.csv"
pmw_testing = f"{work_dir}/PMW_testing.csv"
pmw_testing_new = f"{work_dir}/pmw_testing_new.csv"
snowclassification_testing_data = f"{work_dir}/snowclassification_hackweek_testing.csv"
terrain_testing_data = f"{work_dir}/dem_all.csv_hackweek_subset_testing.csv"
fsca_testing_data = f"{work_dir}/fsca_testing_all_years.csv"
gridmet_testing_data = f"{work_dir}/gridmet_testing_hackweek_subset.csv"


def convert_pmv_to_right_format():
  # convert pmv data into the same format
  column_names = ["date", "lat", "lon", "pmv"]
  
  #pmv_new_df = pd.DataFrame(columns=column_names)
  
  def adjust_column_to_rows(row):
	
    for property_name, value in row.items():
      #print("property_name: ", property_name)
      if property_name == "Time":
          continue

      column_data = row[property_name]  # Access the column data
      #match = re.search(r'\((.*?),\s(.*?)\)', property_name) # for training
      match = re.search(r'([-+]?\d+\.\d+), ([-+]?\d+\.\d+)', property_name) # for testing data

      # Convert the input time string to a datetime object
      datetime_obj = datetime.strptime(row["Time"], "%m/%d/%Y %H:%M")

      # Convert the datetime object to the desired format
      formatted_time_string = datetime_obj.strftime("%Y-%m-%d")

      if match:
          lat = match.group(1)
          lon = match.group(2)
          new_row_data = [formatted_time_string, lat, lon, column_data]
          
          #print("Latitude:", lat, "Longitude:", lon, "pmw:", new_row_data)

          # Create a new DataFrame with the new row
          new_row_df = pd.DataFrame([new_row_data], columns=column_names)

          # Concatenate the new row DataFrame with the original DataFrame
          #pmv_new_df = pd.concat([pmv_new_df, new_row_df], ignore_index=True)
          return pd.Series(new_row_data)

      else:
          match = re.search(r'\((.*?),\s(.*?)', property_name)
          lat = match.group(1)
          lon = match.group(2)
          #print("Latitude:", lat)
          #print("Longitude:", lon)
          new_row_data = [formatted_time_string, lat, lon, column_data]

          # Create a new DataFrame with the new row
          new_row_df = pd.DataFrame([new_row_data], columns=column_names)

          # Concatenate the new row DataFrame with the original DataFrame
          #pmv_new_df = pd.concat([pmv_new_df, new_row_df], ignore_index=True)
          return pd.Series(new_row_data)
  
  pmv_old_df = pd.read_csv(pmw_testing)
  
  pmv_new_df = pmv_old_df.apply(lambda row: adjust_column_to_rows(row), axis=1)
  
  pmv_new_df.columns = column_names

  print(pmv_new_df.head())
  pmv_new_df.to_csv(pmw_testing_new, index=False)
  print(f"New PMV file is saved!!! {pmw_training_new}")

def collect_gridmet_for_testing():
  testing_points_df = pd.read_csv(original_testing_points)
  print(testing_points_df.head())
  
  # download gridmet
  download_gridmet_of_specific_variables([2017, 2018])
  
  # convert all the dates in the year to csvs
  start_date = datetime(2017, 10, 1)
  end_date = datetime(2018, 7, 1)

  # Define the step (1 day)
  step = timedelta(days=1)

  # Initialize the current date with the start date
  current_date = start_date
  
  # Traverse the days using a loop
  all_days_df = None
  while current_date <= end_date:
      current_date_str = current_date.strftime('%Y-%m-%d')
      print(f"processing {current_date_str}")
      #turn_gridmet_nc_to_csv(current_date.strftime('%Y-%m-%d'))
      # example file: 2017_etr_2017-10-01_hackweek_subset.csv
      # step 1: combine all the variables
      single_day_df = None

      year = current_date.year
      for key in gridmet_var_mapping.keys():
        print(key)
        single_year_var_df = pd.read_csv(f"{work_dir}/testing_output/{year}_{key}_{current_date_str}_hackweek_subset.csv")
        single_year_var_df["date"] = current_date_str
        if single_day_df is None or single_day_df.empty:
          single_day_df = single_year_var_df
        else:
          single_day_df = pd.merge(single_day_df, single_year_var_df, on=["Latitude", "Longitude", "date"])

      
      if all_days_df is None or all_days_df.empty:
        all_days_df = single_day_df
      else:
        all_days_df = pd.concat([all_days_df, single_day_df], axis=0) 
        
      
      current_date += step
  print(all_days_df.head())
  all_days_df.to_csv(gridmet_testing_data, index=False)
  print(f"Ok, all gridmet data for testing days are saved to {gridmet_testing_data}")
  
def create_gridmet_dem_mapper_subset():
  all_mapper = pd.read_csv(f'{work_dir}/gridmet_to_dem_mapper.csv')
  min_lat, max_lat = 37.75, 38.75
  min_lon, max_lon = -119.75, -118.75
  filtered_df = all_mapper[(all_mapper['dem_lat'] >= min_lat) & 
                   (all_mapper['dem_lat'] <= max_lat) & 
                   (all_mapper['dem_lon'] >= min_lon) & 
                   (all_mapper['dem_lon'] <= max_lon)]
  subset_csv_path = f'{work_dir}/gridmet_to_dem_mapper_hackweek_subset.csv'
  filtered_df.to_csv(subset_csv_path, index=False)
  print(f"The subset of the rows is saved to {subset_csv_path}")

  # Step 6: copy it to the website folder
  destination_folder = "/var/www/html/swe_forecasting/"
  shutil.copy(subset_csv_path, destination_folder)
  
  # Step 7: Done!
  print("Done")

def collect_terrain_for_testing():
  testing_points_df = pd.read_csv(original_testing_points)
  print(testing_points_df.head())
  
  dem_all_df = pd.read_csv(dem_data)
  print(dem_all_df.head())
  
  # Step 2: Define the geometry bounding box
  # Replace these values with the actual bounding box coordinates
  min_lat, max_lat = 37.75, 38.75
  min_lon, max_lon = -119.75, -118.75

  dem_all_df.rename(columns={'Latitude': 'lat', 
                         'Longitude': 'lon',
                         'Elevation': 'elevation',
                         'Slope': 'slope',
                         'Aspect': 'aspect',
                         'Curvature': 'curvature',
                         'Northness': 'northness',
                         'Eastness': 'eastness'
                        }, inplace=True)
  
  # Step 3: Filter the DataFrame to keep rows within the geometry
  filtered_df = dem_all_df[(dem_all_df['lat'] >= min_lat) & 
                   (dem_all_df['lat'] <= max_lat) & 
                   (dem_all_df['lon'] >= min_lon) & 
                   (dem_all_df['lon'] <= max_lon)]

  #filtered_df['date'] = pd.to_datetime(filtered_df['date'])

  # Filter rows based on the date range (2017 to 2018)
  #start_date = pd.to_datetime('2017-01-01')
  #end_date = pd.to_datetime('2018-12-31')
  #filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
  
  #lat_lon_df = filtered_df[['lat', 'lon']]

  # Step 4: Display or process the filtered data
  print("Filtered Data:")
  print(filtered_df.head())

  # Step 5: (Optional) Write the filtered data to a new CSV file
  subset_csv_path = f'{dem_data}_hackweek_subset_testing.csv'
  filtered_df.to_csv(subset_csv_path, index=False)
  print(f"The subset of the rows is saved to {subset_csv_path}")

  # Step 6: copy it to the website folder
  destination_folder = "/var/www/html/swe_forecasting/"
  shutil.copy(subset_csv_path, destination_folder)
  

  # Step 7: Done!
  print("Done")

def collect_amsr_for_testing():
  # it seems we might not need AMSR
  pass
  

def merge_fsca_testing():
  
  #fsca_testing_single_year_folder = f"{work_dir}/fSCA_testingCells/"
  # Define a list to store the paths to the CSV files
  csv_files = glob.glob(f'{work_dir}/fSCA_testingCells/*.csv')  # Replace 'path/to/files/' with the actual path to your CSV files
  

  # Initialize an empty list to store DataFrames
  dataframes = []

  # Iterate through the CSV files and read them into DataFrames
  for csv_file in csv_files:
      df = pd.read_csv(csv_file)
      dataframes.append(df)

  # Concatenate the DataFrames into one
  merged_df = pd.concat(dataframes, ignore_index=True)
  data_sanity_checks(merged_df)
  
  # Now, 'merged_df' contains the merged data from all CSV files
  print(merged_df)
  final_fsca_testing_file = f'{work_dir}/fsca_testing_all_years.csv'
  merged_df.to_csv(final_fsca_testing_file, index=False)
  print(f"All years of data are saved to {final_fsca_testing_file}")
  
  

def create_accumulative_columns(csv_file_path=None):
  if csv_file_path is None:
  	csv_file_path = f"{work_dir}/all_merged_training_water_year_winter_month_only.csv"
  current_df = pd.read_csv(csv_file_path)
  print(current_df.head())
  current_df['date'] = pd.to_datetime(current_df['date'])
  #current_df['fSCA'] = current_df['fSCA'].fillna(0)

  # Group the DataFrame by 'lat' and 'lon'
  grouped = current_df.groupby(['lat', 'lon'], group_keys=False)
  
  # Sort each group by date and calculate cumulative precipitation
  
  cum_columns = ["etr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "pr"]
  for column in cum_columns:
  	current_df[f'cumulative_{column}'] = grouped.apply(lambda group: group.sort_values('date')[column].cumsum())

  print(current_df.head())
  hackweek_cum_csv = f"{csv_file_path}_cum.csv"
  current_df.to_csv(hackweek_cum_csv, index=False)
  print(f"All the cumulative variables are added successfully! {hackweek_cum_csv}")

def add_elevation_in_feet():
  all_df = pd.read_csv(f"{work_dir}/all_merged_testing_cum_water_year_winter_month_only.csv")
  all_df['station_elevation'] = all_df['elevation'] * 3.28084
  all_df.to_csv(f"{work_dir}/all_merged_testing_with_station_elevation.csv")
  print(f"all data is saved to {work_dir}/all_merged_testing_with_station_elevation.csv")
  

def data_sanity_checks(df):
  #all_csv_df = pd.read_csv(f"{work_dir}/all_merged_testing_with_station_elevation.csv")
  #points_df = pd.read_csv(data_path)
  # Get unique locations
  unique_locations = df[['lat', 'lon']].drop_duplicates()

  # Display the unique locations DataFrame
  #print(unique_locations)
  print(len(unique_locations))
  if len(unique_locations) < 700:
    raise ValueError("Number of unique locations is less than 700")

def merge_all_testing_data_together():
  df5 = pd.read_csv(gridmet_testing_data)
  df5 = df5.rename(columns={'Latitude': 'lat', 
                           'Longitude': 'lon'})
  data_sanity_checks(df5)
  
  df1 = pd.read_csv(landcover_testing)
  data_sanity_checks(df1)
  #df1['date'] = df1['date'].dt.strftime('%Y-%m-%d')
  #df2 = pd.read_csv(pmw_testing_new)
  #data_sanity_checks(df2)
  #df2['date'] = df2['date'].dt.strftime('%Y-%m-%d')
  #print("d2 types: ", df2.dtypes)
  df3 = pd.read_csv(fsca_testing_data)
  data_sanity_checks(df3)
  #df3['date'] = df3['date'].dt.strftime('%Y-%m-%d')
  print(df3.dtypes)
  #df4 = pd.read_csv(snowclassification_testing_data)
  
#   df4 = df4.rename(columns={'Date': 'date', 
#                            'long': 'lon'})
#   data_sanity_checks(df4)
  #df4['date'] = df4['date'].dt.strftime('%Y-%m-%d')
  #print(df4.head())
  df6 = pd.read_csv(terrain_testing_data)
  df6 = df6.rename(columns={'Latitude': 'lat', 
                           'Longitude': 'lon'})
  data_sanity_checks(df6)
  #df6['date'] = df6['date'].dt.strftime('%Y-%m-%d')

  merged_df = pd.merge(df5, df1, on=['date', 'lat', 'lon'], how='left')
  #merged_df = pd.merge(merged_df, df2, on=['date', 'lat', 'lon'], how='left')
  #merged_df["pmv"] = 0
  merged_df = pd.merge(merged_df, df3, on=['date', 'lat', 'lon'], how='left')
  print("check head: ", merged_df.head())
  #merged_df = pd.merge(merged_df, df4, on=['date', 'lat', 'lon'], how='left')
  merged_df = pd.merge(merged_df, df6, on=['lat', 'lon'], how='left')
#   merged_df = pd.merge(merged_df, df3, on=['date', 'lat', 'lon'], how='left')
#   merged_df = pd.merge(merged_df, df4, on=['date', 'lat', 'lon'], how='left')
  #merged_df.drop("Unnamed: 0", axis=1, inplace=True)
  print(merged_df.columns)
  
  # Set a custom index using a combination of date, lat, and lon
  merged_df.set_index(['date', 'lat', 'lon'], inplace=True)

  # Remove duplicated rows based on the custom index
  merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

  # Reset the index to restore the original structure
  merged_df.reset_index(inplace=True)

  # # Print the merged DataFrame
  print(merged_df)
  training_hackweek_csv = f"{work_dir}/all_merged_testing.csv"
  merged_df.to_csv(training_hackweek_csv, index=False)
  print(f"Training data is saved to {training_hackweek_csv}")  


# this is section for preparing testing data
#create_gridmet_dem_mapper_subset()
#collect_gridmet_for_testing()
#collect_terrain_for_testing()
#collect_amsr_for_testing()
#merge_fsca_testing()
#merge_all_testing_data_together()
#create_accumulative_columns(f"{work_dir}/all_merged_testing.csv")
#add_elevation_in_feet()






