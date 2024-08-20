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

# all the training files are here
original_training_points = f"{work_dir}/training_points.csv"
pmw_training = f"{work_dir}/PMW_training.csv"
pmw_training_new = f"{work_dir}/PMW_training_new.csv"
landcover_training = f"{work_dir}/lc_data_train.csv"
ua_model_data = f"{work_dir}/ua_single_csv_file.csv" # this is not used due to no overlap with study area
fsca_training = f"{work_dir}/fSCA_trainingCells/fSCA_trainingCells_2019.csv"
snowclassification_training_data = f"{work_dir}/hackweek_testing2.csv"
amsr_gridmet_terrain_training_data = f"{work_dir}/training_data_20_years_cleaned.csv_hackweek_subset_all_years.csv"
# amsr_testing_data = f"{work_dir}/amsr_2017_2018_testing.csv"
# amsr_training_data = f"{work_dir}/amsr_2017_2018_training_points.csv"

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

  
def filter_2years_from_20_years():
  pd.set_option('display.max_columns', None)

  all_ready_csv_path = f'{work_dir}/training_data_20_years_cleaned.csv'

  #current_ready_csv_path = f'{work_dir}/testing_all_ready.csv'

  # Step 1: Read the CSV file into a pandas DataFrame
  file_path = all_ready_csv_path
  df = pd.read_csv(file_path)
  print(df.columns)

  # Step 2: Define the geometry bounding box
  # Replace these values with the actual bounding box coordinates
  min_lat, max_lat = 37.75, 38.75
  min_lon, max_lon = -119.75, -118.75

  # Step 3: Filter the DataFrame to keep rows within the geometry
  filtered_df = df[(df['lat'] >= min_lat) & 
                   (df['lat'] <= max_lat) & 
                   (df['lon'] >= min_lon) & 
                   (df['lon'] <= max_lon)]

  filtered_df['date'] = pd.to_datetime(filtered_df['date'])

  # Filter rows based on the date range (2017 to 2018)
  start_date = pd.to_datetime('2017-01-01')
  end_date = pd.to_datetime('2018-12-31')
  filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
  
  #lat_lon_df = filtered_df[['lat', 'lon']]

  # Step 4: Display or process the filtered data
  print("Filtered Data:")
  print(filtered_df.head())

  # Step 5: (Optional) Write the filtered data to a new CSV file
  subset_csv_path = f'{all_ready_csv_path}_hackweek_subset_all_years_testing.csv'
  filtered_df.to_csv(subset_csv_path, index=False)
  print(f"The subset of the rows is saved to {subset_csv_path}")

  # Step 6: copy it to the website folder
  destination_folder = "/var/www/html/swe_forecasting/"
  shutil.copy(subset_csv_path, destination_folder)
  

  # Step 7: Done!
  print("Done")



def merge_all_training_data_together():
  df5 = pd.read_csv(amsr_gridmet_terrain_training_data)
  df1 = pd.read_csv(landcover_training)
  df2 = pd.read_csv(pmw_training_new)
  df3 = pd.read_csv(fsca_training)
  df4 = pd.read_csv(snowclassification_training_data)
  df4 = df4.rename(columns={'Date': 'date', 
                           'long': 'lon'})

  merged_df = pd.merge(df5, df1, on=['date', 'lat', 'lon'], how='left')
  merged_df = pd.merge(merged_df, df2, on=['date', 'lat', 'lon'], how='left')
  print(df4.head())
  merged_df = pd.merge(merged_df, df3, on=['date', 'lat', 'lon'], how='left')
  merged_df = pd.merge(merged_df, df4, on=['date', 'lat', 'lon'], how='left')
  merged_df.drop("Unnamed: 0", axis=1, inplace=True)
  print(merged_df.columns)
  
  # Set a custom index using a combination of date, lat, and lon
  merged_df.set_index(['date', 'lat', 'lon'], inplace=True)

  # Remove duplicated rows based on the custom index
  merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

  # Reset the index to restore the original structure
  merged_df.reset_index(inplace=True)

  # # Print the merged DataFrame
  print(merged_df)
  training_hackweek_csv = f"{work_dir}/all_merged_training.csv"
  merged_df.to_csv(training_hackweek_csv, index=False)
  print(f"Training data is saved to {training_hackweek_csv}")
  

  
def filter_water_year_winter_months_only():
  
  df = pd.read_csv(f"{work_dir}/all_merged_training.csv")
  df['date'] = pd.to_datetime(df['date'])

  # Define the time range window
  start_date = pd.to_datetime('2017-10-01')
  end_date = pd.to_datetime('2018-07-01')

  # Create a boolean mask to filter rows within the time range
  mask = (df['date'] >= start_date) & (df['date'] <= end_date)

  # Apply the mask to the DataFrame to keep only the rows within the time range
  df = df[mask]
  print(df.head())
  df.to_csv(f"{work_dir}/all_merged_training_water_year_winter_month_only.csv")


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


def add_no_snow_data_points():
  current_training_file_path = f"{work_dir}/all_merged_training_cum_water_year_winter_month_only.csv"
  final_testing_file_path = f"{work_dir}/all_merged_testing_with_station_elevation.csv"
  final_training_file_path = f"{work_dir}/all_merged_training_water_year_winter_month_only_with_no_snow.csv"
  old_train_df = pd.read_csv(current_training_file_path)
  snotel_covered_len = len(old_train_df)
  print(f"the snotel covered area: {snotel_covered_len}")
  old_train_df.drop(["pmv", "SnowClass", "cumulative_fSCA"], axis=1, inplace=True)
  final_test_df = pd.read_csv(final_testing_file_path)
  zero_fsca_rows = final_test_df[final_test_df['fSCA'] == 0]
  # Use the sample method to randomly select rows
  chosen_no_snow_row_df = zero_fsca_rows.sample(n=snotel_covered_len, random_state=42)  # You can set a random_state for reproducibility
  chosen_no_snow_row_df["swe_value"] = 0
  # Use the list to select the subset of columns
  chosen_no_snow_row_df = chosen_no_snow_row_df[old_train_df.columns]
  print(f"len of no snow dataframe: {len(chosen_no_snow_row_df)}")
  concatenated_df = pd.concat([old_train_df, chosen_no_snow_row_df], ignore_index=True)
  concatenated_df.to_csv(final_training_file_path)
  print(f"final training data is saved to {final_training_file_path}")


# this is section for preparing training data
#convert_pmv_to_right_format()
# filter_2years_from_20_years()
# merge_all_training_data_together()
#filter_water_year_winter_months_only()
#create_accumulative_columns()
add_no_snow_data_points()



