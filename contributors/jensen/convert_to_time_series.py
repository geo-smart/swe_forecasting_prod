import pandas as pd
import os
from snowcast_utils import work_dir
import shutil
import numpy as np

pd.set_option('display.max_columns', None)

current_ready_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3.csv'

target_time_series_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_v1.csv'

backup_time_series_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_v1_bak.csv'

def array_describe(arr):
    stats = {
        'Mean': np.mean(arr),
        'Median': np.median(arr),
        'Standard Deviation': np.std(arr),
        'Variance': np.var(arr),
        'Minimum': np.min(arr),
        'Maximum': np.max(arr),
        'Sum': np.sum(arr),
    }
    
    return stats

def convert_to_time_series():
  columns_to_be_time_series = ["SWE", "Flag", 'air_temperature_tmmn',
  'potential_evapotranspiration', 'mean_vapor_pressure_deficit',
  'relative_humidity_rmax', 'relative_humidity_rmin',
  'precipitation_amount', 'air_temperature_tmmx', 'wind_speed',]
  # Read the cleaned ready CSV and DEM slope CSV
  df = pd.read_csv(current_ready_csv_path)
  # df['location'] = df['lat'].astype(str) + ',' + df['lon'].astype(str)
  # unique_location_pairs = df.drop_duplicates(subset='location')[['lat', 'lon']]

  # print(unique_location_pairs)
  # unique_date = df.drop_duplicates(subset='date')[['date']]
  # print(unique_date)

  # add a 7 days time series to each row
  df.sort_values(by=['lat', 'lon', 'date'], inplace=True)

  
  
  # fill in the missing values of AMSR and gridMet using polynomial values
  # Function to perform polynomial interpolation
  def interpolate_missing_inplace(df, column_name, degree=3):
    x = df.index
    y = df[column_name]

    # Create a mask for missing values
    mask = y > 240
    # Perform interpolation
    new_y = np.interp(x, x[~mask], y[~mask])
    
    if np.any(new_y > 240):
      print("mask: ", mask)
      print("x[~mask]: ", x[~mask])
      print("y[~mask]: ", y[~mask])
      print("new_y: ", new_y)
      raise ValueError("Single group: shouldn't have values > 240 here")

    # Replace missing values with interpolated values
    df[column_name] = new_y
    #print(df[column_name].describe())
    return df
    

  # Group by location and apply interpolation for each column
  # Group the data by 'lat' and 'lon'
  grouped = df.groupby(['lat', 'lon'])
  filled_data = pd.DataFrame()
  for name, group in grouped:
    print(f"Start to filling missing values..{name}")
    new_df = interpolate_missing_inplace(group, 'SWE')
    filled_data = pd.concat([filled_data, group], axis=0)

  filled_data = filled_data.reset_index()
  
  filled_data.reset_index(inplace=True)
  
  if any(filled_data['SWE'] > 240):
    raise ValueError("Error: shouldn't have SWE>240 at this point")
    

  # Create a new DataFrame to store the time series data for each location
  result = pd.DataFrame()

  # Define the number of days to consider (7 days in this case)
  num_days = 7
  
  grouped = filled_data.groupby(['lat', 'lon'])
  for name, group in grouped:
      group = group.set_index('date')
      for day in range(1, num_days + 1):
        for target_col in columns_to_be_time_series:
          new_column_name = f'{target_col}_{day}'
          group[new_column_name] = group[target_col].shift(day)
      result = pd.concat([result, group], axis=0)

  # Reset the index of the result DataFrame
  result = result.reset_index()
  result.to_csv(target_time_series_csv_path, index=False)
  print(f"new data is saved to {target_time_series_csv_path}")
  shutil.copy(target_time_series_csv_path, backup_time_series_csv_path)
  print(f"file is backed up to {backup_time_series_csv_path}")


convert_to_time_series()

# df = pd.read_csv(target_time_series_csv_path)

# print(df.columns)

# df.head()

# description = df.describe(include='all')
# # Print the description
# print(description)


