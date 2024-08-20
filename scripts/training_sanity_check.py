# Do sanity checks on the training.csv to make sure all the columns' vales are extracted correctly
from snowcast_utils import work_dir
import pandas as pd
from gridmet_testing import download_gridmet_of_specific_variables
from datetime import datetime
import xarray as xr
from datetime import date
import numpy as np


# pick the final training csv
current_training_csv_path = f'{work_dir}/final_merged_data_3yrs_all_active_stations_v1.csv_sorted.csv_time_series_cumulative_v1.csv'
df = pd.read_csv(current_training_csv_path)

print("all the current columns: ", df.columns)

# choose several random sample rows for sanity checks
sample_size = 10
random_sample = df.sample(n=sample_size)

print(random_sample)

# all the current columns: Index(['date', 'level_0', 'index', 'lat', 'lon', 'SWE', 'Flag', 'swe_value',
# 'Unnamed: 0', 'air_temperature_tmmn', 'potential_evapotranspiration',
# 'mean_vapor_pressure_deficit', 'relative_humidity_rmax',
# 'relative_humidity_rmin', 'precipitation_amount',
# 'air_temperature_tmmx', 'wind_speed', 'elevation', 'slope', 'curvature',
# 'aspect', 'eastness', 'northness', 'SWE_1', 'Flag_1',
# 'air_temperature_tmmn_1', 'potential_evapotranspiration_1',
# 'mean_vapor_pressure_deficit_1', 'relative_humidity_rmax_1',
# 'relative_humidity_rmin_1', 'precipitation_amount_1',
# 'air_temperature_tmmx_1', 'wind_speed_1', 'SWE_2', 'Flag_2',
# 'air_temperature_tmmn_2', 'potential_evapotranspiration_2',
# 'mean_vapor_pressure_deficit_2', 'relative_humidity_rmax_2',
# 'relative_humidity_rmin_2', 'precipitation_amount_2',
# 'air_temperature_tmmx_2', 'wind_speed_2', 'SWE_3', 'Flag_3',
# 'air_temperature_tmmn_3', 'potential_evapotranspiration_3',
# 'mean_vapor_pressure_deficit_3', 'relative_humidity_rmax_3',
# 'relative_humidity_rmin_3', 'precipitation_amount_3',
# 'air_temperature_tmmx_3', 'wind_speed_3', 'SWE_4', 'Flag_4',
# 'air_temperature_tmmn_4', 'potential_evapotranspiration_4',
# 'mean_vapor_pressure_deficit_4', 'relative_humidity_rmax_4',
# 'relative_humidity_rmin_4', 'precipitation_amount_4',
# 'air_temperature_tmmx_4', 'wind_speed_4', 'SWE_5', 'Flag_5',
# 'air_temperature_tmmn_5', 'potential_evapotranspiration_5',
# 'mean_vapor_pressure_deficit_5', 'relative_humidity_rmax_5',
# 'relative_humidity_rmin_5', 'precipitation_amount_5',
# 'air_temperature_tmmx_5', 'wind_speed_5', 'SWE_6', 'Flag_6',
# 'air_temperature_tmmn_6', 'potential_evapotranspiration_6',
# 'mean_vapor_pressure_deficit_6', 'relative_humidity_rmax_6',
# 'relative_humidity_rmin_6', 'precipitation_amount_6',
# 'air_temperature_tmmx_6', 'wind_speed_6', 'SWE_7', 'Flag_7',
# 'air_temperature_tmmn_7', 'potential_evapotranspiration_7',
# 'mean_vapor_pressure_deficit_7', 'relative_humidity_rmax_7',
# 'relative_humidity_rmin_7', 'precipitation_amount_7',
# 'air_temperature_tmmx_7', 'wind_speed_7'],


def check_gridmet(row):
  # check air_temperature_tmmn, precipitation_amount
  date_value = row["date"]
  lat = row["lat"]
  lon = row["lon"]
  # Specify the format of the date string
  date_format = '%Y-%m-%d'

  # Convert the date string to a date object
  date_object = datetime.strptime(date_value, date_format).date()
  yearlist = [date_object.year]
  download_gridmet_of_specific_variables(yearlist)
  
  nc_file = f"{work_dir}/gridmet_climatology/tmmn_{date_object.year}.nc"
  
  dataset = xr.open_dataset(nc_file)
  reference_date = date(1900, 1, 1)

  # Calculate the difference in days
  days_difference = (date_object - reference_date).days
  
  # Calculate the Euclidean distance between the target coordinate and all grid points
  lat_diff = dataset['lat'].values - lat
  lon_diff = dataset['lon'].values - lon
  #distance = np.sqrt(lat_diff**2 + lon_diff**2)

  # Find the indices (i, j) of the grid cell with the minimum distance
  i = np.argmin(np.abs(lat_diff))
  j = np.argmin(np.abs(lon_diff))
  nearest_gridmet_lat = dataset['lat'][i]
  nearest_gridmet_lon = dataset['lon'][j]
  
  selected_data = dataset.sel(day=date_value, lat=nearest_gridmet_lat, lon=nearest_gridmet_lon)
  
  tmmn_check_values = selected_data.air_temperature.values
  
  if str(tmmn_check_values) != str(row["air_temperature_tmmn"]):
    print("Failed sanity check. Gridmet doesn't match")
    exit(1)

def check_elevation(row):
  lat = row["lat"]
  lon = row["lon"]
  
  pass

def check_amsr(row):
  
  pass

def check_snow_cover_area(row):
  pass

def check_passive_microwave(row):
  pass

def check_snotel_cdec(row):
  
  pass


def check_observed_columns():
  # Assuming 'work_dir' is the path to your working directory
  dask_df = dd.read_csv(f"{work_dir}/final_merged_data_3yrs_all_active_stations_v1.csv_sorted.csv")

  # Print the shape and head of the Dask DataFrame
  print(dask_df.shape[0].compute(), 'rows and', dask_df.shape[1].compute(), 'columns')
  print(dask_df.head().compute())

  # Count the number of empty rows in 'swe_value'
  empty_rows_count = dask_df['swe_value'].isnull().sum().compute()
  print(f"Number of empty rows in 'swe_value': {empty_rows_count}")

if __name__ == "__main__":
  # random_sample.apply(check_gridmet, axis=1)
  # random_sample.apply(check_elevation, axis=1)
  # random_sample.apply(check_amsr, axis=1)
  # random_sample.apply(check_snow_cover_area, axis=1)
  # random_sample.apply(check_passive_microwave, axis=1)
#   random_sample.apply(check_snotel_cdec, axis=1)
  check_observed_columns()

  print("If it reaches here, everything is good. The training.csv passed all our sanity cheks!")




