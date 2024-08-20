import os
import subprocess
import threading
from datetime import datetime
from datetime import timedelta
from pyproj import Transformer
from rasterio.enums import Resampling
import numpy as np

import requests
import earthaccess
from osgeo import gdal
from snowcast_utils import work_dir, homedir, test_start_date, date_to_julian
import pandas as pd
import rasterio
import shutil
import time
from convert_results_to_images import plot_all_variables_in_one_csv
import dask
from dask import delayed
import dask.multiprocessing
import pyproj

# change directory before running the code
os.chdir(f"{homedir}/fsca/")



tile_list = ["h08v04", "h08v05", "h09v04", "h09v05", 
             "h10v04", "h10v05", "h11v04", "h11v05", 
             "h12v04", "h12v05", "h13v04", "h13v05", 
             "h15v04", "h16v03", "h16v04", ]
input_folder = os.getcwd() + "/temp/"
output_folder = os.getcwd() + "/output_folder/"
modis_day_wise = os.getcwd() + "/final_output/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(modis_day_wise, exist_ok=True)
western_us_coords = f'{work_dir}/dem_file.tif.csv'
mapper_file = os.path.join(modis_day_wise, f'modis_to_dem_mapper.csv')


@dask.delayed
def convert_hdf_to_geotiff(hdf_file, output_folder):
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)

    target_subdataset_name = "MOD_Grid_Snow_500m:NDSI_Snow_Cover"
    output_file_name = os.path.splitext(os.path.basename(hdf_file))[0] + ".tif"
    output_path = os.path.join(output_folder, output_file_name)

    if os.path.exists(output_path):
        return f"The file {output_path} exists. skip."
    else:
        for subdataset in hdf_ds.GetSubDatasets():
            if target_subdataset_name in subdataset[0]:
                ds = gdal.Open(subdataset[0], gdal.GA_ReadOnly)
                gdal.Translate(output_path, ds)
                ds = None
                break

    hdf_ds = None
    return f"Converted {os.path.basename(hdf_file)} to GeoTIFF"

def convert_all_hdf_in_folder(folder_path, output_folder):
    file_list = []
    delayed_tasks = []

    for file in os.listdir(folder_path):
        output_file_name = os.path.splitext(os.path.basename(file))[0] + ".tif"
        output_path = os.path.join(output_folder, output_file_name)
        if os.path.exists(output_path):
            # print(f"The file {output_path} exists. skip.")
            continue
        else:
            file_list.append(file)
            if file.lower().endswith(".hdf"):
                hdf_file = os.path.join(folder_path, file)
                task = convert_hdf_to_geotiff(hdf_file, output_folder)
                delayed_tasks.append(task)

    results = dask.compute(*delayed_tasks, scheduler="processes")

    return file_list, results

def get_env_var_for_gdalwarp():
    if "PROJ_LIB" in os.environ:
        os.environ.pop("PROJ_LIB")
        print(f"Environment variable PROJ_LIB removed.")
    if "GDAL_DATA" in os.environ:
        os.environ.pop("GDAL_DATA")
        print(f"Environment variable GDAL_DATA removed.")
    return os.environ
  

def merge_tifs(folder_path, target_date, output_file):
  julian_date = date_to_julian(target_date)
  print("target julian date", julian_date)
  tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif') and julian_date in f]
  if len(tif_files) == 0:
    print(f"uh-oh, didn't find HDFs for date {target_date}")
    print("generate a new csv file with empty values for each point")
    gdal_command = ['/usr/bin/gdal_translate', 
                    '-b', '1', 
                    '-outsize', '100%', '100%', 
                    '-scale', '0', '255', '200', '200', 
                    f"{modis_day_wise}/fsca_template.tif", 
                    output_file]
    print("Running ", " ".join(gdal_command))
    subprocess.run(gdal_command, env=get_env_var_for_gdalwarp())
    #raise ValueError(f"uh-oh, didn't find HDFs for date {target_date}")
  else:
    # gdal_command = ['gdal_merge.py', '-o', output_file, '-of', 'GTiff', '-r', 'cubic'] + tif_files
    #if 'PROJ_LIB' in os.environ:
    #    del os.environ['PROJ_LIB']
    print("pyproj.datadir.get_data_dir() = ", pyproj.datadir.get_data_dir())
    gdal_command = ['/usr/bin/gdalwarp', '-r', 'min', ] + tif_files + [f"{output_file}_500m.tif"]
    print("Running ", ' '.join(gdal_command))
    subprocess.run(gdal_command, env=get_env_var_for_gdalwarp())
    # gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -tr 0.036 0.036  -cutline template.shp -crop_to_cutline -overwrite output_4km.tif output_4km_clipped.tif
    gdal_command = ['/usr/bin/gdalwarp', '-t_srs', 'EPSG:4326', '-tr', '0.036', '0.036', '-cutline', f'{work_dir}/template.shp', '-crop_to_cutline', '-overwrite', f"{output_file}_500m.tif", output_file]
    print("Running ", " ".join(gdal_command))
    subprocess.run(gdal_command, env=get_env_var_for_gdalwarp())


def list_files(directory):
  return [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if
          os.path.isfile(os.path.join(directory, f))]


def merge_tiles(date, hdf_files):
  path = f"data/{date}/"
  files = list_files(path)
  print(files)
  merged_filename = f"data/{date}/merged.tif"
  merge_command = ["/usr/bin/gdal_merge.py", "-o", merged_filename, "-of", "GTiff"] + files
  try:
    subprocess.run(merge_command, env=get_env_var_for_gdalwarp())
    print(f"Merged tiles into {merged_filename}")
  except subprocess.CalledProcessError as e:
    print(f"Error merging tiles: {e}")


def download_url(date, url):
  file_name = url.split('/')[-1]
  if os.path.exists(f'data/{date}/{file_name}'):
    print(f'File: {file_name} already exists, SKIPPING')
    return
  try:
    os.makedirs('data/', exist_ok=True)
    os.makedirs(f'data/{date}', exist_ok=True)
    response = requests.get(url, stream=True)
    with open(f'data/{date}/{file_name}', 'wb') as f:
      for chunk in response.iter_content(chunk_size=8192):
        if chunk:
          f.write(chunk)

    print(f"Downloaded {file_name}")
  except Exception as e:
    print(f"Error downloading {url}: {e}")


def download_all(date, urls):
  threads = []

  for url in urls:
    thread = threading.Thread(target=download_url, args=(date, url,))
    thread.start()
    threads.append(thread)

  for thread in threads:
    thread.join()


def delete_files_in_folder(folder_path):
  if not os.path.exists(folder_path):
    print("Folder does not exist.")
    return

  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      else:
        print(f"Skipping {filename}, as it is not a file.")
    except Exception as e:
      print(f"Failed to delete {file_path}. Reason: {e}")


def download_tiles_and_merge(start_date, end_date):
  date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
  for i in date_list:
    current_date = i.strftime("%Y-%m-%d")
    target_output_tif = f'{modis_day_wise}/{current_date}__snow_cover.tif'
    
    if os.path.exists(target_output_tif):
        file_size_bytes = os.path.getsize(target_output_tif)
        print(f"file_size_bytes: {file_size_bytes}")
        print(f"The file {target_output_tif} exists. skip.")
    else:
        print(f"The file {target_output_tif} does not exist.")
        print("start to download files from NASA server to local")
        earthaccess.login(strategy="netrc")
        results = earthaccess.search_data(short_name="MOD10A1", 
                                          cloud_hosted=False, 
                                          bounding_box=(-124.77, 24.52, -66.95, 49.38),
                                          temporal=(current_date, current_date))
        earthaccess.download(results, input_folder)
        print("done with downloading, start to convert HDF to geotiff..")

        convert_all_hdf_in_folder(input_folder, output_folder)
        print("done with conversion, start to merge geotiff tiles to one tif per day..")

        merge_tifs(folder_path=output_folder, target_date = current_date, output_file=target_output_tif)
        print(f"saved the merged tifs to {target_output_tif}")
    #delete_files_in_folder(input_folder)  # cleanup
    #delete_files_in_folder(output_folder)  # cleanup

def get_value_at_coords(src, lat, lon, band_number=1):
#     transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
#     east, north = transformer.transform(lon, lat)
    if not (src.bounds.left <= lat <= src.bounds.right and src.bounds.bottom <= lon <= src.bounds.top):
      return None
    row, col = src.index(lat, lon)
    if (0 <= row < src.height) and (0 <= col < src.width):
      return src.read(band_number, window=((row, row+1), (col, col+1)), resampling=Resampling.nearest)[0, 0]
    else:
      return None

def get_band_value(row, src):
    #row, col = src.index(row["Latitude"], row["Longitude"])
    #print(row, col, src.height, src.width)
    if (row["modis_y"] < src.height) and (row["modis_x"] < src.width):
      valid_value =  src.read(1, 
                              window=((row["modis_y"], 
                                       row["modis_y"]+1), 
                                      (row["modis_x"], 
                                       row["modis_x"]+1))
                             )
      #print("extracted value array: ", valid_value)
      #print("Found a valid value: ",row, valid_value, src.height, src.width)
      return valid_value[0,0]
    else:
      return None
          
def process_file(file_path, current_date):
    """
    Process a raster file, extract values for specific coordinates, and save the result in a CSV file.

    Parameters:
    - file_path (str): Path to the raster file to be processed.
    - current_date (str): Current date to be associated with the processed data.

    Returns:
    - str: Path to the saved CSV file containing the processed data.
    """

    # Read the station DataFrame from a mapper file (assuming `mapper_file` is defined elsewhere)
    station_df = pd.read_csv(mapper_file)
    print(f"Opening {file_path}")

    # Open the raster file using rasterio
    with rasterio.open(file_path) as src:
        # Apply get_band_value for each row in the DataFrame
        station_df['fsca'] = station_df.apply(get_band_value, axis=1, args=(src,))

    # Filter out None values
    valid_data = station_df[station_df['fsca'].notna()]

    # Prepare final data
    valid_data['date'] = current_date
    output_file = os.path.join(modis_day_wise, f'{current_date}_output.csv')
    print(f"Saving CSV file: {output_file}")
    valid_data.to_csv(output_file, index=False, columns=['date', 'Latitude', 'Longitude', 'fsca'])
    
    return output_file


def merge_cumulative_csv(start_date, end_date, force):
  
  current_date = start_date
  target_date = end_date
  
  input_time_series_file = f"{modis_day_wise}/{end_date.strftime('%Y-%m-%d')}_output_with_time_series.csv"
  target_cumulative_file = f"{modis_day_wise}/{end_date.strftime('%Y-%m-%d')}_output.csv_cumulative.csv"
  
  if os.path.exists(target_cumulative_file) and not force:
    print(f"file already exists {target_cumulative_file}")
    return
  
  df = pd.read_csv(input_time_series_file)

  # add all the columns together and save to new csv
  # Adding all columns except latitude and longitude
  df = df.apply(pd.to_numeric, errors='coerce')

  #new_df = new_df.head(2000)

  fsca_cols = [col for col in df.columns if col.startswith('fsca')]
  print("fsca_cols are: ", fsca_cols)
  
  df['cumulative_fsca'] = df[fsca_cols].sum(axis=1)

  df = df.loc[:, ['Latitude', 'Longitude', f"fsca", 'cumulative_fsca']]
  df["date"] = end_date

  print("new_df final shape: ", df.head())
  df.to_csv(target_cumulative_file, index=False)
  print(f"new df is saved to {target_cumulative_file}")

  print(df['cumulative_fsca'].describe(include='all'))

def add_cumulative_column(df, column_name):
  df[f'cumulative_{column_name}'] = df[column_name].cumsum()
  return df

# Function to perform polynomial interpolation and fill in missing values
def interpolate_missing_and_add_cumulative_inplace(row, column_name, degree=1):
  """
  Interpolate missing values in a Pandas Series using polynomial interpolation
  and add a cumulative column.

  Parameters:
    - row (pd.Series): The input row containing the data to be interpolated.
    - column_name (str): The name of the column to be interpolated.
    - degree (int, optional): The degree of the polynomial fit. Default is 1 (linear).

  Returns:
    - pd.Series: The row with interpolated values and a cumulative column.

  Raises:
    - ValueError: If there are unexpected null values after interpolation.

  Note:
    - For 'SWE' column, values above 240 are treated as gaps and set to 240.
    - For 'fsca' column, values above 100 are treated as gaps and set to 100.

  Examples:
    ```python
    # Example usage:
    interpolated_row = interpolate_missing_and_add_cumulative_inplace(my_row, 'fsca', degree=2)
    ```

  """
  
  # Extract X series (column names)
  x_key = row.index
  x = np.arange(len(x_key))

  # Extract Y series (values from the first row)
  y = row
  

  # Create a mask for missing values
  if column_name == "SWE":
    mask = (y > 240) | y.isnull()
  elif column_name == "fsca":
    y = y.replace([225, 237, 239], 0)
    y[y < 0] = 0
    mask = (y > 100) | y.isnull()
  else:
    mask = y.isnull()

  # Check if all elements in the mask array are True
  all_true = np.all(mask)

  if all_true or len(np.where(~mask)[0]) == 1:
    row.values[:] = 0
  else:
    # Perform interpolation
    #new_y = np.interp(x, x[~mask], y[~mask])
    # Replace missing values with interpolated values
    #df[column_name] = new_y
    
    try:
      # Coefficients of the polynomial fit
      coefficients = np.polyfit(x[~mask], y[~mask], deg=degree)

      # Perform polynomial interpolation
      interpolated_values = np.polyval(coefficients, x)

      # Merge using np.where
      merged_array = np.where(mask, interpolated_values, y)

      row = pd.Series(merged_array, index=x_key)
    except Exception as e:
      # Print the error message and traceback
      import traceback
      traceback.print_exc()
      print("x:", x)
      print("y:", y)
      print("mask:", mask)
      print(f"Error: {e}")
      raise e
      
    if column_name == "AMSR_SWE":
      row = row.clip(upper=240, lower=0)
    elif column_name == "fsca":
      row = row.clip(upper=100, lower=0)

    if row.isnull().any():
      print("x:", x)
      print("y:", y)
      print("mask:", mask)
      print("why row still has values > 100", row)
      raise ValueError("Single group: shouldn't have null values here")

  
  # create the cumulative column after interpolation
  row[f"cumulative_{column_name}"] = row.sum()
  return row


def add_time_series_columns(start_date, end_date, force=True):
  """
  Converts data from a cleaned CSV file into a time series format.

  This function reads a cleaned CSV file, sorts the data, fills in missing values using polynomial interpolation,
  and creates a time series dataset for specific columns. The resulting time series data is saved to a new CSV file.

  Parameters:
    - start_date (str): The start date of the time series data in the format 'YYYY-MM-DD'.
    - end_date (str): The end date of the time series data in the format 'YYYY-MM-DD'.
    - force (bool, optional): If True, forces the regeneration of time series data even if the output file exists.

    Returns:
    - None: The function primarily generates time series data and saves it to a CSV file with side effects.

  Example:
    ```python
    add_time_series_columns('2022-01-01', '2022-12-31', force=True)
    ```

  """
  current_date = start_date
  target_date = end_date
  target_date_str = target_date.strftime('%Y-%m-%d')
  
  output_csv = f"{modis_day_wise}/{target_date_str}_output_with_time_series.csv"
  print(f"add_time_series_columns target csv: {output_csv}")
        
  if os.path.exists(output_csv) and not force:
    print(f"{output_csv} already exists, skipping..")
    return
  
  backup_time_series_csv_path = f"{modis_day_wise}/{target_date_str}_output_with_time_series_backup.csv"
  
  columns_to_be_time_series = ['fsca']

  # Read the all the column merged CSV
  
  date_keyed_objects = {}
  data_dict = {}
  new_df = None
  while current_date <= end_date:
    
    current_date_str = current_date.strftime('%Y-%m-%d')
  
    data_dict[current_date_str] = f"{modis_day_wise}/{current_date_str}_output.csv"
    #print(data_dict[current_date_str])
    current_df = pd.read_csv(data_dict[current_date_str])
    current_df.drop(["date"], axis=1, inplace=True)
    
    if current_date != end_date:
      current_df.rename(columns={"fsca": f"fsca_{current_date_str}"}, inplace=True)
    #print(current_df.head())
    
    if new_df is None:
      new_df = current_df
    else:
      new_df = pd.merge(new_df, current_df, on=['Latitude', 'Longitude'])
      #new_df = new_df.append(current_df, ignore_index=True)

    current_date += timedelta(days=1)

  print("new_df.columns = ", new_df.columns)
  print(new_df.head())
  
  df = new_df
        
  gap_filled_csv = f"{output_csv}_gap_filled.csv"
  if os.path.exists(gap_filled_csv) and not force:
    print(f"{gap_filled_csv} already exists, skipping..")
    df = pd.read_csv(gap_filled_csv)
    print(df["fsca"].describe())
  else:
  
    #df.sort_values(by=['Latitude', 'Longitude'], inplace=True)
    print("All current columns: ", df.columns)
    
    
    print("Start to fill in the missing values")
    print("all the df shape: ", df.shape)
    #grouped = df.groupby(['Latitude', 'Longitude'])
    #num_groups = len(grouped.groups)
    #print(f"Number of groups: {num_groups}")
    filled_data = pd.DataFrame()
    
    num_days = 7
  
    
    # Apply the function to each group
#     no_loc_df = df.drop(["Latitude", "Longitude"], axis=1)
#     filled_data = no_loc_df.apply(lambda row: process_group_value_filling(row, num_days, target_date_str ), axis=1)
#     filled_data["Latitude"] = df["Latitude"]
#     filled_data["Longitude"] = df["Longitude"]
    
    filtered_columns = df.filter(like="fsca")
    print(filtered_columns.columns)
    filtered_columns = filtered_columns.mask(filtered_columns > 100)
    filtered_columns.interpolate(axis=1, method='linear', inplace=True)
    filtered_columns.fillna(0, inplace=True)

    sum_column = filtered_columns.sum(axis=1)
    # Define a specific name for the new column
    df[f'cumulative_fsca'] = sum_column
    df[filtered_columns.columns] = filtered_columns

    if filtered_columns.isnull().any().any():
      print("filtered_columns :", filtered_columns)
      raise ValueError("Single group: shouldn't have null values here")
    
    filled_data = df
    filled_data["date"] = target_date_str
    #filled_data.fillna(0, inplace=True)
  
    if any(filled_data['fsca'] > 100):
      raise ValueError("Error: shouldn't have fsca > 100 at this point")

    print("Finished correctly ", filled_data.head())
    filled_data.to_csv(gap_filled_csv, index=False)
    print(f"New filled values csv is saved to {output_csv}_gap_filled.csv")
    df = filled_data
  
  result = df
  print(result['date'].unique())
  print(result.shape)
  print(result["fsca"].describe())
  result.to_csv(output_csv, index=False)
  print(f"New data is saved to {output_csv}")
  shutil.copy(output_csv, backup_time_series_csv_path)
  print(f"File is backed up to {backup_time_series_csv_path}")
  cumulative_file_path =  f"{modis_day_wise}/{test_start_date}_output.csv_cumulative.csv"
  shutil.copy(output_csv, cumulative_file_path)
  
#   input_time_series_file = f"{modis_day_wise}/{test_start_date}_output_with_time_series.csv_gap_filled.csv"
#   target_cumulative_file = f"{modis_day_wise}/{test_start_date}_output.csv_cumulative.csv"
#   shutil.copy(input_time_series_file, target_cumulative_file)
  


def map_modis_to_station(row, src):
#   transformer = Transformer.from_crs("EPSG:4326", 
#                                      src.crs, 
#                                      always_xy=True)
#   east, north = transformer.transform(row["Longitude"], 
#                                       row["Latitude"])
  drow, dcol = src.index(row["Longitude"], row["Latitude"])
  return drow, dcol
  
  
def prepare_modis_grid_mapper():
  """
  Prepares a mapper file to map coordinates between station coordinates and MODIS grid coordinates.

    This function performs the following steps:
    1. Checks if the mapper file already exists. If yes, the function skips the generation process.
    2. Reads station coordinates from a CSV file (`western_us_coords`) containing 'Longitude' and 'Latitude'.
    3. Uses a sample MODIS TIFF file (`sample_modis_tif`) to map MODIS grid coordinates ('modis_x' and 'modis_y') to station coordinates.
    4. Saves the resulting mapper file as a CSV (`mapper_file`) containing columns 'Latitude', 'Longitude', 'modis_x', and 'modis_y'.

    Note: Ensure that necessary functions like `map_modis_to_station` are defined and available in the same scope.

    Returns:
    - None: The function primarily generates the mapper file with side effects.

    Example:
    ```python
    prepare_modis_grid_mapper()
    ```

  """
  # actually, not sure this applied for modis. The tile HDF must be exactly same extent to make this work. Otherwise, the mapper won't get usable. 
  
  if os.path.exists(mapper_file):
    print(f"The file {mapper_file} exists. skip.")
  else:
    print(f"start to generate {mapper_file}")
    station_df = pd.read_csv(western_us_coords, low_memory=False, usecols=['Longitude', 'Latitude'])

    sample_modis_tif = f"{modis_day_wise}/2022-10-01__snow_cover.tif"

    with rasterio.open(sample_modis_tif) as src:
      # Apply get_band_value for each row in the DataFrame
      station_df['modis_y'], station_df['modis_x'] = zip(*station_df.apply(map_modis_to_station, axis=1, args=(src,)))


      print(f"Saving mapper csv file: {mapper_file}")
      station_df.to_csv(mapper_file, index=False, columns=['Latitude', 'Longitude', 'modis_x', 'modis_y'])
    
def extract_data_for_testing():
  """
    Extracts and processes MODIS data for testing purposes within a specified date range.

    This function performs the following steps:
    1. Determines the start and end dates based on the `test_start_date`.
    2. Prepares the MODIS grid mapper using `prepare_modis_grid_mapper`.
    3. Downloads, tiles, and merges MODIS data between the determined start and end dates using `download_tiles_and_merge`.
    4. Iterates through each day in the date range, extracting data and saving it as day-wise CSV files using `process_file`.
    5. Adds time series columns to the extracted data using `add_time_series_columns`.
    6. Creates a cumulative CSV file by copying the output file with time series information.

    Note: Ensure that necessary functions like `prepare_modis_grid_mapper`, `download_tiles_and_merge`, 
    `process_file`, and `add_time_series_columns` are defined and available in the same scope.

    Returns:
    - None: The function primarily performs data extraction and processing with side effects.

    Example:
    ```python
    extract_data_for_testing()
    ```

  """
  print("get test_start_date = ", test_start_date)
  end_date = datetime.strptime(test_start_date, "%Y-%m-%d")
  print(end_date)
  if end_date.month < 10:
    past_october_1 = datetime(end_date.year - 1, 10, 1)
  else:
    past_october_1 = datetime(end_date.year, 10, 1)
  
  start_date = past_october_1
  print(f"The start_date of the water year {start_date}")
  
  prepare_modis_grid_mapper()
  
  download_tiles_and_merge(start_date, end_date)
  
  date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
  for i in date_list:
    current_date = i.strftime("%Y-%m-%d")
    print(f"extracting data for {current_date}")
    outfile = os.path.join(modis_day_wise, f'{current_date}_output.csv')
    if os.path.exists(outfile):
      print(f"The file {outfile} exists. skip.")
    else:
      process_file(f'{modis_day_wise}/{current_date}__snow_cover.tif', current_date)
  
  add_time_series_columns(start_date, end_date, force=True)
  
  

if __name__ == "__main__":
  extract_data_for_testing()

  # cumulative_file_path =  f"{modis_day_wise}/{test_start_date}_output.csv_cumulative.csv"
  # plot_all_variables_in_one_csv(cumulative_file_path, f"{cumulative_file_path}.png")

  # SnowCover is missing from 10-12 to 10-23
  #download_tiles_and_merge(datetime.strptime("2022-10-24", "%Y-%m-%d"), datetime.strptime("2022-10-24", "%Y-%m-%d"))


