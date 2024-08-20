import os
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
import concurrent.futures
from snowcast_utils import homedir, work_dir, train_start_date, train_end_date
from datetime import datetime, timedelta
import dask.dataframe as dd
import numpy as np

working_dir = f"{homedir}/fsca"
folder_path = f"{working_dir}/final_output/"
new_base_station_list_file = f"{work_dir}/all_snotel_cdec_stations_active_in_westus.csv"
cell_to_modis_mapping = f"{working_dir}/training_cell_to_modis_mapper_original_snotel_stations.csv"
non_station_random_points_file = f"{work_dir}/non_station_random_points_in_westus.csv"
only_active_ghcd_station_in_west_conus_file = f"{working_dir}/active_ghcnd_station_only_list.csv"
ghcd_station_to_modis_mapper_file = f"{working_dir}/active_ghcnd_mapper_modis.csv"
all_training_points_with_snotel_ghcnd_file = f"{work_dir}/all_training_points_snotel_ghcnd_in_westus.csv"
modis_day_wise = f"{working_dir}/final_output/"
os.makedirs(modis_day_wise, exist_ok=True)


def map_modis_to_station(row, src):
  drow, dcol = src.index(row["lon"], row["lat"])
  return drow, dcol


def generate_random_non_station_points():
  # Load the GeoTIFF file
  sample_modis_tif = f"{modis_day_wise}/2022-10-01__snow_cover.tif"
  print(f"loading geotiff {sample_modis_tif}")
  with rasterio.open(sample_modis_tif) as src:
    # Get the raster metadata
    bounds = src.bounds
    transform = src.transform
    width = src.width
    height = src.height

    # Read the raster values as a numpy array
    raster_array = src.read(1)  # Assuming it's a single-band raster

    # Generate random points
    random_points = []
    while len(random_points) < 4000:
      # Generate random coordinates within the bounds of the raster
      random_x = np.random.uniform(bounds.left, bounds.right)
      random_y = np.random.uniform(bounds.bottom, bounds.top)

      # Convert random coordinates to pixel coordinates
      col, row = ~transform * (random_x, random_y)

      # Ensure the generated pixel coordinates are within the raster bounds
      if 0 <= row < height and 0 <= col < width:
        # Get the value at the generated pixel coordinates
        value = raster_array[int(row), int(col)]

        # Check if the value is not 239
        if value != 239 and value != 255:
          # Append the coordinates to the list
          random_points.append((random_x, random_y, col, row))

    # Assuming random_points is a list of tuples where each tuple contains latitude and longitude
    random_points = [(lat, lon, col, row) for lon, lat, col, row in random_points]  # Swap the order to (latitude, longitude)

    # Create a DataFrame from the random_points list
    random_points_df = pd.DataFrame(random_points, columns=['latitude', 'longitude', 'modis_x', 'modis_y'])

    # Save the DataFrame to a CSV file
    random_points_df.to_csv(non_station_random_points_file, index=False)
    print(f"random points are saved to {non_station_random_points_file}")

    


def prepare_modis_grid_mapper_training():
  # actually, not sure this applied for modis. The tile HDF must be exactly same extent to make this work. Otherwise, the mapper won't get usable. 
  
  if os.path.exists(cell_to_modis_mapping):
    print(f"The file {cell_to_modis_mapping} exists. skip.")
  else:
    print(f"start to generate {cell_to_modis_mapping}")
    station_df = pd.read_csv(new_base_station_list_file)
    print("original station_df describe() = ", station_df.describe())

    sample_modis_tif = f"{modis_day_wise}/2022-10-01__snow_cover.tif"

    with rasterio.open(sample_modis_tif) as src:
      # Apply get_band_value for each row in the DataFrame
      #station_df['modis_y'], station_df['modis_x'] = zip(*station_df.apply(map_modis_to_station, axis=1, args=(src,)))
      print("Spatial Extent (Bounding Box):", src.bounds)
      # Get the affine transformation matrix
      transform = src.transform

      # Extract the spatial extent using the affine transformation
      left, bottom, right, top = rasterio.transform.array_bounds(src.height, src.width, transform)

      # Print the spatial extent
      print("Spatial Extent (Bounding Box):", (left, bottom, right, top))
      
      station_df['modis_y'], station_df['modis_x'] = rasterio.transform.rowcol(
        src.transform, 
        station_df["longitude"], 
        station_df["latitude"])
      
      # print(f"Saving mapper csv file: {cell_to_modis_mapping}")
      station_df.to_csv(cell_to_modis_mapping, index=False, columns=['latitude', 'longitude', 'modis_x', 'modis_y'])
      
      print("after mapped modis station_df.describe() = ", station_df.describe())

def merge_station_and_non_station_to_one_csv():
  print(f"new_base_station_list_file = {new_base_station_list_file}")
  print(f"cell_to_modis_mapping = {cell_to_modis_mapping}")
  print(f"non_station_random_points_file = {non_station_random_points_file}")
  df1 = pd.read_csv(cell_to_modis_mapping)
  df2 = pd.read_csv(non_station_random_points_file)
  combined_df = pd.concat([df1, df2], ignore_index=True)
  combined_df.to_csv(all_training_points_with_station_and_non_station_file, index=False)

  print(f"Combined CSV saved to {all_training_points_with_station_and_non_station_file}")

def merge_snotel_ghcnd_station_to_one_csv():
  print(f"new_base_station_list_file = {new_base_station_list_file}")
  print(f"cell_to_modis_mapping = {cell_to_modis_mapping}")
  print(f"ghcnd_to_modis_mapping = {ghcd_station_to_modis_mapper_file}")
  df1 = pd.read_csv(cell_to_modis_mapping)
  df2 = pd.read_csv(ghcd_station_to_modis_mapper_file)
  combined_df = pd.concat([df1, df2], ignore_index=True)
  combined_df.to_csv(all_training_points_with_snotel_ghcnd_file, index=False)

  print(f"Combined CSV saved to {all_training_points_with_snotel_ghcnd_file}")

def prepare_ghcnd_station_mapping_training():
  if os.path.exists(ghcd_station_to_modis_mapper_file):
    print(f"The file {ghcd_station_to_modis_mapper_file} exists. skip.")
  else:
    print(f"start to generate {ghcd_station_to_modis_mapper_file}")
    station_df = pd.read_csv(only_active_ghcd_station_in_west_conus_file)
    station_df = station_df.rename(columns={'Latitude': 'latitude', 
                                            'Longitude': 'longitude'})
    print("original station_df describe() = ", station_df.describe())

    sample_modis_tif = f"{modis_day_wise}/2022-10-01__snow_cover.tif"

    with rasterio.open(sample_modis_tif) as src:
      # Apply get_band_value for each row in the DataFrame
      #station_df['modis_y'], station_df['modis_x'] = zip(*station_df.apply(map_modis_to_station, axis=1, args=(src,)))
      print("Spatial Extent (Bounding Box):", src.bounds)
      # Get the affine transformation matrix
      transform = src.transform

      # Extract the spatial extent using the affine transformation
      left, bottom, right, top = rasterio.transform.array_bounds(src.height, src.width, transform)

      # Print the spatial extent
      print("Spatial Extent (Bounding Box):", (left, bottom, right, top))
      
      station_df['modis_y'], station_df['modis_x'] = rasterio.transform.rowcol(
        src.transform, 
        station_df["longitude"],
        station_df["latitude"])
      
      # print(f"Saving mapper csv file: {cell_to_modis_mapping}")
      station_df.to_csv(ghcd_station_to_modis_mapper_file, index=False, columns=['latitude', 'longitude', 'modis_x', 'modis_y'])
      print(f"the new mapper to the ghcnd is saved to {ghcd_station_to_modis_mapper_file}")
      print("after mapped modis station_df.describe() = ", station_df.describe())
  
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
  if (row["modis_y"] < src.height) and (row["modis_x"] < src.width):
    # print("src.height = ", src.height, " - ", row["modis_y"])
    # print("src.width = ", src.width, " - ", row["modis_x"])
    # print(row)
    valid_value =  src.read(1, 
                            window=((int(row["modis_y"]),
                                     int(row["modis_y"])+1), 
                                    (int(row["modis_x"]),
                                     int(row["modis_x"])+1)))
    # print("valid_value[0,0] = ", valid_value[0,0])
    return valid_value[0,0]
  else:
    return None
          
def process_file(file_path, current_date_str, outfile):
  print(f"processing {file_path}")
  station_df = pd.read_csv(all_training_points_with_snotel_ghcnd_file)
  # print("station_df.head() = ", station_df.head())

  # Apply get_band_value for each row in the DataFrame
  with rasterio.open(file_path) as src:
    # Apply get_band_value for each row in the DataFrame
    # Get the affine transformation matrix
    transform = src.transform

    # Extract the spatial extent using the affine transformation
    left, bottom, right, top = rasterio.transform.array_bounds(src.height, src.width, transform)

    # Print the spatial extent
    # print("Spatial Extent (Bounding Box):", (left, bottom, right, top))

    station_df['fsca'] = station_df.apply(get_band_value, axis=1, args=(src,))

    
  # Prepare final data
  station_df['date'] = current_date_str
  station_df.to_csv(outfile, index=False, 
                    columns=['date', 'latitude', 'longitude', 'fsca'])
  print(f"Saved to csv: {outfile}")

def merge_csv(start_date, end_date):
  import glob
  # Find CSV files within the specified date range
  csv_files = glob.glob(folder_path + '*_training_output_station_corrected.csv')
  relevant_csv_files = []

  for c in csv_files:
    # Extract the date from the file name
    # print("c = ", c)
    file_name = os.path.basename(c)
    date_str = file_name.split('_')[0]  # Assuming the date is part of the file name
    # print("date_str = ", date_str)
    file_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Check if the file date is within the specified range
    if start_date <= file_date <= end_date:
      relevant_csv_files.append(c)
#       # Read and concatenate only relevant CSV files
#       df = []
#       for c in relevant_csv_files:
#         tmp = pd.read_csv(c, low_memory=False, usecols=['date', 'latitude', 'longitude', 'fsca'])
#         df.append(tmp)

#         combined_df = pd.concat(df, ignore_index=True)

  # Initialize a Dask DataFrame
  print("start to use dask to read all csv files")
  dask_df = dd.read_csv(relevant_csv_files)

  # Save the merged DataFrame to a CSV file
  output_file = f'{working_dir}/fsca_final_training_all.csv'
  # Write the Dask DataFrame to a single CSV file
  print(f"saving all csvs into one file: {output_file}")
  dask_df.to_csv(output_file, index=False, single_file=True)
  #combined_df.to_csv(output_file, index=False)

  #print(combined_df.describe())
  print(f"Merged data saved to {output_file}")
  
def main():
  
  start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
  
  end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
  
  prepare_modis_grid_mapper_training()
  prepare_ghcnd_station_mapping_training()
  # running this function will generate a new set of random points
  # generate_random_non_station_points()
  #merge_station_and_non_station_to_one_csv()
  merge_snotel_ghcnd_station_to_one_csv()
  
  date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
  for i in date_list:
    current_date = i.strftime("%Y-%m-%d")
    #print(f"extracting data for {current_date}")
    outfile = os.path.join(modis_day_wise, f'{current_date}_training_output_station_with_ghcnd.csv')
    if os.path.exists(outfile):
      print(f"The file {outfile} exists. skip.")
      pass
    else:
      process_file(f'{modis_day_wise}/{current_date}__snow_cover.tif', current_date, outfile)
  
  merge_csv(start_date, end_date)

if __name__ == "__main__":
  main()
  print("fsca Data extraction complete.")
  

