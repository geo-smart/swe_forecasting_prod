import distutils.dir_util
from snowcast_utils import work_dir
import os
import shutil
import re
import pandas as pd
from datetime import datetime
import time


print("move the plots and the results into the http folder")

def copy_if_modified(source_file, destination_file):
    if os.path.exists(destination_file):
        source_modified_time = os.path.getmtime(source_file)
        dest_modified_time = os.path.getmtime(destination_file)
        
        # If the source file is modified after the destination file
        if source_modified_time > dest_modified_time:
            shutil.copy(source_file, destination_file)
            print(f'Copied: {source_file}')
    else:
        shutil.copy(source_file, destination_file)
        print(f'Copied: {source_file}')

def create_mapserver_map_config(target_geotiff_file_path, force=False):
  geotiff_file_name = os.path.basename(target_geotiff_file_path)
  geotiff_mapserver_file_path = f"/var/www/html/swe_forecasting/map/{geotiff_file_name}.map"
  
  if os.path.exists(geotiff_mapserver_file_path) and not force:
    print(f"{geotiff_mapserver_file_path} already exists")
    return geotiff_mapserver_file_path
  
  # Define a regular expression pattern to match the date in the filename
  pattern = r"\d{4}-\d{2}-\d{2}"

  # Use re.search to find the match
  match = re.search(pattern, geotiff_file_name)

  # Check if a match is found
  if match:
      date_string = match.group()
      print("Date:", date_string)
  else:
      print("No date found in the filename.")
      return f"The file's name {target_geotiff_file} is wrong"
  
  mapserver_config_content = f"""
MAP
  NAME "swemap"
  STATUS ON
  EXTENT -125 25 -100 49
  SIZE 800 400
  UNITS DD
  SHAPEPATH "/var/www/html/swe_forecasting/output/"

  PROJECTION
    "init=epsg:4326"
  END

  WEB
    IMAGEPATH "/temp/"
    IMAGEURL "/temp/"
    METADATA
      "wms_title" "SWE MapServer WMS"
      "wms_onlineresource" "http://geobrain.csiss.gmu.edu/cgi-bin/mapserv?map=/var/www/html/swe_forecasting/output/swe.map&"
      WMS_ENABLE_REQUEST      "*"
      WCS_ENABLE_REQUEST      "*"
      "wms_srs" "epsg:5070 epsg:4326 epsg:3857"
    END
  END


  LAYER
    NAME "predicted_swe_{date_string}"
    TYPE RASTER
    STATUS DEFAULT
    DATA "{target_geotiff_file_path}"

    PROJECTION
      "init=epsg:4326"
    END

    METADATA
      "wms_include_items" "all"
    END
    PROCESSING "NODATA=0"
    STATUS ON
    DUMP TRUE
    TYPE RASTER
    OFFSITE 0 0 0
    CLASSITEM "[pixel]"
    TEMPLATE "template.html"
    INCLUDE "legend_swe.map"
  END
END
"""
  
  with open(geotiff_mapserver_file_path, "w") as file:
    file.write(mapserver_config_content)
    
  print(f"Mapserver config is created at {geotiff_mapserver_file_path}")
  return geotiff_mapserver_file_path

def refresh_available_date_list():
  
  # Define columns for the DataFrame
  columns = ["date", "predicted_swe_url_prefix"]

  # Create an empty DataFrame with columns
  df = pd.DataFrame(columns=columns)
  
  for filename in os.listdir(geotiff_destination_folder):
    target_geotiff_file = os.path.join(geotiff_destination_folder, filename)
    
    date_str = re.search(r"\d{4}-\d{2}-\d{2}", filename).group()
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Append a new row to the DataFrame
    df = df.append({
      "date": date, 
      "predicted_swe_url_prefix": f"../swe_forecasting/output/{filename}"
    }, ignore_index=True)
  
  # Save DataFrame to a CSV file
  df.to_csv("/var/www/html/swe_forecasting/date_list.csv", index=False)
  print("directly write into the server file which might be used at the time might not be a good idea. ")

  # Display the final DataFrame
  print(df)
  

               
def copy_files_to_right_folder():
  
  # copy the variable comparison folder
  source_folder = f"{work_dir}/var_comparison/"
  figure_destination_folder = f"/var/www/html/swe_forecasting/plots/"
  geotiff_destination_folder = f"/var/www/html/swe_forecasting/output/"

  # Copy the folder with overwriting existing files/folders
  distutils.dir_util.copy_tree(source_folder, figure_destination_folder, update=1)

  print(f"Folder '{source_folder}' copied to '{figure_destination_folder}' with overwriting.")


  # copy the png from testing_output to plots
  source_folder = f"{work_dir}/testing_output/"

  # Ensure the destination folder exists, create it if necessary
  if not os.path.exists(figure_destination_folder):
    os.makedirs(figure_destination_folder)
    
  if not os.path.exists(geotiff_destination_folder):
    os.makedirs(geotiff_destination_folder)

  # Loop through the files in the source folder
  for filename in os.listdir(source_folder):
    # Check if the file is a PNG file
    if filename.endswith('.png') or filename.endswith('.tif'):
      # Build the source and destination file paths
      source_file = os.path.join(source_folder, filename)
      destination_file = os.path.join(figure_destination_folder, filename)

      # Copy the file from the source to the destination
      copy_if_modified(source_file, destination_file)
      
      # Copy the file to the output folder if it is geotif
      if filename.endswith('.tif'):
        output_dest_file = os.path.join(geotiff_destination_folder, filename)
        copy_if_modified(source_file, output_dest_file)
        
  

if __name__ == "__main__":
  
  geotiff_destination_folder = f"/var/www/html/swe_forecasting/output/"
  copy_files_to_right_folder()
  
  # create mapserver config for all geotiff files in output folder
  for filename in os.listdir(geotiff_destination_folder):
    destination_file = os.path.join(geotiff_destination_folder, filename)
    create_mapserver_map_config(destination_file, force=True)
  print("Finished creation of all mapserver files.")
    
  # refresh the output file list for the website to refresh its calendar
  refresh_available_date_list()
  print("All done")
  time.sleep(10)

