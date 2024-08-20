import os
import subprocess
import threading
from datetime import datetime, timedelta

import requests
import earthaccess
from osgeo import gdal
from snowcast_utils import date_to_julian, work_dir

# change directory before running the code
os.chdir("/home/chetana/fsca/")

# The start date for downloading and processing MODIS tiles.
start_date = datetime(2018, 1, 1)

# The end date for downloading and processing MODIS tiles.
end_date = datetime(2022, 12, 31)

# A list of MODIS tiles (in horizontal and vertical format) to download and process.
tile_list = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04", "h10v05", 
             "h11v04", "h11v05", "h12v04", "h12v05", "h13v04", "h13v05", 
             "h15v04", "h16v03", "h16v04"]

# The folder path where the HDF files will be temporarily stored.
input_folder = os.getcwd() + "/temp/"

# The folder path where the GeoTIFF files will be stored after conversion from HDF.
output_folder = os.getcwd() + "/output_folder/"

# The folder path where the final merged output GeoTIFF files will be stored.
modis_day_wise = os.getcwd() + "/final_output/"

# Create necessary directories if they do not exist.
os.makedirs(output_folder, exist_ok=True)
os.makedirs(modis_day_wise, exist_ok=True)

def convert_hdf_to_geotiff(hdf_file, output_folder):
    """
    Converts a specified HDF file to a GeoTIFF format.

    Args:
        hdf_file (str): The file path of the HDF file to be converted.
        output_folder (str): The directory where the converted GeoTIFF file will be saved.

    Returns:
        None
    """
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)

    # Specific subdataset name you're interested in
    target_subdataset_name = "MOD_Grid_Snow_500m:NDSI_Snow_Cover"
    
    # Create a name for the output file based on the HDF file name and subdataset
    output_file_name = os.path.splitext(os.path.basename(hdf_file))[0] + ".tif"
    output_path = os.path.join(output_folder, output_file_name)

    if os.path.exists(output_path):
        pass
    else:
        for subdataset in hdf_ds.GetSubDatasets():
            if target_subdataset_name in subdataset[0]:
                ds = gdal.Open(subdataset[0], gdal.GA_ReadOnly)
                gdal.Translate(output_path, ds)
                ds = None
                break

    hdf_ds = None

def convert_all_hdf_in_folder(folder_path, output_folder):
    """
    Converts all HDF files in a given folder to GeoTIFF format.

    Args:
        folder_path (str): The directory containing HDF files to be converted.
        output_folder (str): The directory where the converted GeoTIFF files will be saved.

    Returns:
        list: A list of file names that were found in the folder.
    """
    file_lst = list()
    for file in os.listdir(folder_path):
        file_lst.append(file)
        if file.lower().endswith(".hdf"):
            hdf_file = os.path.join(folder_path, file)
            convert_hdf_to_geotiff(hdf_file, output_folder)
    return file_lst

def merge_tifs(folder_path, target_date, output_file):
    """
    Merges multiple GeoTIFF files into a single GeoTIFF file for a specific date.

    Args:
        folder_path (str): The directory containing GeoTIFF files to be merged.
        target_date (datetime): The date for which the GeoTIFF files should be merged.
        output_file (str): The file path where the merged GeoTIFF file will be saved.

    Returns:
        None
    """
    julian_date = date_to_julian(target_date)
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif') and julian_date in f]
    
    if len(tif_files) == 0:
        gdal_command = ['gdal_translate', '-b', '1', '-outsize', '100%', '100%', '-scale', '0', '255', '200', '200', 
                        f"{modis_day_wise}/fsca_template.tif", output_file]
        subprocess.run(gdal_command)
    else:
        gdal_command = ['gdalwarp', '-r', 'min'] + tif_files + [f"{output_file}_500m.tif"]
        subprocess.run(gdal_command)
        
        gdal_command = ['gdalwarp', '-t_srs', 'EPSG:4326', '-tr', '0.036', '0.036', '-cutline', 
                        f'{work_dir}/template.shp', '-crop_to_cutline', '-overwrite', f"{output_file}_500m.tif", output_file]
        subprocess.run(gdal_command)

def list_files(directory):
    """
    Lists all files in a specified directory.

    Args:
        directory (str): The directory from which to list files.

    Returns:
        list: A list of absolute file paths in the specified directory.
    """
    return [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def merge_tiles(date, hdf_files):
    """
    Merges multiple tiles into a single GeoTIFF file for a specific date.

    Args:
        date (str): The date for which the tiles should be merged (format: YYYY-MM-DD).
        hdf_files (list): A list of HDF file paths to be merged.

    Returns:
        None
    """
    path = f"data/{date}/"
    files = list_files(path)
    merged_filename = f"data/{date}/merged.tif"
    merge_command = ["gdal_merge.py", "-o", merged_filename, "-of", "GTiff"] + files
    try:
        subprocess.run(merge_command)
        print(f"Merged tiles into {merged_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging tiles: {e}")

def download_url(date, url):
    """
    Downloads a file from a specified URL to a local directory for a specific date.

    Args:
        date (str): The date for which the file is being downloaded (format: YYYY-MM-DD).
        url (str): The URL from which to download the file.

    Returns:
        None
    """
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
                  
