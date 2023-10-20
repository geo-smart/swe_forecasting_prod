import os
import pprint

# import gdal
import subprocess
from datetime import datetime, timedelta

# set up your credentials using
# echo 'machine urs.earthdata.nasa.gov login <uid> password <password>' >> ~/.netrc

modis_download_dir = "/home/chetana/modis_download_folder/"
modis_downloaded_data = modis_download_dir + "n5eil01u.ecs.nsidc.org/MOST/MOD10A2.061/"
geo_tiff = modis_download_dir + "geo-tiff/"
vrt_file_dir = modis_download_dir + "vrt_files/"
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

tile_list = ['h09v04', 'h10v04', 'h11v04', 'h08v04', 'h08v05', 'h09v05', 'h10v05', 'h07v06', 'h08v06', 'h09v06']


def get_files(directory):
    """
    Get a list of files in a directory and its subdirectories.

    Args:
        directory (str): The directory to search for files.

    Returns:
        dict: A dictionary where keys are subdirectory names and values are lists of file paths.
    """
    file_directory = list()
    complete_directory_structure = dict()
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_directory.append(file_path)
            complete_directory_structure[str(dirpath).rsplit('/')[-1]] = file_directory

    return complete_directory_structure


def get_latest_date():
    """
    Retrieve the latest date from the MODIS data website.

    Returns:
        datetime: The latest date as a datetime object.
    """
    all_rows = get_web_row_data()

    latest_date = None
    for row in all_rows:
        try:
            new_date = datetime.strptime(row.text[:-1], '%Y.%m.%d')
            if latest_date is None or latest_date < new_date:
                latest_date = new_date
        except:
            continue

    print("Find the latest date: ", latest_date.strftime("%Y.%m.%d"))
    second_latest_date = latest_date - timedelta(days=8)
    return second_latest_date


def get_web_row_data():
    """
    Fetch and parse the MODIS data website content.

    Returns:
        list: A list of rows from the website's table.
    """
    try:
        from BeautifulSoup import BeautifulSoup
    except ImportError:
        from bs4 import BeautifulSoup
    modis_list_url = "https://n5eil01u.ecs.nsidc.org/MOST/MOD10A2.061/"
    print("Source / Product: " + modis_list_url)
    if os.path.exists("index.html"):
        os.remove("index.html")
    subprocess.run(
        f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '
        f'--no-check-certificate --auth-no-challenge=on -np -e robots=off {modis_list_url}',
        shell=True, stderr=subprocess.PIPE)
    index_file = open('index.html', 'r')
    webContent = index_file.read()
    parsed_html = BeautifulSoup(webContent, "html.parser")
    all_rows = parsed_html.body.findAll('td', attrs={'class': 'indexcolname'})
    return all_rows


def download_recent_modis(date=None):
    """
    Download recent MODIS data.

    Args:
        date (datetime, optional): A specific date to download. Defaults to None.
    """
    if date:
        latest_date_str = date.strftime("%Y.%m.%d")
    else:
        latest_date_str = get_latest_date().strftime("%Y.%m.%d")
    for tile in tile_list:
        download_cmd = f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies ' \
                       f'--no-check-certificate --auth-no-challenge=on -r --reject "i' \
                       f'ndex.html*" -P {modis_download_dir} -np -e robots=off ' \
                       f'https://n5eil01u.ecs.nsidc.org/MOST/MOD10A2.061/{latest_date_str}/ -A "*{tile}*.hdf" --quiet'
        # print(download_cmd)
        p = subprocess.run(download_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Downloading tile, ", tile, " with status code ", "OK" if p.returncode == 0 else p.returncode)


# def merge_wrap_tif_into_western_us_tif():
#     latest_date_str = get_latest_date().strftime("%Y.%m.%d")
#     # traverse the folder and find the new download files
#     for filename in os.listdir(f"n5eil01u.ecs.nsidc.org/MOST/MOD10A2.061/{latest_date_str}/"):
#         f = os.path.join(directory, filename)
#         # checking if it is a file
#         if os.path.isfile(f):
#             print(f)
# merge_wrap_tif_into_western_us_tif()

def hdf_tif_cvt(resource_path, destination_path):
    """
    Convert HDF files to GeoTIFF format.

    Args:
        resource_path (str): The path to the source HDF file.
        destination_path (str): The path to save the converted GeoTIFF file.
    """
    if not os.path.isfile(resource_path):
        raise Exception("HDF file not found")

    max_snow_extent_path = destination_path + "maximum_snow_extent/"
    eight_day_snow_cover = destination_path + "eight_day_snow_cover/"
    if not os.path.exists(max_snow_extent_path):
        os.makedirs(max_snow_extent_path)
    if not os.path.exists(eight_day_snow_cover):
        os.makedirs(eight_day_snow_cover)

    tif_file_name_snow_extent = max_snow_extent_path + resource_path.split('/')[-1].split('.hdf')[0]
    tif_file_name_eight_day = eight_day_snow_cover + resource_path.split('/')[-1].split('.hdf')[0]
    tif_file_extension = '.tif'

    maximum_snow_extent_file_name = tif_file_name_snow_extent + '_max_snow_extent' + tif_file_extension
    eight_day_snow_cover_file_name = tif_file_name_eight_day + '_modis_snow_500m' + tif_file_extension

    maximum_snow_extent = f"HDF4_EOS:EOS_GRID:\"{resource_path}\":MOD_Grid_Snow_500m:Maximum_Snow_Extent"
    eight_day_snow_cover = f"HDF4_EOS:EOS_GRID:\"{resource_path}\":MOD_Grid_Snow_500m:Eight_Day_Snow_Cover"

    subprocess.run(f"gdal_translate {maximum_snow_extent} {maximum_snow_extent_file_name}", shell=True)
    subprocess.run(f"gdal_translate {eight_day_snow_cover} {eight_day_snow_cover_file_name}", shell=True)


def combine_geotiff_gdal(vrt_array, destination):
    """
    Combine GeoTIFF files using GDAL.

    Args:
        vrt_array (list): A list of GeoTIFF file paths to combine.
        destination (str): The path to save the combined VRT and GeoTIFF files.
    """
    subprocess.run(f"gdalbuildvrt {destination} {' '.join(vrt_array)}", shell=True)
    tif_name = destination.split('.vrt')[-2] + '.tif'
    subprocess.run(f"gdal_translate -of GTiff {destination} {tif_name}", shell=True)


def hdf_tif_conversion(resource_path, destination_path):
    """
    Convert HDF files to GeoTIFF format using GDAL.

    Args:
        resource_path (str): The path to the source HDF file.
        destination_path (str): The path to save the converted GeoTIFF file.
    """
    hdf_dataset = gdal.Open(resource_path)
    if hdf_dataset is None:
        raise Exception("Could not open HDF dataset")

    maximum_snow_extent = hdf_dataset.GetSubDatasets()[0][0]
    modis_snow_500m = hdf_dataset.GetSubDatasets()[1][0]

    driver = gdal.GetDriverByName('GTiff')

    tif_file_name = destination_path + resource_path.split('/')[-1].split('.hdf')[0]
    tif_file_extension = '.tif'

    maximum_snow_extent_file_name = tif_file_name + '_max_snow_extent' + tif_file_extension
    modis_snow_500m_file_name = tif_file_name + '_modis_snow_500m' + tif_file_extension

    maximum_snow_extent_dataset = gdal.Open(maximum_snow_extent)
    modis_snow_500m_dataset = gdal.Open(modis_snow_500m)

    if maximum_snow_extent_dataset is None:
        raise Exception("Could not open maximum_snow_extent dataset")

    if modis_snow_500m_dataset is None:
        raise Exception("Could not open modis_snow_500m dataset")

    driver.CreateCopy(maximum_snow_extent_file_name, maximum_snow_extent_dataset, 0)
    driver.CreateCopy(modis_snow_500m_file_name, modis_snow_500m_dataset, 0)

    print("HDF to TIF conversion completed successfully.")


def download_modis_archive(*, start_date, end_date):
    """
    Download MODIS data for a specified date range.

    Keyword Args:
        start_date (datetime): The start date of the date range.
        end_date (datetime): The end date of the date range.
    """
    all_archive_dates = list()

    all_rows = get_web_row_data()
    for r in all_rows:
        try:
            all_archive_dates.append(datetime.strptime(r.text.replace('/', ''), '%Y.%m.%d'))
        except:
            continue

    for a in all_archive_dates:
        if start_date <= a <= end_date:
            download_recent_modis(a)


def step_one_download_modis():
  """
  Step one of the main workflow: Download recent MODIS data.
  """
  download_recent_modis()
                   
def step_two_merge_modis_western_us():
  """
  Step two of the main workflow: Merge MODIS data for the western US.
  """
  download_modis_archive(start_date=datetime(2022, 1, 1), end_date=datetime(2022, 12, 31))

  files = get_files(modis_downloaded_data)
  for k, v in get_files(modis_downloaded_data).items():

    conversion_path = modis_download_dir + "geo-tiff/" + k + "/"
    if not os.path.exists(conversion_path):
        os.makedirs(conversion_path)
    for hdf_file in v:
        # print(hdf_file.split('/')[-1].split('.hdf')[0], 1)
        hdf_tif_cvt(hdf_file, conversion_path)

  if not os.path.exists(vrt_file_dir):
    os.makedirs(vrt_file_dir)


  directories = [d for d in os.listdir(geo_tiff) if   os.path.isdir(os.path.join(geo_tiff, d))]

  for d in directories:
    eight_day_snow_cover = geo_tiff + d + '/eight_day_snow_cover'
    maximum_snow_extent = geo_tiff + d + '/maximum_snow_extent'

    eight_day_abs_path = list()
    snow_extent_abs_path = list()

    for file in os.listdir(eight_day_snow_cover):
        file_path = os.path.abspath(os.path.join(eight_day_snow_cover, file))
        eight_day_abs_path.append(file_path)

    for file in os.listdir(maximum_snow_extent):
        file_path = os.path.abspath(os.path.join(maximum_snow_extent, file))
        snow_extent_abs_path.append(file_path)

    combine_geotiff_gdal(eight_day_abs_path, vrt_file_dir + f"{d}_eight_day.vrt")
    combine_geotiff_gdal(snow_extent_abs_path, vrt_file_dir + f"{d}_snow_extent.vrt")

                   
# main workflow is here:
step_one_download_modis()
step_two_merge_modis_western_us()



