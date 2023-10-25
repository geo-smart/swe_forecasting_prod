import os
import glob
import urllib.request
from datetime import date

import pandas as pd
import xarray as xr
from pathlib import Path
from snowcast_utils import work_dir
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

start_date = date(2019, 1, 1)
end_date = date(2022, 12, 31)

year_list = [start_date.year + i for i in range(end_date.year - start_date.year + 1)]

working_dir = work_dir
stations = pd.read_csv(f'{working_dir}/station_cell_mapping.csv')
gridmet_save_location = f'{working_dir}/gridmet_climatology'
final_merged_csv = f'{working_dir}/gridmet_climatology/training_ready_gridmet.csv'


def get_files_in_directory():
    f = list()
    for files in glob.glob(gridmet_save_location + "/*.nc"):
        f.append(files)
    return f


def download_file(url, save_location):
    try:
        print("download_file")
        with urllib.request.urlopen(url) as response:
            file_content = response.read()
        file_name = os.path.basename(url)
        save_path = os.path.join(save_location, file_name)
        with open(save_path, 'wb') as file:
            file.write(file_content)
        print(f"File downloaded successfully and saved as: {save_path}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")


def gridmet_climatology():
    folder_name = gridmet_save_location
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    base_metadata_url = "http://www.northwestknowledge.net/metdata/data/"
    variables_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'etr', 'rmax', 'rmin', 'vs']

    for var in variables_list:
        for y in year_list:
            download_link = base_metadata_url + var + '_' + '%s' % y + '.nc'
            print("downloading", download_link)
            if not os.path.exists(os.path.join(folder_name, var + '_' + '%s' % y + '.nc')):
                download_file(download_link, folder_name)


def get_gridmet_variable(file_name):
    print(f"reading values from {file_name}")
    result_data = []
    ds = xr.open_dataset(file_name)
    var_to_extract = list(ds.keys())
    print(var_to_extract)
    var_name = var_to_extract[0]
    
    df = pd.DataFrame(columns=['day', 'lat', 'lon', var_name])
    
    csv_file = f'{gridmet_save_location}/{Path(file_name).stem}.csv'
    if os.path.exists(csv_file):
    	print(f"The file '{csv_file}' exists.")
        return

    for idx, row in stations.iterrows():
        lat = row['lat']
        lon = row['lon']
		
        subset_data = ds.sel(lat=lat, lon=lon, method='nearest')
        subset_data['lat'] = lat
        subset_data['lon'] = lon
        # print('subset data:', lat, lon, subset_data.values())
        converted_df = subset_data.to_dataframe()
        #print("converted_df: ", converted_df.head())
        #print("converted_df columns: ", converted_df.columns)
        converted_df = converted_df.reset_index(drop=False)
        #print("convert to columns: ", converted_df.columns)
        converted_df = converted_df.drop('crs', axis=1)
        df = pd.concat([df, converted_df], ignore_index=True)
        
    result_df = df
    print("got result_df : ", result_df.head())
    result_df.to_csv(csv_file, index=False)
    print(f'completed extracting data for {file_name}')


def merge_similar_variables_from_different_years():
    files = os.listdir(gridmet_save_location)
    file_groups = {}

    for filename in files:
        base_name, year_ext = os.path.splitext(filename)
        parts = base_name.split('_')
        if len(parts) == 2 and year_ext == '.csv':
            file_groups.setdefault(parts[0], []).append(filename)

    for base_name, file_list in file_groups.items():
        if len(file_list) > 1:
            dfs = []
            for filename in file_list:
                df = pd.read_csv(os.path.join(gridmet_save_location, filename))
                dfs.append(df)
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_filename = f"{base_name}_merged.csv"
            merged_df.to_csv(os.path.join(gridmet_save_location, merged_filename), index=False)
            print(f"Merged {file_list} into {merged_filename}")


def merge_all_variables_together():
    merged_df = None
    file_paths = []

    for filename in os.listdir(gridmet_save_location):
        if filename.endswith("_merged.csv"):
            file_paths.append(os.path.join(gridmet_save_location, filename))
	
    rmin_merged_path = os.path.join(gridmet_save_location, 'rmin_merged.csv')
    rmax_merged_path = os.path.join(gridmet_save_location, 'rmax_merged.csv')
    tmmn_merged_path = os.path.join(gridmet_save_location, 'tmmn_merged.csv')
    tmmx_merged_path = os.path.join(gridmet_save_location, 'tmmx_merged.csv')
    
    df_rmin = pd.read_csv(rmin_merged_path)
    df_rmax = pd.read_csv(rmax_merged_path)
    df_tmmn = pd.read_csv(tmmn_merged_path)
    df_tmmx = pd.read_csv(tmmx_merged_path)
    
    df_rmin.rename(columns={'relative_humidity': 'relative_humidity_rmin'}, inplace=True)
    df_rmax.rename(columns={'relative_humidity': 'relative_humidity_rmax'}, inplace=True)
    df_tmmn.rename(columns={'air_temperature': 'air_temperature_tmmn'}, inplace=True)
    df_tmmx.rename(columns={'air_temperature': 'air_temperature_tmmx'}, inplace=True)
    
    df_rmin.to_csv(os.path.join(gridmet_save_location, 'rmin_merged.csv'))
    df_rmax.to_csv(os.path.join(gridmet_save_location, 'rmax_merged.csv'))
    df_tmmn.to_csv(os.path.join(gridmet_save_location, 'tmmn_merged.csv'))
    df_tmmx.to_csv(os.path.join(gridmet_save_location, 'tmmx_merged.csv'))
    
    if file_paths:
        merged_df = pd.read_csv(file_paths[0])
        for file_path in file_paths[1:]:
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df], axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        merged_df.to_csv(final_merged_csv, index=False)


gridmet_climatology()
nc_files = get_files_in_directory()

for nc in nc_files:
    get_gridmet_variable(nc)
merge_similar_variables_from_different_years()
merge_all_variables_together()

