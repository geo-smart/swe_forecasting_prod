import os
import glob
import urllib.request
from datetime import date, datetime

import pandas as pd
import xarray as xr
from pathlib import Path
from snowcast_utils import work_dir, train_start_date, train_end_date
import warnings
import dask.dataframe as dd
from dask.delayed import delayed

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
end_date = datetime.strptime(train_end_date, "%Y-%m-%d")

year_list = [start_date.year + i for i in range(end_date.year - start_date.year + 1)]

working_dir = work_dir
#stations = pd.read_csv(f'{working_dir}/station_cell_mapping.csv')
all_training_points_with_snotel_ghcnd_file = f"{work_dir}/all_training_points_snotel_ghcnd_in_westus.csv"
stations = pd.read_csv(all_training_points_with_snotel_ghcnd_file)
gridmet_save_location = f'{working_dir}/gridmet_climatology'
final_merged_csv = f"{work_dir}/training_all_point_gridmet_with_snotel_ghcnd.csv"


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


def download_gridmet_climatology():
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


def process_station(ds, lat, lon):
    subset_data = ds.sel(lat=lat, lon=lon, method='nearest')
    subset_data['lat'] = lat
    subset_data['lon'] = lon
    converted_df = subset_data.to_dataframe()
    converted_df = converted_df.reset_index(drop=False)
    converted_df = converted_df.drop('crs', axis=1)
    return converted_df

def get_gridmet_variable(file_name):
    print(f"Reading values from {file_name}")
    ds = xr.open_dataset(file_name)
    var_name = list(ds.keys())[0]

    csv_file = f'{gridmet_save_location}/{Path(file_name).stem}_snotel_ghcnd.csv'
    if os.path.exists(csv_file):
        print(f"The file '{csv_file}' exists.")
        return

    result_data = []
    for _, row in stations.iterrows():
        delayed_process_data = delayed(process_station)(ds, row['latitude'], row['longitude'])
        result_data.append(delayed_process_data)

    print("ddf = dd.from_delayed(result_data)")
    ddf = dd.from_delayed(result_data)
    
    print("result_df = ddf.compute()")
    result_df = ddf.compute()
    result_df.to_csv(csv_file, index=False)
    print(f'Completed extracting data for {file_name}')


def merge_similar_variables_from_different_years():
    files = os.listdir(gridmet_save_location)
    file_groups = {}

    for filename in files:
        base_name, year_ext = os.path.splitext(filename)
        parts = base_name.split('_')
        print(parts)
        print(year_list)
        if len(parts) == 4 and parts[3] == "ghcnd" and year_ext == '.csv' and int(parts[1]) in year_list:
            file_groups.setdefault(parts[0], []).append(filename)

    for base_name, file_list in file_groups.items():
        if len(file_list) > 1:
            dfs = []
            for filename in file_list:
                df = pd.read_csv(os.path.join(gridmet_save_location, filename))
                dfs.append(df)
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_filename = f"{base_name}_merged_snotel_ghcnd.csv"
            merged_df.to_csv(os.path.join(gridmet_save_location, merged_filename), index=False)
            print(f"Merged {file_list} into {merged_filename}")


def merge_all_variables_together():
    merged_df = None
    file_paths = []

    for filename in os.listdir(gridmet_save_location):
        if filename.endswith("_merged_snotel_ghcnd.csv"):
            file_paths.append(os.path.join(gridmet_save_location, filename))
	
    rmin_merged_path = os.path.join(gridmet_save_location, 'rmin_merged_snotel_ghcnd.csv')
    rmax_merged_path = os.path.join(gridmet_save_location, 'rmax_merged_snotel_ghcnd.csv')
    tmmn_merged_path = os.path.join(gridmet_save_location, 'tmmn_merged_snotel_ghcnd.csv')
    tmmx_merged_path = os.path.join(gridmet_save_location, 'tmmx_merged_snotel_ghcnd.csv')
    
    df_rmin = pd.read_csv(rmin_merged_path)
    df_rmax = pd.read_csv(rmax_merged_path)
    df_tmmn = pd.read_csv(tmmn_merged_path)
    df_tmmx = pd.read_csv(tmmx_merged_path)
    
    df_rmin.rename(columns={'relative_humidity': 'relative_humidity_rmin'}, inplace=True)
    df_rmax.rename(columns={'relative_humidity': 'relative_humidity_rmax'}, inplace=True)
    df_tmmn.rename(columns={'air_temperature': 'air_temperature_tmmn'}, inplace=True)
    df_tmmx.rename(columns={'air_temperature': 'air_temperature_tmmx'}, inplace=True)
    
    df_rmin.to_csv(rmin_merged_path)
    df_rmax.to_csv(rmax_merged_path)
    df_tmmn.to_csv(tmmn_merged_path)
    df_tmmx.to_csv(tmmx_merged_path)
    
    if file_paths:
        merged_df = pd.read_csv(file_paths[0])
        for file_path in file_paths[1:]:
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df], axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        merged_df.to_csv(final_merged_csv, index=False)
        print(f"all files are saved to {final_merged_csv}")


if __name__ == "__main__":
    
    #download_gridmet_climatology()
    
    # mock out as this takes too long
    #nc_files = get_files_in_directory()
    #for nc in nc_files:
    #.   # should check if the nc file year number is in the year_list
    #    get_gridmet_variable(nc)
    
    merge_similar_variables_from_different_years()
    merge_all_variables_together()

