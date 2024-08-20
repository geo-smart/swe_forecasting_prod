# 2019 first https://nsidc.org/data/nsidc-0719/versions/1#anchor-1

# TODO: change LAT LONG TO GRID CELL COORDS
# TODO: adjust using grid cell geojson in data integration
# TODO: adjust to make model validation working (model_train_validate)

"""
Broxton, P., X. Zeng, and N. Dawson. 2019. Daily 4 km Gridded SWE and Snow Depth from
Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1. 2019-2021.
Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center.
https://doi.org/10.5067/0GGPB220EX6A. 11/02/2022.

To enable wget to directly download netcdf from NSIDC, use:

echo 'machine urs.earthdata.nasa.gov login <uid> password <password>' >> ~/.netrc
chmod 0600 ~/.netrc

"""

from math import cos, asin, sqrt, radians
import pandas as pd
import numpy as np
import os.path
import netCDF4 as nc
import datetime
import geojson
from sklearn import neighbors as sk
import sys

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)

# read grid cell
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
grid_cells = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"
print("station_cell_mapper_file = ", station_cell_mapper_file)
station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)
# open nsidc data file (netCDF)
# crs, lat, lon, time, time_str, DEPTH, SWE, SWE_MASK
# change to make it work
end_year = 2019
# https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/4km_SWE_Depth_WY2019_v01.nc
nsidc_data_file = f"{homedir}/Documents/data/4km_SWE_Depth_WY{end_year}_v01.nc"
nsidc_data_ds = nc.Dataset(nsidc_data_file)

print(nsidc_data_ds)
for dim in nsidc_data_ds.dimensions.values():
    print(dim)
for var in nsidc_data_ds.variables.values():
    print(var)

# dates based on Water Year 2019 (not normal year)
org_name = 'nsidc'
product_name = 'NSIDC'
start_date = '2018-10-01'
end_date = '2019-09-30'

dfolder = f"{homedir}/Documents/GitHub/SnowCast/data/sim_training/{org_name}/"
if not os.path.exists(dfolder):
    os.makedirs(dfolder)

# Removes duplicate indices
scmd = set(station_cell_mapper_df['cell_id'])

lat = nsidc_data_ds.variables['lat'][:]
lon = nsidc_data_ds.variables['lon'][:]
depth = nsidc_data_ds.variables['DEPTH']
swe = nsidc_data_ds.variables['SWE']
time = nsidc_data_ds.variables['time']
columns = ['Year', 'Month', 'Day', 'Lat', 'Lon', 'SWE', 'Depth']

start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
# conversion factor so we can get days from 0-364 for array
days_1900_start = int((start_date_dt - datetime.datetime(1900,1,1)).days)

all_cells_df = pd.DataFrame(columns=columns)
ind = 0


# haversine formula
def coord_distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))


# inefficient and bad, don't use this
def find_nearest(find_lat, find_lng):
    min_dist = 999999999
    curr_min_lat_idx = 0
    curr_min_lon_idx = 0

    lat_len = len(lat)
    lon_len = len(lon)
    # iterate through lat and long to find closest val
    for lat_idx in range(lat_len):
        for lon_idx in range(lon_len):
            if coord_distance(lat[lat_idx], lon[lon_idx], find_lat, find_lng) < min_dist:
                if depth[23, lat_idx, lon_idx] != '--':
                    min_dist = coord_distance(lat[lat_idx], lon[lon_idx], find_lat, find_lng)
                    curr_min_lat_idx = lat_idx
                    curr_min_lon_idx = lon_idx

    return curr_min_lat_idx, curr_min_lon_idx


# for generating the list of all valid lat long pairs
def gen_pairs():
    temp = []
    lat_len = len(lat)
    lon_len = len(lon)
    # iterate through lat and long to find closest val
    for lat_idx in range(lat_len):
        for lon_idx in range(lon_len):
            if depth[23, lat_idx, lon_idx] != '--':
                temp.append((lat[lat_idx], lon[lon_idx]))
    temp = np.array(temp)
    print(temp)
    np.save(f"{dfolder}/valid_pairs.npy", temp)


# use balltree to find closest neighbors, convert to radians first so the haversine thing works correctly
# (that's why there's a separate rad thing)
def find_nearest_2(find_lat, find_lng):
    ball_tree = sk.BallTree(lat_lon_pairs_rad, metric="haversine")

    dist, ind = ball_tree.query([(radians(find_lat), radians(find_lng))], return_distance=True)
    print(dist)
    print(ind)
    print(lat_lon_pairs[ind])
    curr_min_lat_idx = lat_lon_pairs[ind][0][0][0]
    curr_min_lon_idx = lat_lon_pairs[ind][0][0][1]
    return curr_min_lat_idx, curr_min_lon_idx


def turn_nsidc_nc_to_csv():
    # generate valid pairs, or just load if they already exist
    if not os.path.exists(f"{dfolder}/valid_pairs.npy"):
        print("file doesn't exist, generating new")
        gen_pairs()
    lat_lon_pairs = np.load(f"{dfolder}/valid_pairs.npy")
    lat_lon_pairs_rad = np.array([[radians(x[0]), radians(x[1])] for x in lat_lon_pairs])

    # comment out if bulk writing!!
    # all_cells_df.to_csv(f"{dfolder}/test.csv", index=False)

    for ind, current_cell_id in enumerate(scmd):
        # comment out if bulk writing
        # all_cells_df = pd.DataFrame(columns=columns)

        # Location information
        longitude = station_cell_mapper_df['lon'][ind]
        latitude = station_cell_mapper_df['lat'][ind]

    #     print(latitude)
    #     print(longitude)

        # find closest lat long
        lat_val, lon_val = find_nearest_2(latitude, longitude)
        lat_idx = np.where(lat == lat_val)[0]
        lon_idx = np.where(lon == lon_val)[0]
    #     print(lat_val)
    #     print(lon_val)

        depth_time = depth[:, lat_idx, lon_idx]
        swe_time = swe[:, lat_idx, lon_idx]

        for ele in time:
            time_index = int(ele.data - days_1900_start)
            time_index_dt = datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(int(ele.data))
            depth_val = depth_time[time_index][0][0]
            swe_val = swe_time[time_index][0][0]

            all_cells_df.loc[len(all_cells_df.index)] = [time_index_dt.year, time_index_dt.month, time_index_dt.day, lat_val, lon_val, swe_val, depth_val]

        # comment out if bulk writing
        # all_cells_df.to_csv(f"{dfolder}/test.csv", mode='a', header=False, index=False)

    # uncomment to bulk write at end of program
    all_cells_df.to_csv(f"{dfolder}/{end_year}nsidc_data.csv")

    print("finished")

if __name__ == "__main__":
    # call this method to extract the 
    turn_nsidc_nc_to_csv()
