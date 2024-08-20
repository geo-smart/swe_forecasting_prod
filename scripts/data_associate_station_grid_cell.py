import json
import pandas as pd
import ee
import seaborn as sns
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import geojson
import numpy as np
import os.path
import math
from snowcast_utils import work_dir, read_json_file

def calculateDistance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


if __name__ == "__main__":
  
    # pd.set_option('display.max_columns', None)

    # read the grid geometry file
    homedir = os.path.expanduser('~')
    print(homedir)
    github_dir = f"{homedir}/Documents/GitHub/SnowCast"
    # read grid cell
    gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
    model_dir = f"{github_dir}/model/"
    training_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_train_features.csv"
    testing_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_test_features.csv"
    train_labels_file = f"{github_dir}/data/snowcast_provided/train_labels.csv"
    ground_measure_metadata_file = f"{github_dir}/data/snowcast_provided/ground_measures_metadata.csv"
    station_list_file = f"{work_dir}/training_snotel_station_list_elevation.csv"

    ready_for_training_folder = f"{github_dir}/data/ready_for_training/"

    result_mapping_file = f"{ready_for_training_folder}station_cell_mapping.csv"

    station_locations = read_json_file(f'{work_dir}/snotelStations.json')
    # print(station_locations)

    result_df = pd.DataFrame(columns=['station_name', 'elevation', 'lat', 'lon'])
    for station in station_locations:
        print(f'station {station["name"]} completed.')

        location_name = station['name']
        location_triplet = station['triplet']
        location_elevation = station['elevation']
        location_station_lat = station['location']['lat']
        location_station_long = station['location']['lng']
        new_df = pd.DataFrame([{
            'station_name': location_name,
            'elevation': location_elevation,
            'lat': location_station_lat,
            'lon': location_station_long
        }])
        result_df = pd.concat([result_df, new_df], axis=0, ignore_index=True)
             
              
    print(result_df)
    # Save the DataFrame to a CSV file
    result_df.to_csv(station_list_file, index=False)





