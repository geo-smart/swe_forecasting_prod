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

ready_for_training_folder = f"{github_dir}/data/ready_for_training/"

result_mapping_file = f"{ready_for_training_folder}station_cell_mapping.csv"

if os.path.exists(result_mapping_file):
    exit()

gridcells = geojson.load(open(gridcells_file))
training_df = pd.read_csv(training_feature_file, header=0)
testing_df = pd.read_csv(testing_feature_file, header=0)
ground_measure_metadata_df = pd.read_csv(ground_measure_metadata_file, header=0)
train_labels_df = pd.read_csv(train_labels_file, header=0)

print("training: ", training_df.head())
print("testing: ", testing_df.head())
print("ground measure metadata: ", ground_measure_metadata_df.head())
print("training labels: ", train_labels_df.head())


def calculateDistance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# prepare the training data

station_cell_mapper_df = pd.DataFrame(columns=["station_id", "cell_id", "lat", "lon"])

ground_measure_metadata_df = ground_measure_metadata_df.reset_index()  # make sure indexes pair with number of rows
for index, row in ground_measure_metadata_df.iterrows():

    print(row['station_id'], row['name'], row['latitude'], row['longitude'])
    station_lat = row['latitude']
    station_lon = row['longitude']

    shortest_dis = 999
    associated_cell_id = None
    associated_lat = None
    associated_lon = None

    for idx, cell in enumerate(gridcells['features']):

        current_cell_id = cell['properties']['cell_id']

        # print("collecting ", current_cell_id)
        cell_lon = np.unique(np.ravel(cell['geometry']['coordinates'])[0::2]).mean()
        cell_lat = np.unique(np.ravel(cell['geometry']['coordinates'])[1::2]).mean()

        dist = calculateDistance(station_lat, station_lon, cell_lat, cell_lon)

        if dist < shortest_dis:
            associated_cell_id = current_cell_id
            shortest_dis = dist
            associated_lat = cell_lat
            associated_lon = cell_lon

    station_cell_mapper_df.loc[len(station_cell_mapper_df.index)] = [row['station_id'], associated_cell_id,
                                                                     associated_lat, associated_lon]

print(station_cell_mapper_df.head())
station_cell_mapper_df.to_csv(f"{ready_for_training_folder}station_cell_mapping.csv")





