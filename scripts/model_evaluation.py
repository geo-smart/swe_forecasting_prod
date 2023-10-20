# Predict results using the model

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn import tree
import joblib
import os
from pathlib import Path
import json
import geopandas as gpd
import geojson
import os.path
import math
from sklearn.model_selection import RandomizedSearchCV

exit(0)  # for now, the workflow is not ready yet

# read the grid geometry file


# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
modis_test_ready_file = f"{github_dir}/data/ready_for_training/modis_test_ready.csv"
modis_test_ready_pd = pd.read_csv(modis_test_ready_file, header=0, index_col=0)

pd_to_clean = modis_test_ready_pd[["year", "m", "doy", "ndsi", "swe", "station_id", "cell_id"]].dropna()

all_features = pd_to_clean[["year", "m", "doy", "ndsi"]].to_numpy()
all_labels = pd_to_clean[["swe"]].to_numpy().ravel()

def evaluate(model, test_features, y_test, model_name):
    y_predicted = model.predict(test_features)
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)
    rmse = math.sqrt(mse)

    print("The {} model performance for testing set".format(model_name))
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))
    print('RMSE is {}'.format(rmse))
    
    return y_predicted

base_model = joblib.load(f"{homedir}/Documents/GitHub/snowcast_trained_model/model/wormhole_random_forest_basic.joblib")
basic_predicted_values = evaluate(base_model, all_features, all_labels, "Base Model")

best_random = joblib.load(f"{homedir}/Documents/GitHub/snowcast_trained_model/model/wormhole_random_forest.joblib")
random_predicted_values = evaluate(best_random, all_features, all_labels, "Optimized")



