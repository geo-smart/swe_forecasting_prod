"""
This script defines the RandomForestHole class, which is used for training and evaluating a Random Forest Regressor model for hole analysis.

Attributes:
    RandomForestHole (class): A class for training and using a Random Forest Regressor model for hole analysis.

Functions:
    get_model(): Returns the Random Forest Regressor model with specified hyperparameters.
    evaluate(): Evaluates the performance of the trained model and returns metrics such as MAE, MSE, R2, and RMSE.
"""

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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from base_hole import BaseHole
from sklearn.model_selection import train_test_split
from datetime import datetime

homedir = os.path.expanduser('~')
github_dir = os.path.join(homedir, 'Documents', 'GitHub', 'SnowCast')

class RandomForestHole(BaseHole):
  
    def get_model(self):
        """
        Returns the Random Forest Regressor model with specified hyperparameters.

        Returns:
            Pipeline: The Random Forest Regressor model wrapped in a scikit-learn pipeline.
        """
        rfc_pipeline = Pipeline(steps=[
            ('data_scaling', StandardScaler()),
            ('model', RandomForestRegressor(max_depth=15,
                                           min_samples_leaf=0.004,
                                           min_samples_split=0.008,
                                           n_estimators=25))
        ])
        return rfc_pipeline

    def evaluate(self):
        """
        Evaluates the performance of the trained model and returns metrics such as MAE, MSE, R2, and RMSE.

        Returns:
            dict: A dictionary containing MAE, MSE, R2, and RMSE metrics.
        """
        mae = metrics.mean_absolute_error(self.test_y, self.test_y_results)
        mse = metrics.mean_squared_error(self.test_y, self.test_y_results)
        r2 = metrics.r2_score(self.test_y, self.test_y_results)
        rmse = math.sqrt(mse)

        print("The random forest model performance for testing set")
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('R2 score is {}'.format(r2))
        print('RMSE is {}'.format(rmse))
        return {"mae": mae, "mse": mse, "r2": r2, "rmse": rmse}

