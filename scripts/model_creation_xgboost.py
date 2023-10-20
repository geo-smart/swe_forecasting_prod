"""
This script defines the XGBoostHole class, which is used to train and evaluate an Extra Trees Regression model for hole analysis.

Attributes:
    XGBoostHole (class): A class for training and using an Extra Trees Regression model for hole analysis.

Functions:
    get_model(): Returns the Extra Trees Regressor model with specified hyperparameters.
"""

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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from base_hole import BaseHole
from sklearn.model_selection import train_test_split
from datetime import datetime
from model_creation_rf import RandomForestHole
from sklearn.ensemble import ExtraTreesRegressor

class XGBoostHole(RandomForestHole):

    def get_model(self):
        """
        Returns the Extra Trees Regressor model with specified hyperparameters.

        Returns:
            ExtraTreesRegressor: The Extra Trees Regressor model.
        """
        etmodel = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0,
                    min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                    random_state=123, verbose=0, warm_start=False)
        return etmodel

