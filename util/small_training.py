'''
This file provides the ETHoleTiny model class and helper functions to support small training experiments.
'''

from random import random
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

class ETHoleTiny:
    def __init__(self):
        '''
        Initialize a tiny wormhole ET regressor.
        '''
        regressor = ExtraTreesRegressor(n_jobs=-1, random_state=123)
        scaler = StandardScaler()

        self.model = Pipeline(steps=[
            ("data_scaling", scaler),
            ("model", regressor)
        ])

        self.train_x        = None
        self.train_y        = None
        self.test_x         = None
        self.test_y         = None
        self.test_y_result  = None
        self.train_y_result = None
        self.features       = None

    def preprocess_data(self, filepath, chosen_columns=None, verbose=False):
        '''
        Preprocess a dataset at filepath, split it, and save
        the resulting arrays to this object.
        '''
        if isinstance(filepath, str): 
            if verbose: print("Using file", filepath)
            data = pd.read_csv(filepath)
        elif isinstance(filepath, pd.DataFrame):
            data = filepath.copy()
        if verbose: print("Shape:", data.shape)

        # Convert date to season
        data["date"] = (pd.to_datetime(data["date"]).dt.month-1) % 3

        # Replace NAs
        data.replace("--", pd.NA, inplace=True)
        data.fillna(-999, inplace=True)
        data = data[data["swe_value"]!=-999]

        if chosen_columns is None:
            # Discard non-numeric columns
            non_numeric = data.select_dtypes(exclude=["number"]).columns
            if verbose: print("Dropping non-numeric columns:", non_numeric)
            data = data.drop(columns=non_numeric)
            # Also drop date, lat, lon
            data = data.drop(columns=["date", "lat", "lon"])
        else:
            data = data[chosen_columns]

        X = data.drop("swe_value", axis=1)
        if verbose: print("Using features", X.columns)
        y = data["swe_value"]

        if verbose:
            print("Descriptive statistics")
            print("-- Training data --")
            print(X.describe())
            print("-- Target data --")
            print(y.describe())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.train_x = X_train
        self.train_y = y_train
        self.test_x  = X_test
        self.test_y  = y_test
        self.features = X_train.columns

    def fit(self):
        '''
        Train model and store results.
        '''
        self.model.fit(self.train_x, self.train_y)
        self.train_y_result = self.model.predict(self.train_x)
        self.test_y_result = self.model.predict(self.test_x)

    def evaluate(self):
        '''
        Evaluate model output with MAE, RMSE, and R2.
        '''
        if self.test_y_result is None:
            raise RuntimeError("Model must be trained before evaluating!")

        return {
            "rmse": metrics.root_mean_squared_error(self.test_y, self.test_y_result),
            "mae": metrics.mean_absolute_error(self.test_y, self.test_y_result),
            "r2": metrics.r2_score(self.test_y, self.test_y_result)
        }

    def predict(self, input_x):
        '''
        Run model on new data.
        '''
        return self.model.predict(input_x)

def compare_spatial_output(df, outputs, lat_column="lat", lon_column="lon",
                          fig_args=dict(), plot_args=dict()):
    '''
    Simultaenously plot multiple variables in a dataframe that also contains
    spatial coordinates.
    '''
    df_as_xr = df.set_index([lat_column, lon_column]).to_xarray()
    fig, axes = plt.subplots(nrows=len(outputs), ncols=1, figsize=(5, 3*len(outputs)), **fig_args)
    for i in range(len(outputs)):
        df_as_xr[outputs[i]].plot(ax=axes[i], **plot_args)
    plt.show()