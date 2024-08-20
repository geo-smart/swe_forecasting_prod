"""
This script defines the ETHole class, which is used for training and evaluating an Extra Trees Regressor model for SWE prediction.

Attributes:
    ETHole (class): A class for training and using an Extra Trees Regressor model for SWE prediction.

Functions:
    custom_loss(y_true, y_pred): A custom loss function that penalizes errors for values greater than 10.
    get_model(): Returns the Extra Trees Regressor model with specified hyperparameters.
    create_sample_weights(y, scale_factor): Creates sample weights based on target values and a scaling factor.
    preprocessing(): Preprocesses the training data, including data cleaning and feature extraction.
    train(): Trains the Extra Trees Regressor model.
    post_processing(): Performs post-processing, including feature importance analysis and visualization.
"""

import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from model_creation_rf import RandomForestHole
from snowcast_utils import work_dir, month_to_season
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight


working_dir = work_dir

class ETHole(RandomForestHole):
  
    def custom_loss(y_true, y_pred):
        """
        A custom loss function that penalizes errors for values greater than 10.

        Args:
            y_true (numpy.ndarray): True target values.
            y_pred (numpy.ndarray): Predicted target values.

        Returns:
            numpy.ndarray: Custom loss values.
        """
        errors = np.abs(y_true - y_pred)
        
        return np.where(y_true > 10, 2 * errors, errors)

    def get_model(self):
        """
        Returns the Extra Trees Regressor model with specified hyperparameters.

        Returns:
            ExtraTreesRegressor: The Extra Trees Regressor model.
        """
#         return ExtraTreesRegressor(n_estimators=200, 
#                                    max_depth=None,
#                                    random_state=42, 
#                                    min_samples_split=2,
#                                    min_samples_leaf=1,
#                                    n_jobs=5
#                                   )
        return ExtraTreesRegressor(n_jobs=-1, random_state=123)

    def create_sample_weights(self, X, y, scale_factor, columns):
        """
        Creates sample weights based on target values and a scaling factor.

        Args:
            y (numpy.ndarray): Target values.
            scale_factor (float): Scaling factor for sample weights.

        Returns:
            numpy.ndarray: Sample weights.
        """
        #return np.where(X["fsca"] < 100, scale_factor, 1)
        return (y - np.min(y)) / (np.max(y) - np.min(y)) * scale_factor
        # Create a weight vector to assign weights to features - this is not a good idea
#         feature_weights = {'date': 0.1, 'SWE': 1.5, 'wind_speed': 1.5, 'precipitation_amount': 2.0}
#         default_weight = 1.0

#         # Create an array of sample weights based on feature_weights
#         sample_weights = np.array([feature_weights.get(feature, default_weight) for feature in columns])
        #return sample_weights

      
    def preprocessing(self, chosen_columns=None):
        """
        Preprocesses the training data, including data cleaning and feature extraction.
        """
        #training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned.csv'
        #training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned_v3.csv'
        #training_data_path = f'{working_dir}/all_merged_training_cum_water_year_winter_month_only.csv' # snotel points
#         training_data_path = f'{working_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
        training_data_path = f"{working_dir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv"
        
        print("preparing training data from csv: ", training_data_path)
        data = pd.read_csv(training_data_path)
        print("data.shape = ", data.shape)
        print(data.head())
        
        data['date'] = pd.to_datetime(data['date'])
        #reference_date = pd.to_datetime('1900-01-01')
        #data['date'] = (data['date'] - reference_date).dt.days
        # just use julian day
        #data['date'] = data['date'].dt.strftime('%j').astype(int)
        # just use the season to reduce the bias on month or dates
        data['date'] = data['date'].dt.month.apply(month_to_season)
        
        data.replace('--', pd.NA, inplace=True)
        data.fillna(-999, inplace=True)
        
        data = data[(data['swe_value'] != -999)]
        
        if chosen_columns == None:
#           data = data.drop('Unnamed: 0', axis=1)
          non_numeric_columns = data.select_dtypes(exclude=['number']).columns
          # Drop non-numeric columns
          data = data.drop(columns=non_numeric_columns)
          print("all non-numeric columns are dropped: ", non_numeric_columns)
          #data = data.drop('level_0', axis=1)
          data = data.drop(['date'], axis=1)
          data = data.drop(['lat'], axis=1)
          data = data.drop(['lon'], axis=1)
        else:
          data = data[chosen_columns]
#         (['lat', 'lon', 'SWE', 'Flag', 'air_temperature_tmmn',
# 'potential_evapotranspiration', 'mean_vapor_pressure_deficit',
# 'relative_humidity_rmax', 'relative_humidity_rmin',
# 'precipitation_amount', 'air_temperature_tmmx', 'wind_speed',
# 'elevation', 'slope', 'curvature', 'aspect', 'eastness', 'northness']
        
        
        X = data.drop('swe_value', axis=1)
        print('required features:', X.columns)
        y = data['swe_value']
        print("describe the statistics of training input: ", X.describe())
        print("describe the statistics of swe_value: ", y.describe())
        
        print("input features and order: ", X.columns)
        print("training data row number: ", len(X))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize the StandardScaler
        #scaler = StandardScaler()

        # Fit the scaler on the training data and transform both training and testing data
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
        
        self.weights = self.create_sample_weights(X_train, y_train, scale_factor=10, columns=X.columns)

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        #self.train_x, self.train_y = X_train_scaled, y_train
        #self.test_x, self.test_y = X_test_scaled, y_test
        self.feature_names = X_train.columns
        
    def train(self):
        """
        Trains the Extra Trees Regressor model.
        """
        # Calculate sample weights based on errors (you may need to customize this)
#         self.classifier.fit(self.train_x, self.train_y)
#         errors = abs(self.train_y - self.classifier.predict(self.train_x))
#         print(errors)
        
        #self.weights = 1+self.train_y # You can adjust this formula as needed
#         weights = np.zeros_like(self.train_y, dtype=float)

#         # Set weight to 1 if the target variable is 0
#         weights[self.train_y == 0] = 10.0

#         # Calculate weights for non-zero target values
#         non_zero_indices = self.train_y != 0
#         weights[non_zero_indices] = 0.1 / np.abs(self.train_y[non_zero_indices])

#         self.classifier.fit(self.train_x, self.train_y, sample_weight=self.weights)
#         self.classifier.fit(self.train_x, self.train_y)
        
#         errors = abs(self.train_y - self.classifier.predict(self.train_x))
#         self.weights = 1 / (1 + errors)  # You can adjust this formula as needed
#         self.classifier.fit(self.train_x, self.train_y, sample_weight=self.weights)

        # Fit the classifier
        self.classifier.fit(self.train_x, self.train_y)

        # Make predictions
        predictions = self.classifier.predict(self.train_x)

        # Calculate absolute errors
        errors = np.abs(self.train_y - predictions)

        # Assign weights based on errors (higher errors get higher weights)
        weights = compute_sample_weight('balanced', errors)
        self.classifier.fit(self.train_x, self.train_y, sample_weight=weights)

    def post_processing(self, chosen_columns=None):
        """
        Performs post-processing, including feature importance analysis and visualization.
        """
        feature_importances = self.classifier.feature_importances_
        feature_names = self.feature_names
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_names)), sorted_importances, tick_label=sorted_feature_names)
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance Plot (ET model)')
        plt.tight_layout()
        if chosen_columns == None:
          feature_png = f'{work_dir}/testing_output/et-model-feature-importance-latest.png'
        else:
          feature_png = f'{work_dir}/testing_output/et-model-feature-importance-{len(chosen_columns)}.png'
        plt.savefig(feature_png)
        print(f"Feature image is saved {feature_png}")

# Instantiate ETHole class and perform tasks

# all_used_columns = ['station_elevation', 'elevation', 'aspect', 'curvature', 'slope',
# 'eastness', 'northness', 'etr', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx',
# 'vpd', 'vs', 'lc_code',  'fSCA',  'cumulative_etr',
# 'cumulative_rmax', 'cumulative_rmin', 'cumulative_tmmn',
# 'cumulative_tmmx', 'cumulative_vpd', 'cumulative_vs', 'cumulative_pr', 'swe_value']

#all_used_columns = ['cumulative_pr','station_elevation', 'cumulative_tmmn', 'cumulative_tmmx', 'northness', 'cumulative_vs', 'cumulative_rmax', 'cumulative_etr','aspect','cumulative_rmin', 'elevation', 'cumulative_vpd',  'swe_value']
# selected_columns = ["lat","lon","elevation","slope","curvature","aspect","eastness","northness","cumulative_SWE","cumulative_Flag","cumulative_air_temperature_tmmn","cumulative_potential_evapotranspiration","cumulative_mean_vapor_pressure_deficit","cumulative_relative_humidity_rmax","cumulative_relative_humidity_rmin","cumulative_precipitation_amount","cumulative_air_temperature_tmmx","cumulative_wind_speed", "swe_value"]

# all current variables without time series
# selected_columns = ['SWE', 'Flag', 'air_temperature_tmmn', 'potential_evapotranspiration',
# 'mean_vapor_pressure_deficit', 'relative_humidity_rmax',
# 'relative_humidity_rmin', 'precipitation_amount',
# 'air_temperature_tmmx', 'wind_speed', 'elevation', 'slope', 'curvature',
# 'aspect', 'eastness', 'northness', 'cumulative_SWE',
# 'cumulative_Flag', 'cumulative_air_temperature_tmmn',
# 'cumulative_potential_evapotranspiration',
# 'cumulative_mean_vapor_pressure_deficit',
# 'cumulative_relative_humidity_rmax',
# 'cumulative_relative_humidity_rmin', 'cumulative_precipitation_amount',
# 'cumulative_air_temperature_tmmx', 'cumulative_wind_speed', 'swe_value']


selected_columns = [
  'swe_value',
  'SWE',
  'cumulative_SWE',
#   'cumulative_relative_humidity_rmin',
#   'cumulative_air_temperature_tmmx', 
#   'cumulative_air_temperature_tmmn',
#   'cumulative_relative_humidity_rmax',
#   'cumulative_potential_evapotranspiration',
#   'cumulative_wind_speed',
  #'cumulative_fsca',
  'fsca',
  'air_temperature_tmmx', 
  'air_temperature_tmmn', 
  'potential_evapotranspiration', 
  'relative_humidity_rmax', 
  'Elevation',	
  'Slope',	
  'Curvature',	
  'Aspect',	
  'Eastness',	
  'Northness',
]

# ['cumulative_relative_humidity_rmin', 'cumulative_air_temperature_tmmx', 'cumulative_air_temperature_tmmn', 'cumulative_relative_humidity_rmax', 'cumulative_potential_evapotranspiration', 'cumulative_wind_speed'] 

if __name__ == "__main__":
  hole = ETHole()
  hole.preprocessing(chosen_columns = selected_columns)
#   hole.preprocessing()
  hole.train()
  hole.test()
  hole.evaluate()
  hole.save()
  hole.post_processing(chosen_columns = selected_columns)
#   hole.post_processing()


