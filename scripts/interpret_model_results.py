# do real interpretation for the model results and find real reasons for bad predictions
# prevent aimless and headless attempts that are just wasting time.
# this is an essential step in the loop

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from snowcast_utils import work_dir, test_start_date, month_to_season
import os
from sklearn.inspection import partial_dependence,PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt

feature_names = None

def load_model(model_path):
    """
    Load a trained machine learning model from a given path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        object: The loaded machine learning model.
    """
    return joblib.load(model_path)

def load_data(file_path):
    """
    Load data from a CSV file and return it as a DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the input data by converting date columns, handling missing values,
    renaming columns, and reordering columns.

    Args:
        data (pd.DataFrame): The input data to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data['date'] = pd.to_datetime(data['date'])
    #reference_date = pd.to_datetime('1900-01-01')
    #data['date'] = (data['date'] - reference_date).dt.days
    data['date'] = data['date'].dt.month.apply(month_to_season)
    data.replace('--', pd.NA, inplace=True)
    
    data = data.apply(pd.to_numeric, errors='coerce')
    data.rename(columns={'Latitude': 'lat', 
                         'Longitude': 'lon',
                         'vpd': 'mean_vapor_pressure_deficit',
                         'vs': 'wind_speed', 
                         'pr': 'precipitation_amount', 
                         'etr': 'potential_evapotranspiration',
                         'tmmn': 'air_temperature_tmmn',
                         'tmmx': 'air_temperature_tmmx',
                         'rmin': 'relative_humidity_rmin',
                         'rmax': 'relative_humidity_rmax',
                         'Elevation': 'elevation',
                         'Slope': 'slope',
                         'Aspect': 'aspect',
                         'Curvature': 'curvature',
                         'Northness': 'northness',
                         'Eastness': 'eastness',
                         'cumulative_AMSR_SWE': 'cumulative_SWE',
                         'cumulative_AMSR_Flag': 'cumulative_Flag',
                         'cumulative_tmmn':'cumulative_air_temperature_tmmn',
                         'cumulative_etr': 'cumulative_potential_evapotranspiration',
                         'cumulative_vpd': 'cumulative_mean_vapor_pressure_deficit',
                         'cumulative_rmax': 'cumulative_relative_humidity_rmax', 
                         'cumulative_rmin': 'cumulative_relative_humidity_rmin',
                         'cumulative_pr': 'cumulative_precipitation_amount',
                         'cumulative_tmmx': 'cumulative_air_temperature_tmmx',
                         'cumulative_vs': 'cumulative_wind_speed',
                         'AMSR_SWE': 'SWE',
                         'AMSR_Flag': 'Flag',
                        }, inplace=True)

    desired_order = ['lat', 'lon', 'SWE', 'Flag', 'air_temperature_tmmn', 'potential_evapotranspiration',
'mean_vapor_pressure_deficit', 'relative_humidity_rmax',
'relative_humidity_rmin', 'precipitation_amount',
'air_temperature_tmmx', 'wind_speed', 'elevation', 'slope', 'curvature',
'aspect', 'eastness', 'northness', 'cumulative_SWE',
'cumulative_Flag', 'cumulative_air_temperature_tmmn',
'cumulative_potential_evapotranspiration',
'cumulative_mean_vapor_pressure_deficit',
'cumulative_relative_humidity_rmax',
'cumulative_relative_humidity_rmin', 'cumulative_precipitation_amount',
'cumulative_air_temperature_tmmx', 'cumulative_wind_speed']
    
    feature_names = desired_order
    
    data = data[desired_order]
    data = data.reindex(columns=desired_order)
    
    data.to_csv(f'{work_dir}/testing_all_ready_for_check.csv', index=False)
    
    data = data.fillna(-999)
    print("how many rows are left?", len(data))
    print('data.shape: ', data.shape)
    
    #data = data.drop(['date', 'SWE', 'Flag', 'mean_vapor_pressure_deficit', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 'relative_humidity_rmin', ], axis=1)
    data = data.drop(['lat', 'lon',], axis=1)
    
    return data

def predict_swe(model, data):
    """
    Use a trained model to predict SWE values for the input data.

    Args:
        model (object): The trained machine learning model.
        data (pd.DataFrame): The input data for prediction.

    Returns:
        pd.DataFrame: Input data with predicted SWE values.
    """
    print(data.head())
    print("how many rows are there?", len(data))
    
    predictions = model.predict(data)
    data['predicted_swe'] = predictions
    print("predicted swe: ", data['predicted_swe'].describe())
    
    return data, model

def merge_data(original_data, predicted_data):
    """
    Merge the original data with predicted SWE values.

    Args:
        original_data (pd.DataFrame): The original data.
        predicted_data (pd.DataFrame): Data with predicted SWE values.

    Returns:
        pd.DataFrame: Merged data.
    """
    #new_data_extracted = predicted_data[["date", "lat", "lon", "predicted_swe"]]
    new_data_extracted = predicted_data[[ "lat", "lon", "predicted_swe"]]
    # merged_df = original_data.merge(new_data_extracted, on=["date", 'lat', 'lon'], how='left')
    merged_df = original_data.merge(new_data_extracted, on=['lat', 'lon'], how='left')
    print("Columns after merge:", merged_df.columns)
    
    return merged_df
  
def plot_feature_importance():
    model_path = f'/home/chetana/Documents/GitHub/SnowCast/model/wormhole_ETHole_latest.joblib'
    model = load_model(model_path)
    
    training_data_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
    print("preparing training data from csv: ", training_data_path)
    data = pd.read_csv(training_data_path)
    data = data.drop('swe_value', axis=1) 
    data = data.drop('Unnamed: 0', axis=1)
    
    
    # Step 1: Feature Importance
    analysis_plot_output_folder = f'{work_dir}/testing_output/'
    feature_importances = model.feature_importances_
    feature_names = data.columns

    # Create a bar plot of feature importances
    plt.figure(figsize=(10, 6))
    print(feature_names.shape)
    print(feature_importances.shape)
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    plt.savefig(f'{analysis_plot_output_folder}/importance_summary_plot_latest_model.png')
  
def interpret_prediction():
    """
    Interpret the model results and find real reasons for bad predictions.

    Returns:
        None
    """
    height = 666
    width = 694
    model_path = f'/home/chetana/Documents/GitHub/SnowCast/model/wormhole_ETHole_latest.joblib'
    print(f"using model : {model_path}")
    
    new_data_path = f'{work_dir}/testing_all_ready.csv'
    output_path = f'{work_dir}/test_data_predicted.csv'
    
    if os.path.exists(output_path):
        # If the file exists, remove it
        os.remove(output_path)
        print(f"File '{output_path}' has been removed.")

    model = load_model(model_path)
    new_data = load_data(new_data_path)
    print("new_data shape: ", new_data.shape)

    preprocessed_data = preprocess_data(new_data)
    print('data preprocessing completed.')
    print(f'model used: {model_path}')
    predicted_data, current_model = predict_swe(model, preprocessed_data)
    
    

    # Step 2: Partial Dependence Plots
    # Select features for partial dependence plots (e.g., the first two features)
    features_to_plot = feature_names
    
#     partial_dependence_display = PartialDependenceDisplay.from_estimator(
#       current_model, 
#       preprocessed_data.drop('predicted_swe', axis=1), 
#       features=features_to_plot, grid_resolution=50
# )
#     partial_dependence_display.figure_.suptitle('Partial Dependence Plots')
#     partial_dependence_display.figure_.subplots_adjust(top=0.9)
#     partial_dependence_display.figure_.savefig(f'{analysis_plot_output_folder}/partial_dependence_summary_plot_{test_start_date}.png')

    # Step 3: SHAP Values
#     explainer = shap.Explainer(current_model)
#     shap_values = explainer.shap_values(
#       preprocessed_data.drop('predicted_swe', axis=1))

#     # Summary plot of SHAP values
#     shap.summary_plot(shap_values, preprocessed_data)
#     plt.title('Summary Plot of SHAP Values')
#     plt.savefig(f'{analysis_plot_output_folder}/shap_summary_plot_{test_start_date}.png')
    
    # Additional code for SHAP interpretation can be added here
    # Select a single data point for which you want to explain the prediction
    # Create a SHAP explainer and calculate SHAP values
    # Visualize the SHAP values as needed
    
#   predicted_data = merge_data(preprocessed_data, predicted_data)
#   print('data prediction completed.')
    
#   predicted_data.to_csv(output_path, index=False)
#   print("Prediction successfully done ", output_path)

#   if len(predicted_data) == height * width:
#     print(f"The image width, height match with the number of rows in the csv. {len(predicted_data)} rows")
#   else:
#     raise Exception("The total number of rows do not match")


#plot_feature_importance()  # no need, this step is already done in the model post processing step. 
interpret_prediction()

