import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from snowcast_utils import work_dir, month_to_season
import os

def load_model(model_path):
    """
    Load a machine learning model from a file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model.
    """
    return joblib.load(model_path)

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the input data for model prediction.

    Args:
        data (pd.DataFrame): Input data in the form of a pandas DataFrame.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    data['date'] = pd.to_datetime(data['date'])
    #data['date'] = data['date'].dt.strftime('%j').astype(int)
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
                         'AMSR_SWE': 'SWE',
                         'AMSR_Flag': 'Flag',
                         'Elevation': 'elevation',
                         'Slope': 'slope',
                         'Aspect': 'aspect',
                         'Curvature': 'curvature',
                         'Northness': 'northness',
                         'Eastness': 'eastness'
                        }, inplace=True)

    desired_order = ['date', 'lat', 'lon', 'SWE', 'Flag',
                     'air_temperature_tmmn', 'potential_evapotranspiration',
                     'mean_vapor_pressure_deficit', 'relative_humidity_rmax',
                     'relative_humidity_rmin', 'precipitation_amount',
                     'air_temperature_tmmx', 'wind_speed', 'elevation', 'slope', 'curvature',
                     'aspect', 'eastness', 'northness']
    
    data = data[desired_order]
    data = data.reindex(columns=desired_order)
    
    return data

def predict_swe(model, data):
    """
    Predict snow water equivalent (SWE) using a machine learning model.

    Args:
        model: The machine learning model for prediction.
        data (pd.DataFrame): Input data for prediction.

    Returns:
        pd.DataFrame: Dataframe with predicted SWE values.
    """
    data = data.fillna(-999)
    input_data = data
    #input_data = data.drop(['date', 'SWE', 'Flag', 'mean_vapor_pressure_deficit', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 'relative_humidity_rmin',], axis=1)
    predictions = model.predict(input_data)
    data['predicted_swe'] = predictions
    return data

def merge_data(original_data, predicted_data):
    """
    Merge predicted SWE data with the original data.

    Args:
        original_data (pd.DataFrame): Original input data.
        predicted_data (pd.DataFrame): Dataframe with predicted SWE values.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    #new_data_extracted = predicted_data[["date", "lat", "lon", "predicted_swe"]]
    new_data_extracted = predicted_data[["lat", "lon", "predicted_swe"]]
    #merged_df = original_data.merge(new_data_extracted, on=["date", 'lat', 'lon'], how='left')
    merged_df = original_data.merge(new_data_extracted, on=['lat', 'lon'], how='left')
    return merged_df

def predict():
    """
    Main function for predicting snow water equivalent (SWE).

    Returns:
        None
    """
    height = 666
    width = 694
    model_path = f'/home/chetana/Documents/GitHub/SnowCast/model/wormhole_ETHole_latest.joblib'
    print(f"Using model: {model_path}")
  
    new_data_path = f'{work_dir}/testing_all_ready.csv'
    output_path = f'{work_dir}/test_data_predicted.csv'
  
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"File '{output_path}' has been removed.")

    model = load_model(model_path)
    new_data = load_data(new_data_path)
    print("new_data shape: ", new_data.shape)

    preprocessed_data = preprocess_data(new_data)
    print('Data preprocessing completed.')
    print(f'Model used: {model_path}')
    predicted_data = predict_swe(model, preprocessed_data)
    predicted_data = merge_data(preprocessed_data, predicted_data)
    print('Data prediction completed.')
  
    predicted_data.to_csv(output_path, index=False)
    print("Prediction successfully done ", output_path)

    if len(predicted_data) == height * width:
        print(f"The image width, height match with the number of rows in the CSV. {len(predicted_data)} rows")
    else:
        raise Exception("The total number of rows does not match")

predict()

