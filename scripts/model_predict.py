import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from snowcast_utils import homedir, work_dir, month_to_season, test_start_date
import os
import random
import string
import shutil
from model_creation_et import selected_columns

def generate_random_string(length):
    # Define the characters that can be used in the random string
    characters = string.ascii_letters + string.digits  # You can customize this to include other characters if needed

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string
  

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

def preprocess_data(data, is_model_input: bool = True):
    """
    Preprocess the input data for model prediction.

    Args:
        data (pd.DataFrame): Input data in the form of a pandas DataFrame.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    
    #print("check date format: ", data.head())
    #data['date'] = data['date'].dt.strftime('%j').astype(int)
    #data['date'] = data['date'].dt.month.apply(month_to_season)
    data.replace('--', pd.NA, inplace=True)
    
    
    #data = data.apply(pd.to_numeric, errors='coerce')

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
#                          'Elevation': 'elevation',
#                          'Slope': 'Slope',
#                          'Aspect': 'Aspect',
#                          'Curvature': 'Curvature',
#                          'Northness': 'Northness',
#                          'Eastness': 'Eastness',
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
#                          'relative_humidity_rmin': '',
#                          'cumulative_rmin',
#                          'mean_vapor_pressure_deficit', 
#                          'cumulative_vpd', 
#                          'wind_speed',
#                          'cumulative_vs', 
#                          'relative_humidity_rmax', 'cumulative_rmax',

# 'precipitation_amount', 'cumulative_pr', 'air_temperature_tmmx',

# 'cumulative_tmmx', 'potential_evapotranspiration', 'cumulative_etr',

# 'air_temperature_tmmn', 'cumulative_tmmn', 'x', 'y', 'elevation',

# 'slope', 'aspect', 'curvature', 'northness', 'eastness', 'AMSR_SWE',

# 'cumulative_AMSR_SWE', 'AMSR_Flag', 'cumulative_AMSR_Flag',
                        }, inplace=True)

    print(data.head())
    print(data.columns)
    
    # filter out three days for final visualization to accelerate the process
    #dates_to_match = ['2018-03-15', '2018-04-15', '2018-05-15']
    #mask = data['date'].dt.strftime('%Y-%m-%d').isin(dates_to_match)
    # Filter the DataFrame based on the mask
    #data = data[mask]
    if is_model_input:
        data['date'] = pd.to_datetime(data['date'])
        selected_columns.remove("swe_value")
        desired_order = selected_columns + ['lat', 'lon',]
        
        data = data[desired_order]
        data = data.reindex(columns=desired_order)
        
        print("reorganized columns: ", data.columns)
    
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
    input_data = data.drop(["lat", "lon"], axis=1)
    #input_data = data.drop(['date', 'SWE', 'Flag', 'mean_vapor_pressure_deficit', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 'relative_humidity_rmin',], axis=1)
    #scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing data
    #input_data_scaled = scaler.fit_transform(input_data)
    
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
    if "date" not in predicted_data:
    	predicted_data["date"] = test_start_date
    new_data_extracted = predicted_data[["date", "lat", "lon", "predicted_swe"]]
    print("original_data.columns: ", original_data.columns)
    print("new_data_extracted.columns: ", new_data_extracted.columns)
    print("new prediction statistics: ", new_data_extracted["predicted_swe"].describe())
    #merged_df = original_data.merge(new_data_extracted, on=["date", 'lat', 'lon'], how='left')
    merged_df = original_data.merge(new_data_extracted, on=['lat', 'lon'], how='left')
    print("first merged df: ", merged_df.columns)

    merged_df.loc[merged_df['fsca'] == 237, 'predicted_swe'] = 0
    merged_df.loc[merged_df['fsca'] == 239, 'predicted_swe'] = 0
    merged_df.loc[merged_df['fsca'] == 225, 'predicted_swe'] = 0
    #merged_df.loc[merged_df['cumulative_fsca'] == 0, 'predicted_swe'] = 0
    merged_df.loc[merged_df['fsca'] == 0, 'predicted_swe'] = 0
    
    merged_df.loc[merged_df['air_temperature_tmmx'].isnull(), 
                  'predicted_swe'] = 0

    merged_df.loc[merged_df['lc_prop3'] == 3, 'predicted_swe'] = 0
    merged_df.loc[merged_df['lc_prop3'] == 255, 'predicted_swe'] = 0
    merged_df.loc[merged_df['lc_prop3'] == 27, 'predicted_swe'] = 0

    return merged_df

def predict():
    """
    Main function for predicting snow water equivalent (SWE).

    Returns:
        None
    """
    height = 666
    width = 694
    model_path = f'{homedir}/Documents/GitHub/SnowCast/model/wormhole_ETHole_latest.joblib'
    print(f"Using model: {model_path}")
  
    new_data_path = f'{work_dir}/testing_all_ready_{test_start_date}.csv'
    #output_path = f'{work_dir}/test_data_predicted_three_days_only.csv'
    latest_output_path = f'{work_dir}/test_data_predicted_latest_{test_start_date}.csv'
    output_path = f'{work_dir}/test_data_predicted_{generate_random_string(5)}.csv'
  
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"File '{output_path}' has been removed.")

    model = load_model(model_path)
    print(f"loading {new_data_path}")
    new_data = load_data(new_data_path)
    new_data = new_data.drop(["date.1",], axis=1)
    print("new_data.columns: ", new_data.columns)

    preprocessed_data = preprocess_data(new_data, is_model_input=True)
    if len(new_data) < len(preprocessed_data):
      raise ValueError("Why the preprocessed data increased?")
    #print('Data preprocessing completed.', preprocessed_data.head())
    #print(f'Model used: {model_path}')
    predicted_data = predict_swe(model, preprocessed_data)
    print("how many predicted? ", len(predicted_data))
    
    if "date" not in preprocessed_data:
    	preprocessed_data["date"] = test_start_date

    full_preprocessed_data = preprocess_data(new_data, is_model_input=False)
    predicted_data = merge_data(full_preprocessed_data, predicted_data)
    
    
    #print('Data prediction completed.')
  
    #print(predicted_data['date'])
    predicted_data.to_csv(output_path, index=False)
    print("Prediction successfully done ", output_path)
    
    shutil.copy(output_path, latest_output_path)
    print(f"Copied to {latest_output_path}")

#     if len(predicted_data) == height * width:
#         print(f"The image width, height match with the number of rows in the CSV. {len(predicted_data)} rows")
#     else:
#         raise Exception("The total number of rows does not match")

if __name__ == "__main__":
	predict()

