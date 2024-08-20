# compare patterns in training and testing
# plot the comparison of training and testing variables

# This process only analyzes data; we don't touch the model here.

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from snowcast_utils import work_dir, test_start_date
import os
import pandas as pd
import matplotlib.pyplot as plt

def clean_train_df(data):
    """
    Clean and preprocess the training data.

    Args:
        data (pd.DataFrame): The training data to be cleaned.

    Returns:
        pd.DataFrame: Cleaned training data.
    """
    data['date'] = pd.to_datetime(data['date'])
    reference_date = pd.to_datetime('1900-01-01')
    data['date'] = (data['date'] - reference_date).dt.days
    data.replace('--', pd.NA, inplace=True)
    data.fillna(-999, inplace=True)
    
    # Remove all the rows that have 'swe_value' as -999
    data = data[(data['swe_value'] != -999)]

    print("Get slope statistics")
    print(data["slope"].describe())
  
    print("Get SWE statistics")
    print(data["swe_value"].describe())

    data = data.drop('Unnamed: 0', axis=1)
    

    return data

def compare():
    """
    Compare training and testing data and create variable comparison plots.

    Returns:
        None
    """
    new_testing_data_path = f'{work_dir}/testing_all_ready_for_check.csv'
    training_data_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'

    tr_df = pd.read_csv(training_data_path)
    tr_df = clean_train_df(tr_df)
    te_df = pd.read_csv(new_testing_data_path)
    
    #tr_df = tr_df.drop('date', axis=1)
    #te_df = te_df.drop('date', axis=1)

    print("Training DataFrame: ", tr_df)
    print("Testing DataFrame: ", te_df)
    
    te_df = te_df.apply(pd.to_numeric, errors='coerce')
    print("te_df describe: ", te_df.describe())

    print("Training columns: ", tr_df.columns)
    print("Testing columns: ", te_df.columns)

    var_comparison_plot_path = f"{work_dir}/var_comparison/"
    if not os.path.exists(var_comparison_plot_path):
        os.makedirs(var_comparison_plot_path)
        
    num_cols = len(tr_df.columns)
    new_num_cols = int(num_cols**0.5)  # Square grid
    new_num_rows = int(num_cols / new_num_cols) + 1
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(new_num_rows, new_num_cols, figsize=(24, 20))
    
    # Flatten the axs array to iterate through subplots
    axs = axs.flatten()
    print("length: ", len(tr_df.columns))
    # Iterate over columns and create subplots
    for i, col in enumerate(tr_df.columns):
        print(i, " - ", col)
        axs[i].hist(tr_df[col], bins=100, alpha=0.5, color='blue', label='Train')
        if col in te_df.columns:
            axs[i].hist(te_df[col], bins=100, alpha=0.5, color='red', label='Test')
        else:
          print(f"Error: {col} is not in testing csv")

        axs[i].set_title(f'{col}')
        axs[i].legend()
        
    
    
    plt.tight_layout()
    plt.savefig(f'{var_comparison_plot_path}/{test_start_date}_final_comparison.png')
    plt.close()

def calculate_feature_colleration_in_training():
  training_data_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
  tr_df = pd.read_csv(training_data_path)
  tr_df = clean_train_df(tr_df)
  
    
compare()


