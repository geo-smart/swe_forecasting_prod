'''
The wrapper for all the snowcast_wormhole predictors.
'''

import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

class BaseHole:
    '''
    Base class for snowcast_wormhole predictors.

    Attributes:
        all_ready_file (str): The path to the CSV file containing the data for training.
        classifier: The machine learning model used for prediction.
        holename (str): The name of the wormhole class.
        train_x (numpy.ndarray): The training input data.
        train_y (numpy.ndarray): The training target data.
        test_x (numpy.ndarray): The testing input data.
        test_y (numpy.ndarray): The testing target data.
        test_y_results (numpy.ndarray): The predicted results on the test data.
        save_file (str): The path to save the trained model.
    '''

    all_ready_file = f"{github_dir}/data/ready_for_training/all_ready_new.csv"

    def __init__(self):
        '''
        Initializes a new instance of the BaseHole class.
        '''
        self.classifier = self.get_model()
        self.holename = self.__class__.__name__ 
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.test_y_results = None
        self.save_file = None
    
    def save(self):
        '''
        Save the trained model to a joblib file with a timestamp.

        Returns:
            None
        '''
        now = datetime.now()
        date_time = now.strftime("%Y%d%m%H%M%S")
        self.save_file = f"{github_dir}/model/wormhole_{self.holename}_{date_time}.joblib"
        
        directory = os.path.dirname(self.save_file)
        if not os.path.exists(directory):
          os.makedirs(directory)
        
        print(f"Saving model to {self.save_file}")
        joblib.dump(self.classifier, self.save_file)
        # copy a version to the latest file placeholder
        latest_copy_file = f"{github_dir}/model/wormhole_{self.holename}_latest.joblib"
        shutil.copy(self.save_file, latest_copy_file)
        print(f"a copy of the model is saved to {latest_copy_file}")
  
    def preprocessing(self):
        '''
        Preprocesses the data for training and testing.

        Returns:
            None
        '''
        all_ready_pd = pd.read_csv(self.all_ready_file, header=0, index_col=0)
        print("all columns: ", all_ready_pd.columns)
        all_ready_pd = all_ready_pd[all_cols]
        all_ready_pd = all_ready_pd.dropna()
        train, test = train_test_split(all_ready_pd, test_size=0.2)
        self.train_x, self.train_y = train[input_columns].to_numpy().astype('float'), train[['swe_value']].to_numpy().astype('float')
        self.test_x, self.test_y = test[input_columns].to_numpy().astype('float'), test[['swe_value']].to_numpy().astype('float')
  
    def train(self):
        '''
        Trains the machine learning model.

        Returns:
            None
        '''
        self.classifier.fit(self.train_x, self.train_y)
  
    def test(self):
        '''
        Tests the machine learning model on the testing data.

        Returns:
            numpy.ndarray: The predicted results on the testing data.
        '''
        self.test_y_results = self.classifier.predict(self.test_x)
        return self.test_y_results
  
    def predict(self, input_x):
        '''
        Makes predictions using the trained model on new input data.

        Args:
            input_x (numpy.ndarray): The input data for prediction.

        Returns:
            numpy.ndarray: The predicted results.
        '''
        return self.classifier.predict(input_x)
  
    def evaluate(self):
        '''
        Evaluates the performance of the machine learning model.

        Returns:
            None
        '''
        pass
  
    def get_model(self):
        '''
        Get the machine learning model.

        Returns:
            object: The machine learning model.
        '''
        pass
  
    def post_processing(self):
        '''
        Perform post-processing on the model's predictions.

        Returns:
            None
        '''
        pass

