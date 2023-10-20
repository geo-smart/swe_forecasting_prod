"""
This script trains and validates machine learning models for hole analysis.

Attributes:
    RandomForestHole (class): A class for training and using a Random Forest model.
    XGBoostHole (class): A class for training and using an XGBoost model.
    ETHole (class): A class for training and using an Extra Trees model.

Functions:
    main(): The main function that trains and validates machine learning models for hole analysis.
"""

from model_creation_rf import RandomForestHole
from model_creation_xgboost import XGBoostHole
from model_creation_et import ETHole

def main():
    print("Train Models")

    # Choose the machine learning models to train (e.g., RandomForestHole, XGBoostHole, ETHole)
    worm_holes = [ETHole()]

    for hole in worm_holes:
        # Perform preprocessing for the selected model
        hole.preprocessing()
        print(hole.train_x.shape)
        print(hole.train_y.shape)
        
        # Train the machine learning model
        hole.train()
        
        # Test the trained model
        hole.test()
        
        # Evaluate the model's performance
        hole.evaluate()
        
        # Save the trained model
        hole.save()

    print("Finished training and validating all the models.")

if __name__ == "__main__":
    main()

