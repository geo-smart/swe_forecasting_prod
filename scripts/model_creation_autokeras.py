import pandas as pd
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from snowcast_utils import work_dir, month_to_season
from datetime import datetime
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import optuna
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor


working_dir = work_dir

homedir = os.path.expanduser('~')
now = datetime.now()
date_time = now.strftime("%Y%d%m%H%M%S")

github_dir = f"{homedir}/Documents/GitHub/SnowCast"
training_data_path = f"{working_dir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv"

model_save_file = f"{github_dir}/model/wormhole_autokeras_{date_time}.joblib"

# Read the data from the CSV file
print(f"start to read data {training_data_path}")
df = pd.read_csv(training_data_path)
#df.dropna(inplace=True)

print(df.head())
print(df.columns())

# Load and prepare your data
# data = pd.read_csv('your_dataset.csv')  # Replace with your actual data source
X = df.drop(columns=['swe_value', 'Date'])  # Replace with your actual target column
y = df['swe_value']


# Define the function to create a diverse Keras model
def create_model(trial):
    input_shape = X_train.shape[1]
    model_type = trial.suggest_categorical('model_type', ['dense', 'cnn', 'lstm', 'transformer', 'tabnet'])

    if model_type == 'dense':
        model = Sequential()
        num_layers = trial.suggest_int('num_layers', 1, 5)
        for i in range(num_layers):
            num_units = trial.suggest_int(f'num_units_l{i}', 16, 128, log=True)
            if i == 0:
                model.add(Dense(num_units, activation='relu', input_shape=(input_shape,)))
            else:
                model.add(Dense(num_units, activation='relu'))
            dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.1, 0.5)
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'cnn':
        model = Sequential()
        model.add(Conv1D(filters=trial.suggest_int('filters', 16, 64, log=True),
                         kernel_size=trial.suggest_int('kernel_size', 3, 5),
                         activation='relu',
                         input_shape=(input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'lstm':
        model = Sequential()
        model.add(LSTM(units=trial.suggest_int('lstm_units', 16, 64, log=True),
                       input_shape=(input_shape, 1)))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'transformer':
        inputs = Input(shape=(input_shape, 1))
        attention = MultiHeadAttention(num_heads=trial.suggest_int('num_heads', 2, 8), key_dim=trial.suggest_int('key_dim', 16, 64))(inputs, inputs)
        attention = Add()([inputs, attention])
        attention = LayerNormalization(epsilon=1e-6)(attention)
        outputs = Flatten()(attention)
        outputs = Dense(1, activation='linear')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
    
    elif model_type == 'tabnet':
        tabnet_model = TabNetRegressor(
            n_d=trial.suggest_int('n_d', 8, 64),
            n_a=trial.suggest_int('n_a', 8, 64),
            n_steps=trial.suggest_int('n_steps', 3, 10),
            gamma=trial.suggest_float('gamma', 1.0, 2.0),
            lambda_sparse=trial.suggest_float('lambda_sparse', 1e-6, 1e-3)
        )
        tabnet_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
            verbose=0
        )
        return tabnet_model

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

# Define the objective function for Optuna
def objective(trial):
    # Use a small random subset of the data
    subset_idx = np.random.choice(len(X_train), size=int(0.1 * len(X_train)), replace=False)
    X_subset = X_train[subset_idx]
    y_subset = y_train[subset_idx]

    model = create_model(trial)

    if isinstance(model, TabNetRegressor):
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
    else:
        model.fit(X_subset, y_subset, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
        loss, mae = model.evaluate(X_val, y_val, verbose=0)

    return mae

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# If using CNN, LSTM, or Transformer, add a new dimension for channels
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

# Run the Bayesian optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the best model on the full dataset
best_model = create_model(study.best_trial)
if isinstance(best_model, TabNetRegressor):
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['mae'],
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        verbose=0
    )
    best_model.save_model('best_tabnet_model')
    print("Best TabNet model saved as best_tabnet_model.zip")
else:
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
    best_model.save(model_save_file)
    print(f"Best model saved as {model_save_file}")


