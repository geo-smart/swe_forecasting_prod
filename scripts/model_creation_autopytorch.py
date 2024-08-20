import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import autopytorch as apt
from snowcast_utils import work_dir, month_to_season


working_dir = work_dir


training_data_path = f"{working_dir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv"
# Load the data from a CSV file
df = pd.read_csv(training_data_path)

# Remove rows with missing values
df.dropna(inplace=True)

# Initialize a label encoder
label_encoder = LabelEncoder()

# Drop unnecessary columns from the DataFrame
df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
df.drop('Date', inplace=True, axis=1)
df.drop('mapping_cell_id', inplace=True, axis=1)
df.drop('cell_id', inplace=True, axis=1)
df.drop('station_id', inplace=True, axis=1)
df.drop('mapping_station_id', inplace=True, axis=1)
df.drop('station_triplet', inplace=True, axis=1)
df.drop('station_name', inplace=True, axis=1)

# Rename columns for better readability
df.rename(columns={
    'Change In Snow Water Equivalent (in)': 'swe_change',
    'Snow Depth (in) Start of Day Values': 'swe_value',
    'Change In Snow Depth (in)': 'snow_depth_change',
    'Air Temperature Observed (degF) Start of Day Values': 'snotel_air_temp',
    'Elevation [m]': 'elevation',
    'Aspect [deg]': 'aspect',
    'Curvature [ratio]': 'curvature',
    'Slope [deg]': 'slope',
    'Eastness [unitCirc.]': 'eastness',
    'Northness [unitCirc.]': 'northness'
}, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop('swe_value', axis=1)
y = df['swe_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Auto-PyTorch configuration
config = apt.AutoNetRegressionConfig()

# Initialize and train the Auto-PyTorch regressor
reg = apt.AutoNetRegressor(config=config)
reg.fit(X_train, y_train)

# Evaluate the model
predictions = reg.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

# Print the evaluation metrics
print("RMSE:", rmse)
print("R2 Score:", r2)

