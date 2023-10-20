import pandas as pd
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data from a CSV file
df = pd.read_csv('/home/chetana/gridmet_test_run/five_years_data.csv')

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

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['swe_value'])
y = df['swe_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the AutoKeras regressor
reg = ak.StructuredDataRegressor(max_trials=10, overwrite=True)
reg.fit(X_train, y_train, epochs=10)

# Evaluate the AutoKeras regressor on the test set
predictions = reg.predict(X_test)

# Calculate and print evaluation metrics
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print('RMSE:', rmse)
print('R2 Score:', r2)

