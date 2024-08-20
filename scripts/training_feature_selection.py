import dask.dataframe as dd

# Replace 'data.csv' with the path to your 50GB CSV file
input_csv = '/home/chetana/gridmet_test_run/model_training_data.csv'

# List of columns you want to extract
selected_columns = ['date', 'lat', 'lon', 'etr', 'pr', 'rmax',
                    'rmin', 'tmmn', 'tmmx', 'vpd', 'vs', 
                    'elevation',
                    'slope', 'curvature', 'aspect', 'eastness',
                    'northness', 'Snow Water Equivalent (in) Start of Day Values']

# Read the CSV file into a Dask DataFrame
df = dd.read_csv(input_csv, usecols=selected_columns)

# Rename the column as you intended
df = df.rename(columns={"Snow Water Equivalent (in) Start of Day Values": "swe_value"})

# Replace 'output.csv' with the desired output file name
output_csv = '/home/chetana/gridmet_test_run/model_training_cleaned.csv'

# Write the selected columns to a new CSV file
df.to_csv(output_csv, index=False, single_file=True)  # single_file=True 



