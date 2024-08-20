"""
This process need to retrieve the snow depth ground station data from GHCNd.

"""

import pandas as pd
import requests
import re
from io import StringIO
from snowcast_utils import work_dir, southwest_lat, southwest_lon, northeast_lat, northeast_lon, train_start_date, train_end_date
import dask
import dask.dataframe as dd

working_dir = work_dir
all_ghcd_station_file = f"{working_dir}/all_ghcn_station_list.csv"
only_active_ghcd_station_in_west_conus_file = f"{working_dir}/active_station_only_list.csv"
snowdepth_csv_file = f'{only_active_ghcd_station_in_west_conus_file}_all_vars.csv'
mask_non_snow_days_ghcd_csv_file =  f'{only_active_ghcd_station_in_west_conus_file}_all_vars_masked_non_snow.csv'

def download_convert_and_read():
  
    url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
    # Download the text file from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print("Error: Failed to download the file.")
        return None
    
    # Parse the text content into columns using regex
    pattern = r"(\S+)\s+"
    parsed_data = re.findall(pattern, response.text)
    print("len(parsed_data) = ", len(parsed_data))
    
    # Create a new list to store the rows
    rows = []
    for i in range(0, len(parsed_data), 6):
        rows.append(parsed_data[i:i+6])
    
    print("rows[0:20] = ", rows[0:20])
    # Convert the rows into a CSV-like format
    csv_data = "\n".join([",".join(row) for row in rows])
    
    # Save the CSV-like string to a file
    with open(all_ghcd_station_file, "w") as file:
        file.write(csv_data)
    
    # Read the CSV-like data into a pandas DataFrame
    column_names = ['Station', 'Latitude', 'Longitude', 'Variable', 'Year_Start', 'Year_End']
    df = pd.read_csv(all_ghcd_station_file, header=None, names=column_names)
    print(df.head())
    
    # Remove rows where the last column is not equal to "2024"
    df = df[(df['Year_End'] == 2024) & (df['Variable']=='SNWD')]
    print("Removed non-active stations: ", df.head())
    
    # Filter rows within the latitude and longitude ranges
    df = df[
      (df['Latitude'] >= southwest_lat) & (df['Latitude'] <= northeast_lat) &
      (df['Longitude'] >= southwest_lon) & (df['Longitude'] <= northeast_lon)
    ]
    
    df.to_csv(only_active_ghcd_station_in_west_conus_file, index=False)
    print(f"saved to {only_active_ghcd_station_in_west_conus_file}")
    
    
    return df

  
def get_snow_depth_observations_from_ghcn():
    
    new_base_df = pd.read_csv(only_active_ghcd_station_in_west_conus_file)
    print(new_base_df.shape)
    
    start_date = train_start_date
    end_date = train_end_date
	
    # Create an empty Pandas DataFrame with the desired columns
    result_df = pd.DataFrame(columns=[
      'station_name', 
      'date', 
      'lat', 
      'lon', 
      'snow_depth',
    ])
    
    train_start_date_obj = pd.to_datetime(train_start_date)
    train_end_date_obj = pd.to_datetime(train_end_date)

    # Function to process each station
    @dask.delayed
    def process_station(station):
        station_name = station['Station']
        print(f"retrieving for {station_name}")
        station_lat = station['Latitude']
        station_long = station['Longitude']
        try:
          url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station_name}.csv"
          response = requests.get(url)
          df = pd.read_csv(StringIO(response.text))
          #"STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","NAME","PRCP","PRCP_ATTRIBUTES","SNOW","SNOW_ATTRIBUTES","SNWD","SNWD_ATTRIBUTES","TMAX","TMAX_ATTRIBUTES","TMIN","TMIN_ATTRIBUTES","PGTM","PGTM_ATTRIBUTES","WDFG","WDFG_ATTRIBUTES","WSFG","WSFG_ATTRIBUTES","WT03","WT03_ATTRIBUTES","WT08","WT08_ATTRIBUTES","WT16","WT16_ATTRIBUTES"
          columns_to_keep = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'SNWD']
          df = df[columns_to_keep]
          # Convert the date column to datetime objects
          df['DATE'] = pd.to_datetime(df['DATE'])
          # Filter rows based on the training period
          df = df[(df['DATE'] >= train_start_date_obj) & (df['DATE'] <= train_end_date_obj)]
          # print(df.head())
          return df
        except Exception as e:
          print("An error occurred:", e)

    # List of delayed computations for each station
    delayed_results = [process_station(row) for _, row in new_base_df.iterrows()]

    # Compute the delayed results
    result_lists = dask.compute(*delayed_results)

    # Concatenate the lists into a Pandas DataFrame
    result_df = pd.concat(result_lists, ignore_index=True)

    # Print the final Pandas DataFrame
    print(result_df.head())

    # Save the DataFrame to a CSV file
    result_df.to_csv(snowdepth_csv_file, index=False)
    print(f"All the data are saved to {snowdepth_csv_file}")
#     result_df.to_csv(csv_file, index=False)

def mask_out_all_non_zero_snowdepth_days():
    print(f"reading {snowdepth_csv_file}")
    df = pd.read_csv(snowdepth_csv_file)
    # Create the new column 'swe_value' and assign values based on conditions
    df['swe_value'] = 0  # Initialize all values to 0

    # Assign NaN to 'swe_value' where 'snow_depth' is non-zero
    df.loc[df['SNWD'] != 0, 'swe_value'] = -999

    # Display the first few rows of the DataFrame
    print(df.head())
    df.to_csv(mask_non_snow_days_ghcd_csv_file, index=False)
    print(f"The masked non snow var file is saved to {mask_non_snow_days_ghcd_csv_file}")

if __name__ == "__main__":
    #download_convert_and_read()
    #get_snow_depth_observations_from_ghcn()
    mask_out_all_non_zero_snowdepth_days()


