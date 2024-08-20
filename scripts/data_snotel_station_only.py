import math
import json
import requests
import pandas as pd
import csv
import io
import os
import dask
import dask.dataframe as dd
from snowcast_utils import work_dir, southwest_lat, southwest_lon, northeast_lat, northeast_lon, train_start_date, train_end_date

working_dir = work_dir


def download_station_json():
    # https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?activeOnly=true&returnForecastPointMetadata=false&returnReservoirMetadata=false&returnStationElements=false
    output_json_file = f'{working_dir}/all_snotel_cdec_stations.json'
    if not os.path.exists(output_json_file):
        # Fetch data from the URL
        response = requests.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations?activeOnly=true&returnForecastPointMetadata=false&returnReservoirMetadata=false&returnStationElements=false")
        

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Decode the JSON content
            json_content = response.json()

            # Save the JSON content to a file
            with open(output_json_file, 'w') as json_file:
                json.dump(json_content, json_file, indent=2)

            print(f"Data downloaded and saved to {output_json_file}")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")
    else:
        print(f"The file {output_json_file} already exists.")
        
    
    # read the json file and convert it to csv
    csv_file_path = f'{working_dir}/all_snotel_cdec_stations.csv'
    if not os.path.exists(csv_file_path):
        # Read the JSON file
        with open(output_json_file, 'r') as json_file:
            json_content = json.load(json_file)

        # Check the content (print or analyze as needed)
        #print("JSON Content:")
        #print(json.dumps(json_content, indent=2))

        # Convert JSON data to a list of dictionaries (assuming JSON is a list of objects)
        data_list = json_content if isinstance(json_content, list) else [json_content]

        # Get the header from the keys of the first dictionary (assuming consistent structure)
        header = data_list[0].keys()
        # Write to CSV file
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=header)
            csv_writer.writeheader()
            csv_writer.writerows(data_list)

        print(f"Data converted and saved to {csv_file_path}")
    
    else:
        print(f"The csv all snotel/cdec stations exists.")
        
        
    active_csv_file_path = f'{working_dir}/all_snotel_cdec_stations_active_in_westus.csv'
    if not os.path.exists(active_csv_file_path):
        all_df = pd.read_csv(csv_file_path)
        print(all_df.head())
        all_df['endDate'] = pd.to_datetime(all_df['endDate'])
        print(all_df.shape)
        end_date = pd.to_datetime('2050-01-01')
        filtered_df = all_df[all_df['endDate'] > end_date]
        
        # Filter rows within the latitude and longitude ranges
        filtered_df = filtered_df[
            (filtered_df['latitude'] >= southwest_lat) & (filtered_df['latitude'] <= northeast_lat) &
            (filtered_df['longitude'] >= southwest_lon) & (filtered_df['longitude'] <= northeast_lon)
        ]

        # Print the original and filtered DataFrames
        print("Filtered DataFrame:")
        print(filtered_df.shape)
        filtered_df.to_csv(active_csv_file_path, index=False)
    else:
        print(f"The active csv already exists: {active_csv_file_path}")
	

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as json_file:
        data = json.load(json_file)
        return data


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lat = lat2 - lat1
    d_long = lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # Earth's radius in kilometers
    return distance


def find_nearest_location(locations, target_lat, target_lon):
    n_location = None
    min_distance = float('inf')
    for location in locations:
        lat = location['location']['lat']
        lon = location['location']['lng']
        distance = haversine(lat, lon, target_lat, target_lon)
        if distance < min_distance:
            min_distance = distance
            n_location = location
            return n_location


def csv_to_json(csv_text):
    lines = csv_text.splitlines()
    header = lines[0]
    field_names = header.split(',')
    json_list = []
    for line in lines[1:]:
        values = line.split(',')
        row_dict = {}
        for i, field_name in enumerate(field_names):
            row_dict[field_name] = values[i]
            json_list.append(row_dict)
            json_string = json.dumps(json_list)
            return json_string


def remove_commented_lines(text):
    lines = text.split(os.linesep)
    cleaned_lines = []
    for line in lines:
        if not line.startswith('#'):
            cleaned_lines.append(line)
    cleaned_text = os.linesep.join(cleaned_lines)
    return cleaned_text

  
def get_swe_observations_from_snotel_cdec():
    new_base_station_list_file = f"{work_dir}/all_snotel_cdec_stations_active_in_westus.csv"
    new_base_df = pd.read_csv(new_base_station_list_file)
    print(new_base_df.head())
  	
    old_messed_file = f"{work_dir}/"
    csv_file = f'{new_base_station_list_file}_swe_restored_dask_all_vars.csv'
    start_date = train_start_date
    end_date = train_end_date
	
    # Create an empty Pandas DataFrame with the desired columns
    result_df = pd.DataFrame(columns=[
      'station_name', 
      'date', 
      'lat', 
      'lon', 
      'swe_value', 
      'change_in_swe_inch', 
      'snow_depth', 
      'change_in_swe_inch', 
      'air_temperature_observed_f'
    ])

    # Function to process each station
    @dask.delayed
    def process_station(station):
        location_name = station['name']
        location_triplet = station['stationTriplet']
        location_elevation = station['elevation']
        location_station_lat = station['latitude']
        location_station_long = station['longitude']

        url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{location_triplet}%7Cid%3D%22%22%7Cname/{start_date},{end_date}%2C0/WTEQ%3A%3Avalue%2CWTEQ%3A%3Adelta%2CSNWD%3A%3Avalue%2CSNWD%3A%3Adelta%2CTOBS%3A%3Avalue"

        r = requests.get(url)
        text = remove_commented_lines(r.text)
        reader = csv.DictReader(io.StringIO(text))
        json_data = json.loads(json.dumps(list(reader)))

        entries = []
        
        for entry in json_data:
            try:
              # {'Date': '2021-06-18', 'Snow Water Equivalent (in) Start of Day Values': '', 'Change In Snow Water Equivalent (in)': '', 'Snow Depth (in) Start of Day Values': '', 'Change In Snow Depth (in)': '', 'Air Temperature Observed (degF) Start of Day Values': '70.5'}
              required_data = {
                'station_name': location_name,
                'date': entry['Date'],
                'lat': location_station_lat, 
                'lon': location_station_long,
                'swe_value': entry['Snow Water Equivalent (in) Start of Day Values'],
                'change_in_swe_inch': entry['Change In Snow Water Equivalent (in)'],
                'snow_depth': entry['Snow Depth (in) Start of Day Values'],
                'change_in_swe_inch': entry['Change In Snow Depth (in)'],
                'air_temperature_observed_f': entry['Air Temperature Observed (degF) Start of Day Values']
              }
              entries.append(required_data)
            except Exception as e:
              print("entry = ", entry)
              raise e
        return pd.DataFrame(entries)

    # List of delayed computations for each station
    delayed_results = [process_station(row) for _, row in new_base_df.iterrows()]

    # Compute the delayed results
    result_lists = dask.compute(*delayed_results)

    # Concatenate the lists into a Pandas DataFrame
    result_df = pd.concat(result_lists, ignore_index=True)

    # Print the final Pandas DataFrame
    print(result_df.head())

    # Save the DataFrame to a CSV file
    result_df.to_csv(csv_file, index=False)
#     result_df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    download_station_json()
    get_swe_observations_from_snotel_cdec()

