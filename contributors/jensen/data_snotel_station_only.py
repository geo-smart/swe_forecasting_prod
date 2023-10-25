import math
import json
import requests
import pandas as pd
import csv
import io
import os
from snowcast_utils import work_dir

working_dir = work_dir

def read_json_file(file_path):
    '''
    Read and parse a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON data.
    '''
    with open(file_path, 'r', encoding='utf-8-sig') as json_file:
        data = json.load(json_file)
        return data

def haversine(lat1, lon1, lat2, lon2):
    '''
    Calculate the Haversine distance between two sets of latitude and longitude coordinates.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The Haversine distance between the two points in kilometers.
    '''
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lat = lat2 - lat1
    d_long = lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # Earth's radius in kilometers
    return distance

def find_nearest_location(locations, target_lat, target_lon):
    '''
    Find the nearest location in a list of locations to a target latitude and longitude.

    Args:
        locations (list): List of locations, each represented as a dictionary.
        target_lat (float): Target latitude.
        target_lon (float): Target longitude.

    Returns:
        dict: The nearest location from the list.
    '''
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
    '''
    Convert CSV text to JSON format.

    Args:
        csv_text (str): The CSV text to convert.

    Returns:
        str: The JSON representation of the CSV data.
    '''
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
    '''
    Remove lines starting with '#' from the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The input text with lines starting with '#' removed.
    '''
    lines = text.split(os.linesep)
    cleaned_lines = []
    for line in lines:
        if not line.startswith('#'):
            cleaned_lines.append(line)
    cleaned_text = os.linesep.join(cleaned_lines)
    return cleaned_text

def start_to_collect_snotel():
    '''
    Start the process of collecting SNOTEL data and saving it to a CSV file.
    '''
    csv_file = f'{working_dir}/training_data_ready_snotel_3_yrs.csv'
    start_date = "2019-01-01"
    end_date = "2022-12-12"

    if os.path.exists(csv_file):
        print(f"The file '{csv_file}' exists.")
        return

    station_mapping = pd.read_csv(f'{working_dir}/station_cell_mapping.csv')

    result_df = pd.DataFrame(columns=['date', 'lat', 'lon', 'swe_value'])
    for index, row in station_mapping.iterrows():
        print(index, ' / ', len(station_mapping), ' iterations completed.')
        station_locations = read_json_file(f'{working_dir}/snotelStations.json')
        nearest_location = find_nearest_location(station_locations, row['lat'], row['lon'])

        location_name = nearest_location['name']
        location_triplet = nearest_location['triplet']
        location_elevation = nearest_location['elevation']
        location_station_lat = nearest_location['location']['lat']
        location_station_long = nearest_location['location']['lng']

        url = f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/{location_triplet}%7Cid%3D%22%22%7Cname/{start_date},{end_date}%2C0/WTEQ%3A%3Avalue%2CWTEQ%3A%3Adelta%2CSNWD%3A%3Avalue%2CSNWD%3A%3Adelta%2CTOBS%3A%3Avalue"

        r = requests.get(url)
        text = remove_commented_lines(r.text)
        reader = csv.DictReader(io.StringIO(text))
        json_data = json.loads(json.dumps(list(reader)))
        for entry in json_data:
            required_data = {'date': entry['Date'], 'lat': row['lat'], 'lon': row['lon'],
                             'swe_value': entry['Snow Water Equivalent (in) Start of Day Values']}
            result_df.loc[len(result_df.index)] = required_data

    # Save the DataFrame to a CSV file
    result_df.to_csv(csv_file, index=False)

start_to_collect_snotel()

