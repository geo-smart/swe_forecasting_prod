import os
import pandas as pd
import numpy as np
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import earthaccess
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from haversine import haversine

# Directory setup
modis_input_folder = os.getcwd() + "/modis_surface_reflectance/"
os.makedirs(modis_input_folder, exist_ok=True)

# Define the time range and bounding box
start_date = "2018-01-01"
end_date = "2018-01-07"
bounding_box = (-109.0, 36.0, -102.0, 41.0)  # Bounding box for Colorado Rockies

# Download MODIS surface reflectance bands
def download_modis_surface_reflectance(start_date, end_date, bounding_box, input_folder):
    print(f"Starting MODIS surface reflectance download for {start_date} to {end_date}...")
    results = earthaccess.search_data(short_name="MOD09GA", 
                                      cloud_hosted=True, 
                                      bounding_box=bounding_box, 
                                      temporal=(start_date, end_date))
    earthaccess.download(results, input_folder)
    print("Download completed.")

# Extract surface reflectance bands from HDF files
def extract_surface_reflectance_bands(hdf_file):
    print(f"Extracting surface reflectance bands from {hdf_file}...")
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    bands = {
        "Band_1": gdal.Open(hdf_ds.GetSubDatasets()[0][0]).ReadAsArray(),  # Red
        "Band_2": gdal.Open(hdf_ds.GetSubDatasets()[1][0]).ReadAsArray(),  # NIR
        "Band_3": gdal.Open(hdf_ds.GetSubDatasets()[2][0]).ReadAsArray(),  # Blue
        "Band_4": gdal.Open(hdf_ds.GetSubDatasets()[3][0]).ReadAsArray(),  # Green
        "Band_5": gdal.Open(hdf_ds.GetSubDatasets()[4][0]).ReadAsArray(),  # SWIR 1
        "Band_6": gdal.Open(hdf_ds.GetSubDatasets()[5][0]).ReadAsArray(),  # SWIR 2
        "Band_7": gdal.Open(hdf_ds.GetSubDatasets()[6][0]).ReadAsArray(),  # SWIR 3
    }
    print("Band extraction completed.")
    return bands

# Preprocess SNOTEL data
def preprocess_snotel_data(snotel_df, start_date, end_date):
    print(f"Preprocessing SNOTEL data for date range {start_date} to {end_date}...")
    filtered_snotel = snotel_df[(snotel_df['date'] >= start_date) & (snotel_df['date'] <= end_date)]
    filtered_snotel = filtered_snotel.dropna(subset=['swe_value'])
    print(f"Preprocessing completed. Filtered data contains {len(filtered_snotel)} records.")
    return filtered_snotel[['lat', 'lon', 'swe_value']]

# Calculate distance between two geographic points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Integrate MODIS bands and SNOTEL data
@delayed
def process_file(hdf_file, snotel_data):
    X = []
    y = []
    
    bands = extract_surface_reflectance_bands(hdf_file)
    
    # Extract pixel coordinates from bands (placeholder)
    pixel_coords = np.zeros((bands["Band_1"].shape[0], bands["Band_1"].shape[1], 2))  # Placeholder
    
    for lat, lon, swe in snotel_data[['lat', 'lon', 'swe_value']].values:
        # Vectorized approach to compute all distances and find the closest pixel
        latlon_array = np.dstack((pixel_coords[:, :, 0].flatten(), pixel_coords[:, :, 1].flatten()))[0]
        distances = np.apply_along_axis(lambda x: haversine(lon, lat, x[1], x[0]), 1, latlon_array)
        closest_pixel_idx = np.argmin(distances)
        min_distance = distances[closest_pixel_idx]
        
        if min_distance <= 10:  # Only use if within 10 km
            closest_pixel = np.unravel_index(closest_pixel_idx, (bands["Band_1"].shape[0], bands["Band_1"].shape[1]))
            band_values = [bands[band].flatten()[closest_pixel_idx] for band in bands]
            X.append(band_values)
            y.append(swe)
    
    return np.array(X), np.array(y)

def integrate_modis_snotel(modis_folder, snotel_data):
    print("Integrating MODIS bands with SNOTEL data...")
    
    hdf_files = [os.path.join(modis_folder, file) for file in os.listdir(modis_folder) if file.endswith(".hdf")]
    
    # Process files in parallel using Dask
    results = [process_file(hdf_file, snotel_data) for hdf_file in hdf_files]
    with ProgressBar():
        computed_results = compute(*results)
    
    # Combine the results from all files
    X = np.vstack([res[0] for res in computed_results])
    y = np.hstack([res[1] for res in computed_results])
    
    print(f"Integration completed. Number of samples: {len(X)}")
    return X, y

# Train and evaluate the ML model
def train_and_evaluate_model(X, y):
    print("Training and evaluating the ML model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model training completed. Mean Squared Error: {mse}")
    return model

# Main workflow
def main():
    print("Starting workflow...")
    download_modis_surface_reflectance(start_date, end_date, bounding_box, modis_input_folder)
    
    # Load SNOTEL data
    snotel_file = "/home/jovyan/shared-public/ml_swe_monitoring_prod/all_snotel_cdec_stations_active_in_westus.csv_swe_restored_dask_all_vars.csv"
    print(f"Loading SNOTEL data from {snotel_file}...")
    snotel_df = pd.read_csv(snotel_file)
    
    # Preprocess SNOTEL data
    snotel_data = preprocess_snotel_data(snotel_df, start_date, end_date)
    
    # Integrate MODIS bands with SNOTEL SWE data
    X, y = integrate_modis_snotel(modis_input_folder, snotel_data)
    
    # Train and evaluate the model
    model = train_and_evaluate_model(X, y)
    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()
