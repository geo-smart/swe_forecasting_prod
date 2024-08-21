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
import xarray as xr

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

def filter_snotel_data_within_bounding_box(snotel_data, bounding_box):
    lon_min, lat_min, lon_max, lat_max = bounding_box
    filtered_data = snotel_data[
        (snotel_data['lat'] >= lat_min) &
        (snotel_data['lat'] <= lat_max) &
        (snotel_data['lon'] >= lon_min) &
        (snotel_data['lon'] <= lon_max)
    ]
    print(f"Filtered SNOTEL data contains {len(filtered_data)} records.")
    return filtered_data

def extract_band_data(hdf_file, bounding_box, band_indices, band_names, output_csv):
    try:
        # Check if file exists
        if not os.path.isfile(hdf_file):
            print(f"File not found: {hdf_file}")
            return None

        print(f"Opening HDF file: {hdf_file}")

        # Try opening the file with xarray using 'h5netcdf' engine for HDF5
        try:
            ds = xr.open_dataset(hdf_file, engine='h5netcdf')
            print(f"HDF file opened successfully: {hdf_file}")
        except Exception as e:
            print(f"Failed to open HDF file with xarray: {e}")
            return None

        # Verify bands are present
        available_bands = [f'Band_{i+1}' for i in band_indices]
        missing_bands = [band for band in available_bands if band not in ds]
        if missing_bands:
            print(f"Missing bands in {hdf_file}: {missing_bands}")
            return None

        # Extract the bands
        bands = [ds[f'Band_{i+1}'] for i in band_indices]
        
        # Check if 'Longitude' and 'Latitude' exist
        if 'Longitude' not in ds or 'Latitude' not in ds:
            print(f"Longitude or Latitude not found in {hdf_file}")
            return None

        # Get latitude and longitude arrays
        lon = ds['Longitude'].values
        lat = ds['Latitude'].values
        print("Longitude and Latitude arrays retrieved")

        # Define a function to extract and mask the band data
        def extract_band(band, bounding_box):
            print(f"Extracting band data with bounding box: {bounding_box}")
            lon_min, lat_min, lon_max, lat_max = bounding_box
            mask = (
                (lon >= lon_min) & (lon <= lon_max) &
                (lat >= lat_min) & (lat <= lat_max)
            )
            band_data = band.where(mask, drop=True).values
            print(f"Band data shape after masking: {band_data.shape}")
            return band_data
        
        # Extract data for each band
        band_data_list = []
        for band, band_name in zip(bands, band_names):
            print(f"Processing band: {band_name}")
            band_data = extract_band(band, bounding_box)
            band_data_flat = band_data.flatten()
            print(f"Flattened band data shape: {band_data_flat.shape}")
            band_data_list.append(band_data_flat)
        
        # Create DataFrame
        print("Creating DataFrame with latitude, longitude, and band data")
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        df = pd.DataFrame({
            'Longitude': lon_flat,
            'Latitude': lat_flat
        })
        
        for band_name, band_data in zip(band_names, band_data_list):
            df[band_name] = band_data
        
        # Save DataFrame to CSV
        df.to_csv(output_csv, index=False)
        print(f"Data saved to CSV: {output_csv}")
        
        return df
    except Exception as e:
        print(f"Error extracting band data for {hdf_file}: {e}")
        return None


def process_hdf_files(hdf_folder, bounding_box, output_folder):
    # Example usage
    #hdf_folder = 'modis_surface_reflectance'
    #bounding_box = (-120, 35, -115, 40)  # Example bounding box: (lon_min, lat_min, lon_max, lat_max)
    band_indices = [0, 1, 2, 3, 4, 5, 6]  # Indices of the bands to process
    band_names = ['Red', 'NIR', 'Blue', 'Green', 'SWIR_1', 'SWIR_2', 'SWIR_3']  # Band names
    #output_folder = 'csv_output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for hdf_file in os.listdir(hdf_folder):
        if hdf_file.endswith('.hdf'):
            hdf_path = os.path.join(hdf_folder, hdf_file)
            output_csv = os.path.join(output_folder, f"{os.path.splitext(hdf_file)[0]}.csv")
            extract_band_data(hdf_path, bounding_box, band_indices, band_names, output_csv)


def extract_surface_reflectance_bands(hdf_file, bounding_box):
    if not tile_intersects_bounding_box(hdf_file, bounding_box):
        print(f"Skipping {hdf_file} as it does not intersect with the bounding box.")
        return None
    
    print(f"Extracting surface reflectance bands from {hdf_file}...")
    try:
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
    except Exception as e:
        print(f"Error extracting bands from {hdf_file}: {e}")
        return None

def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@delayed
def process_file(hdf_file, filtered_snotel_data, bounding_box):
    X = []
    y = []
    
    bands = extract_surface_reflectance_bands(hdf_file, bounding_box)
    
    if bands is None:
        print(f"Skipping file {hdf_file} because no bands were extracted.")
        return np.array(X), np.array(y)

    # Print shapes and statistics of the bands
    print(f"File: {hdf_file}")
    for band_name, band_array in bands.items():
        print(f"Band: {band_name}")
        print(f"Shape: {band_array.shape}")
        print(f"Min value: {np.min(band_array)}")
        print(f"Max value: {np.max(band_array)}")
        print(f"Mean value: {np.mean(band_array)}")
        print(f"Standard deviation: {np.std(band_array)}")
    
    pixel_coords = np.zeros((bands["Band_1"].shape[0], bands["Band_1"].shape[1], 2))  # Placeholder
    
    for lat, lon, swe in filtered_snotel_data[['lat', 'lon', 'swe_value']].values:
        latlon_array = np.dstack((pixel_coords[:, :, 0].flatten(), pixel_coords[:, :, 1].flatten()))[0]
        distances = np.apply_along_axis(lambda x: haversine(lon, lat, x[1], x[0]), 1, latlon_array)
        closest_pixel_idx = np.argmin(distances)
        min_distance = distances[closest_pixel_idx]
        
        if min_distance <= 10:  # Only use if within 10 km
            closest_pixel = np.unravel_index(closest_pixel_idx, (bands["Band_1"].shape[0], bands["Band_1"].shape[1]))
            band_values = [bands[band].flatten()[closest_pixel_idx] for band in bands]
            X.append(band_values)
            y.append(swe)

    # Print statistics of X and y arrays
    if X.size > 0:
        print(f"X array shape: {X.shape}")
        print(f"X array min value: {np.min(X)}")
        print(f"X array max value: {np.max(X)}")
        print(f"X array mean value: {np.mean(X)}")
        print(f"X array standard deviation: {np.std(X)}")
    else:
        print("X array is empty.")
    
    if y.size > 0:
        print(f"y array shape: {y.shape}")
        print(f"y array min value: {np.min(y)}")
        print(f"y array max value: {np.max(y)}")
        print(f"y array mean value: {np.mean(y)}")
        print(f"y array standard deviation: {np.std(y)}")
    else:
        print("y array is empty.")
    
    return np.array(X), np.array(y)

def integrate_modis_snotel(modis_folder, snotel_data, bounding_box):
    print("Integrating MODIS bands with SNOTEL data...")
    
    filtered_snotel_data = filter_snotel_data_within_bounding_box(snotel_data, bounding_box)
    
    hdf_files = [os.path.join(modis_folder, file) for file in os.listdir(modis_folder) if file.endswith(".hdf")]
    
    if not hdf_files:
        print("No MODIS HDF files found in the directory.")
        raise ValueError("No MODIS HDF files found in the directory.")
    
    print(f"Found MODIS files: {hdf_files}")
    
    results = [process_file(hdf_file, filtered_snotel_data, bounding_box) for hdf_file in hdf_files]
    with ProgressBar():
        computed_results = compute(*results)

    X_results = [res[0] for res in computed_results if res[0].size > 0]
    y_results = [res[1] for res in computed_results if res[1].size > 0]
    
    # Print shapes of results for debugging
    print(f"Shapes of X_results: {[x.shape for x in X_results]}")
    print(f"Shapes of y_results: {[y.shape for y in y_results]}")

    if not X_results or not y_results:
        raise ValueError("No valid data to concatenate. Check the MODIS files and SNOTEL data.")
    
    X = np.vstack(X_results)
    y = np.hstack(y_results)
    
    print(f"Integration completed. Number of samples: {len(X)}")
    return X, y

def train_and_evaluate_model(X, y):
    print("Training and evaluating the ML model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model training completed. Mean Squared Error: {mse}")
    return model

def main():
    print("Starting workflow...")
    # download_modis_surface_reflectance(start_date, end_date, bounding_box, modis_input_folder)

    # clip the data out from the HDF file and convert them into a CSV file
    folder_path = "modis_surface_reflectance"
    output_folder = "processed_bands_csv"
    
    process_hdf_files(folder_path, bounding_box, output_folder)
    
    # snotel_file = "/home/jovyan/shared-public/ml_swe_monitoring_prod/all_snotel_cdec_stations_active_in_westus.csv_swe_restored_dask_all_vars.csv"
    # print(f"Loading SNOTEL data from {snotel_file}...")
    # snotel_df = pd.read_csv(snotel_file)
    
    # filtered_snotel_data = filter_snotel_data_within_bounding_box(snotel_df, bounding_box)
    
    # X, y = integrate_modis_snotel(modis_input_folder, filtered_snotel_data, bounding_box)
    # model = train_and_evaluate_model(X, y)
    
    print("Workflow completed.")

if __name__ == "__main__":
    main()
