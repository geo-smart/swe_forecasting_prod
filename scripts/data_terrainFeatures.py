# Load dependencies
import geopandas as gpd
import json
import geojson
from pystac_client import Client
import planetary_computer
import xarray
import rioxarray
import xrspatial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Proj, transform
import os
import sys, traceback
import requests
from snowcast_utils import work_dir


home_dir = os.path.expanduser('~')
snowcast_github_dir = f"{home_dir}/Documents/GitHub/SnowCast/"


#exit() # this process no longer need to execute, we need to make Geoweaver to specify which process doesn't need to run


# user-defined paths for data-access
data_dir = f'{snowcast_github_dir}data/'
gridcells_file = data_dir+'snowcast_provided/grid_cells_eval.geojson'
#stations_file = data_dir+'snowcast_provided/ground_measures_metadata.csv'
stations_file = f"{work_dir}/all_snotel_cdec_stations_active_in_westus.csv"
#stations_file = data_dir+'snowcast_provided/ground_measures_metadata.csv'
all_training_points_with_station_and_non_station_file = f"{work_dir}/all_training_points_in_westus.csv"
all_training_points_with_snotel_ghcnd_file = f"{work_dir}/all_training_points_snotel_ghcnd_in_westus.csv"
gridcells_outfile = data_dir+'terrain/gridcells_terrainData_eval.csv'
#stations_outfile = f"{work_dir}/training_all_active_snotel_station_list_elevation.csv_terrain_4km_grid_shift.csv"
stations_outfile = f"{work_dir}/all_training_points_with_ghcnd_in_westus.csv_terrain_4km_grid_shift.csv"


def get_planetary_client():
  #requests.get('https://planetarycomputer.microsoft.com/api/stac/v1')

  # setup client for handshaking and data-access
  print("setup planetary computer client")
  client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",ignore_conformance=True)
  
  return client

def prepareGridCellTerrain():
  client = get_planetary_client()
  # Load metadata
  gridcellsGPD = gpd.read_file(gridcells_file)
  gridcells = geojson.load(open(gridcells_file))
  stations = pd.read_csv(stations_file)

  # instantiate output panda dataframes
  df_gridcells = df = pd.DataFrame(columns=(
    "Longitude [deg]","Latitude [deg]",
    "Elevation [m]","Aspect [deg]",
    "Curvature [ratio]","Slope [deg]",
    "Eastness [unitCirc.]","Northness [unitCirc.]"))
  # instantiate output panda dataframes
  # Calculate gridcell characteristics using Copernicus DEM data
  print("Prepare GridCell Terrain data")
  for idx,cell in enumerate(gridcells['features']):
      print("Processing grid ", idx)
      search = client.search(
          collections=["cop-dem-glo-30"],
          intersects={"type":"Polygon", "coordinates":cell['geometry']['coordinates']},
      )
      items = list(search.get_items())
      print("==> Searched items: ", len(items))

      cropped_data = None
      try:
          signed_asset = planetary_computer.sign(items[0].assets["data"])
          data = (
              #xarray.open_rasterio(signed_asset.href)
              xarray.open_rasterio(signed_asset.href)
              .squeeze()
              .drop("band")
              .coarsen({"y": 1, "x": 1})
              .mean()
          )
          cropped_data = data.rio.clip(gridcellsGPD['geometry'][idx:idx+1])
      except:
          signed_asset = planetary_computer.sign(items[1].assets["data"])
          data = (
              xarray.open_rasterio(signed_asset.href)
              .squeeze()
              .drop("band")
              .coarsen({"y": 1, "x": 1})
              .mean()
          )
          cropped_data = data.rio.clip(gridcellsGPD['geometry'][idx:idx+1])

      # calculate lat/long of center of gridcell
      longitude = np.unique(np.ravel(cell['geometry']['coordinates'])[0::2]).mean()
      latitude = np.unique(np.ravel(cell['geometry']['coordinates'])[1::2]).mean()

      print("reproject data to EPSG:32612")
      # reproject the cropped dem data
      cropped_data = cropped_data.rio.reproject("EPSG:32612")

      # Mean elevation of gridcell
      mean_elev = cropped_data.mean().values
      print("Elevation: ", mean_elev)

      # Calculate directional components
      aspect = xrspatial.aspect(cropped_data)
      aspect_xcomp = np.nansum(np.cos(aspect.values*(np.pi/180)))
      aspect_ycomp = np.nansum(np.sin(aspect.values*(np.pi/180)))
      mean_aspect = np.arctan2(aspect_ycomp,aspect_xcomp)*(180/np.pi)
      if mean_aspect < 0:
          mean_aspect = 360 + mean_aspect
      print("Aspect: ", mean_aspect)
      mean_eastness = np.cos(mean_aspect*(np.pi/180))
      mean_northness = np.sin(mean_aspect*(np.pi/180))
      print("Eastness: ", mean_eastness)
      print("Northness: ", mean_northness)

      # Positive curvature = upward convex
      curvature = xrspatial.curvature(cropped_data)
      mean_curvature = curvature.mean().values
      print("Curvature: ", mean_curvature)

      # Calculate mean slope
      slope = xrspatial.slope(cropped_data)
      mean_slope = slope.mean().values
      print("Slope: ", mean_slope)

      # Fill pandas dataframe
      df_gridcells.loc[idx] = [longitude,latitude,
                               mean_elev,mean_aspect,
                               mean_curvature,mean_slope,
                               mean_eastness,mean_northness]

      # Comment out for debugging/filling purposes
      # if idx % 250 == 0:
      #     df_gridcells.set_index(gridcellsGPD['cell_id'][0:idx+1],inplace=True)
      #     df_gridcells.to_csv(gridcells_outfile)

  # Save output data into csv format
  df_gridcells.set_index(gridcellsGPD['cell_id'][0:idx+1],inplace=True)
  df_gridcells.to_csv(gridcells_outfile)

def prepareStationTerrain():
  client = get_planetary_client()
  
  df_station = pd.DataFrame(columns=("Longitude [deg]","Latitude [deg]",
                                     "Elevation [m]","Elevation_30 [m]","Elevation_1000 [m]",
                                     "Aspect_30 [deg]","Aspect_1000 [deg]",
                                     "Curvature_30 [ratio]","Curvature_1000 [ratio]",
                                     "Slope_30 [deg]","Slope_1000 [deg]",
                                     "Eastness_30 [unitCirc.]","Northness_30 [unitCirc.]",
                                     "Eastness_1000 [unitCirc.]","Northness_1000 [unitCirc.]"))
  
  stations_df = pd.read_csv(stations_file)
  print(stations_df.head())
  # Calculate terrain characteristics of stations, and surrounding regions using COP 30
  for idx,station in stations_df.iterrows():
      search = client.search(
          collections=["cop-dem-glo-30"],
          intersects={
            "type": "Point", 
            "coordinates": [
              stations_df['lon'],
              stations_df['lat']
            ]
          },
      )
      items = list(search.get_items())
      print(f"Returned {len(items)} items")

      try:
          signed_asset = planetary_computer.sign(items[0].assets["data"])
          data = (
              xarray.open_rasterio(signed_asset.href)
              .squeeze()
              .drop("band")
              .coarsen({"y": 1, "x": 1})
              .mean()
          )
          xdiff = np.abs(data.x-stations_df['lon'])
          ydiff = np.abs(data.y-stations_df['lat'])
          xdiff = np.where(xdiff == xdiff.min())[0][0]
          ydiff = np.where(ydiff == ydiff.min())[0][0]
          data = data[ydiff-33:ydiff+33,xdiff-33:xdiff+33].rio.reproject("EPSG:32612")
      except:
          traceback.print_exc(file=sys.stdout)
          signed_asset = planetary_computer.sign(items[1].assets["data"])
          data = (
              xarray.open_rasterio(signed_asset.href)
              .squeeze()
              .drop("band")
              .coarsen({"y": 1, "x": 1})
              .mean()
          )
          xdiff = np.abs(data.x-stations_df['lon'])
          ydiff = np.abs(data.y-stations_df['lat'])
          xdiff = np.where(xdiff == xdiff.min())[0][0]
          ydiff = np.where(ydiff == ydiff.min())[0][0]
          data = data[ydiff-33:ydiff+33,xdiff-33:xdiff+33].rio.reproject("EPSG:32612")

      # Reproject the station data to better include only 1000m surrounding area
      inProj = Proj(init='epsg:4326')
      outProj = Proj(init='epsg:32612')
      new_x,new_y = transform(inProj,outProj,
                              stations_df['lon'],
                              stations_df['lat'])

      # Calculate elevation of station and surroundings
      mean_elevation = data.mean().values
      elevation = data.sel(x=new_x,y=new_y,method='nearest')
      print(elevation.values)

      # Calcuate directional components
      aspect = xrspatial.aspect(data)
      aspect_xcomp = np.nansum(np.cos(aspect.values*(np.pi/180)))
      aspect_ycomp = np.nansum(np.sin(aspect.values*(np.pi/180)))
      mean_aspect = np.arctan2(aspect_ycomp,aspect_xcomp)*(180/np.pi)
      if mean_aspect < 0:
          mean_aspect = 360 + mean_aspect
      #print(mean_aspect)
      aspect = aspect.sel(x=new_x,y=new_y,method='nearest')
      #print(aspect.values)
      eastness = np.cos(aspect*(np.pi/180))
      northness = np.sin(aspect*(np.pi/180))
      mean_eastness = np.cos(mean_aspect*(np.pi/180))
      mean_northness = np.sin(mean_aspect*(np.pi/180))

      # Positive curvature = upward convex
      curvature = xrspatial.curvature(data)
      mean_curvature = curvature.mean().values
      curvature = curvature.sel(x=new_x,y=new_y,method='nearest')
      print(curvature.values)

      # Calculate slope
      slope = xrspatial.slope(data)
      mean_slope = slope.mean().values
      slope = slope.sel(x=new_x,y=new_y,method='nearest')
      print(slope.values)

      # Fill pandas dataframe
      df_station.loc[idx] = [stations_df['lon'],
                             stations_df['lat'],
                             station['elevation_m'],
                             elevation.values,mean_elevation,
                             aspect.values,mean_aspect,
                             curvature.values,mean_curvature,
                             slope.values,mean_slope,
                             eastness.values,northness.values,
                             mean_eastness,mean_northness]

  # Save output data into CSV format
  df_station.set_index(stations_df['station_name'][0:idx+1],inplace=True)
  df_station.to_csv(stations_outfile)


def add_more_points_to_the_gridcells():
  # check how many points are in the current grid_cell json
  station_cell_mapping = f"{work_dir}/station_cell_mapping.csv"
  current_grid_df = pd.read_csv(station_cell_mapping)
  
  print(current_grid_df.columns)
  print(current_grid_df.shape)
  
  western_us_coords = f'{work_dir}/dem_file.tif.csv'
  dem_df = pd.read_csv(western_us_coords)
  print(dem_df.head())
  print(dem_df.shape)
  filtered_df = dem_df[dem_df['Elevation'] > 20]  # choose samples from points higher than 20 meters

  # Randomly choose 700 rows from the filtered DataFrame
  random_rows = filtered_df.sample(n=700)
  random_rows = random_rows[["Latitude", "Longitude"]]
  random_rows.rename(columns={
    'Latitude': 'lat', 
    'Longitude': 'lon'
  }, inplace=True)
  previous_cells = current_grid_df[["lat", "lon"]]
  result_df = previous_cells.append(random_rows, ignore_index=True)
  print(result_df.shape)
  result_df.to_csv(f"{work_dir}/new_training_points_with_random_dem_locations.csv")
  print(f"New training points are saved to {work_dir}/new_training_points_with_random_dem_locations.csv")
  
  
  
  # find the random points that are on land from the dem.json
  
  # merge the grid_cell.json with the new dem points into a new grid_cell.json
  
def find_closest_index(target_latitude, target_longitude, lat_grid, lon_grid):
    """
    Find the closest grid point indices for a target latitude and longitude.

    Parameters:
        target_latitude (float): Target latitude.
        target_longitude (float): Target longitude.
        lat_grid (numpy.ndarray): Array of latitude values.
        lon_grid (numpy.ndarray): Array of longitude values.

    Returns:
        int: Latitude index.
        int: Longitude index.
        float: Closest latitude value.
        float: Closest longitude value.
    """
    lat_diff = np.float64(np.abs(lat_grid - target_latitude))
    lon_diff = np.float64(np.abs(lon_grid - target_longitude))
    #print("lat_diff = ", lat_diff)
    #print("lon_diff = ", lon_diff)

    #lat_idx = np.argmin(lat_diff)
    #lon_idx = np.argmin(lon_diff)
    # Find the indices corresponding to the minimum differences
    #lat_idx, lon_idx = np.unravel_index(np.argmin(lat_diff + lon_diff), lat_grid.shape)
    row_idx = np.argmin(lat_diff + lon_diff)

    return row_idx
  
  
def read_terrain_from_dem_csv():
  western_us_coords = f'{work_dir}/dem_all.csv'
  western_df = pd.read_csv(western_us_coords)
  print("western_df.head() = ", western_df.head())
  
  stations_file_df = pd.read_csv(all_training_points_with_snotel_ghcnd_file)
  print("stations_file_df.head() = ", stations_file_df.head())
  
  def find_closest_dem_row(row, western_df):
    #print(row)
    row_idx = find_closest_index(
      row["latitude"],
      row["longitude"],
      western_df["Latitude"], 
      western_df["Longitude"]
    )
    #print("row_idx = ", row_idx)
    dem_row = western_df.iloc[row_idx]
    #print("dem_row = ", dem_row)
    new_row = pd.concat([row, dem_row], axis=0)
    #print("result_series = ", new_row)
    #exit(1)
    return new_row
  
  stations_file_df = stations_file_df.apply(find_closest_dem_row, args=(western_df,), axis=1)
  stations_file_df.to_csv(stations_outfile, index=False)
  print(f"New elevation csv is aved to {stations_outfile}")
  

if __name__ == "__main__":
  try:
    #prepareGridCellTerrain()
    #prepareStationTerrain()
    
    #add_more_points_to_the_gridcells()
    read_terrain_from_dem_csv()
  except:
    traceback.print_exc(file=sys.stdout)

