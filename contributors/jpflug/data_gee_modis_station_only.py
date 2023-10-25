

# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal

import json
import pandas as pd
import ee
import seaborn as sns
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import geojson
import numpy as np
import os.path
import eeauth as e

#exit() # done, uncomment if you want to download new files.

try:
    ee.Initialize(e.creds())
except Exception as e:
    # the following is for the server
    #service_account = 'eartheginegcloud@earthengine58.iam.gserviceaccount.com'
#creds = ee.ServiceAccountCredentials(
    #service_account, '/home/chetana/bhargavi-creds.json')
    #ee.Initialize(creds)
    ee.Authenticate() # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"

org_name = 'modis'
product_name = f'MODIS/006/MOD10A1'
var_name = 'NDSI'
column_name = 'mod10a1_ndsi'

#org_name = 'sentinel1'
#product_name = 'COPERNICUS/S1_GRD'
#var_name = 'VV'
#column_name = 's1_grd_vv'

station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)

all_cell_df = pd.DataFrame(columns = ['date', column_name, 'cell_id', 'latitude', 'longitude'])

for ind in station_cell_mapper_df.index:
    
    try:
      
  	  print(station_cell_mapper_df['station_id'][ind], station_cell_mapper_df['cell_id'][ind])
  	  current_cell_id = station_cell_mapper_df['cell_id'][ind]
  	  print("collecting ", current_cell_id)
  	  single_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/modis/{column_name}_{current_cell_id}.csv"

  	  if os.path.exists(single_csv_file):
  	    print("exists skipping..")
  	    continue

  	  longitude = station_cell_mapper_df['lon'][ind]
  	  latitude = station_cell_mapper_df['lat'][ind]

  	  # identify a 500 meter buffer around our Point Of Interest (POI)
  	  poi = ee.Geometry.Point(longitude, latitude).buffer(30)

  	  def poi_mean(img):
  	      reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30)
  	      mean = reducer.get(var_name)
  	      return img.set('date', img.date().format()).set(column_name,mean)
        
  	  viirs1 = ee.ImageCollection(product_name).filterDate('2013-01-01','2017-12-31')
  	  poi_reduced_imgs1 = viirs1.map(poi_mean)
  	  nested_list1 = poi_reduced_imgs1.reduceColumns(ee.Reducer.toList(2), ['date',column_name]).values().get(0)
  	  # dont forget we need to call the callback method "getInfo" to retrieve the data
  	  df1 = pd.DataFrame(nested_list1.getInfo(), columns=['date',column_name])
      
  	  viirs2 = ee.ImageCollection(product_name).filterDate('2018-01-01','2021-12-31')
  	  poi_reduced_imgs2 = viirs2.map(poi_mean)
  	  nested_list2 = poi_reduced_imgs2.reduceColumns(ee.Reducer.toList(2), ['date',column_name]).values().get(0)
  	  # dont forget we need to call the callback method "getInfo" to retrieve the data
  	  df2 = pd.DataFrame(nested_list2.getInfo(), columns=['date',column_name])
      

  	  df = pd.concat([df1, df2])
  	  df['date'] = pd.to_datetime(df['date'])
  	  df = df.set_index('date')
  	  df['cell_id'] = current_cell_id
  	  df['latitude'] = latitude
  	  df['longitude'] = longitude
  	  df.to_csv(single_csv_file)

  	  df_list = [all_cell_df, df]
  	  all_cell_df = pd.concat(df_list) # merge into big dataframe
      
    except Exception as e:
      
  	  print(e)
  	  pass
    
    
all_cell_df.to_csv(f"{homedir}/Documents/GitHub/SnowCast/data/{org_name}/{column_name}.csv")  



