# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernel

from all_dependencies import *
from snowcast_utils import *

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate() # This must be run in the terminal instead of Geoweaver. Geoweaver doesn't support prompts.
    ee.Initialize()

# Read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
# Read grid cell
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# Read grid cell
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
submission_format_df = pd.read_csv(submission_format_file, header=0, index_col=0)

print("submission_format_df shape: ", submission_format_df.shape)

all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
all_cell_coords_df = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

# Start_date = "2022-04-20" # Test_start_date
start_date = findLastStopDate(f"{github_dir}/data/sat_testing/sentinel1", "%Y-%m-%d %H:%M:%S")
end_date = test_end_date

org_name = 'sentinel1'
product_name = 'COPERNICUS/S1_GRD'
var_name = 'VV'
column_name = 's1_grd_vv'

final_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/sat_testing/{org_name}/{column_name}_{start_date}_{end_date}.csv"
print(f"Results will be saved to {final_csv_file}")

if os.path.exists(final_csv_file):
    #print("exists skipping..")
    #exit()
    os.remove(final_csv_file)

all_cell_df = pd.DataFrame(columns=['date', column_name, 'cell_id', 'latitude', 'longitude'])

for current_cell_id in submission_format_df.index:

    try:
        #print("collecting ", current_cell_id)

        longitude = all_cell_coords_df['lon'][current_cell_id]
        latitude = all_cell_coords_df['lat'][current_cell_id]

        # Identify a 500-meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(longitude, latitude).buffer(10)

        viirs = ee.ImageCollection(product_name) \
            .filterDate(start_date, end_date) \
            .filterBounds(poi) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV')

        def poi_mean(img):
            reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi)
            mean = reducer.get(var_name)
            return img.set('date', img.date().format()).set(column_name, mean)

        poi_reduced_imgs = viirs.map(poi_mean)

        nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date', column_name]).values().get(0)

        # Don't forget we need to call the callback method "getInfo" to retrieve the data
        df = pd.DataFrame(nested_list.getInfo(), columns=['date', column_name])

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df['cell_id'] = current_cell_id
        df['latitude'] = latitude
        df['longitude'] = longitude

        df_list = [all_cell_df, df]
        all_cell_df = pd.concat(df_list)  # Merge into a big dataframe

    except Exception as e:

        #print(e)
        pass

all_cell_df.to_csv(final_csv_file)

