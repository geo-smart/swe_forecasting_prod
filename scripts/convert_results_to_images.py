import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import timedelta, datetime
import numpy as np
import uuid
from snowcast_utils import day_index
import matplotlib.colors as mcolors

# Import utility functions and variables from 'snowcast_utils'
from snowcast_utils import work_dir, test_start_date

# Define a custom colormap with specified colors and ranges
colors = [
    (0.8627, 0.8627, 0.8627),  # #DCDCDC - 0 - 1
    (0.8627, 1.0000, 1.0000),  # #DCFFFF - 1 - 2
    (0.6000, 1.0000, 1.0000),  # #99FFFF - 2 - 4
    (0.5569, 0.8235, 1.0000),  # #8ED2FF - 4 - 6
    (0.4509, 0.6196, 0.8745),  # #739EDF - 6 - 8
    (0.4157, 0.4706, 1.0000),  # #6A78FF - 8 - 10
    (0.4235, 0.2784, 1.0000),  # #6C47FF - 10 - 12
    (0.5529, 0.0980, 1.0000),  # #8D19FF - 12 - 14
    (0.7333, 0.0000, 0.9176),  # #BB00EA - 14 - 16
    (0.8392, 0.0000, 0.7490),  # #D600BF - 16 - 18
    (0.7569, 0.0039, 0.4549),  # #C10074 - 18 - 20
    (0.6784, 0.0000, 0.1961),  # #AD0032 - 20 - 30
    (0.5020, 0.0000, 0.0000)   # #800000 - > 30
]

cmap_name = 'custom_snow_colormap'
custom_cmap = mcolors.ListedColormap(colors)

# Define the lat_lon_to_map_coordinates function
def lat_lon_to_map_coordinates(lon, lat, m):
    """
    Convert latitude and longitude coordinates to map coordinates.

    Args:
        lon (float or array-like): Longitude coordinate(s).
        lat (float or array-like): Latitude coordinate(s).
        m (Basemap): Basemap object representing the map projection.

    Returns:
        tuple: Tuple containing the converted map coordinates (x, y).
    """
    x, y = m(lon, lat)
    return x, y

# Define value ranges for color mapping
fixed_value_ranges = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30]

def create_color_maps_with_value_range(df_col, value_ranges=None):
    """
    Create a colormap for value ranges and map data values to colors.

    Args:
        df_col (pd.Series): A Pandas Series containing data values.
        value_ranges (list, optional): A list of value ranges for color mapping.
            If not provided, the ranges will be determined automatically.

    Returns:
        tuple: Tuple containing the color mapping and the updated value ranges.
    """
    new_value_ranges = value_ranges
    if value_ranges is None:
        max_value = df_col.max()
        min_value = df_col.min()
        if min_value < 0:
            min_value = 0
        step_size = (max_value - min_value) / 12

        # Create 10 periods
        new_value_ranges = [min_value + i * step_size for i in range(12)]
    
    print("new_value_ranges: ", new_value_ranges)
  
    # Define a custom function to map data values to colors
    def map_value_to_color(value):
        # Iterate through the value ranges to find the appropriate color index
        for i, range_max in enumerate(new_value_ranges):
            if value <= range_max:
                return colors[i]

        # If the value is greater than the largest range, return the last color
        return colors[-1]

    # Map predicted_swe values to colors using the custom function
    color_mapping = [map_value_to_color(value) for value in df_col.values]
    return color_mapping, new_value_ranges

def convert_csvs_to_images():
    """
    Convert CSV data to images with color-coded SWE predictions.

    Returns:
        None
    """
    global fixed_value_ranges
    data = pd.read_csv("/home/chetana/gridmet_test_run/test_data_predicted.csv")
    print("statistic of predicted_swe: ", data['predicted_swe'].describe())
    data['predicted_swe'].fillna(0, inplace=True)
    
    for column in data.columns:
        column_data = data[column]
        print(column_data.describe())
    
    # Create a figure with a white background
    fig = plt.figure(facecolor='white')

    lon_min, lon_max = -125, -100
    lat_min, lat_max = 25, 49.5

    m = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                projection='merc', resolution='i')

    x, y = m(data['lon'].values, data['lat'].values)
    print(data.columns)

    color_mapping, value_ranges = create_color_maps_with_value_range(data["predicted_swe"], fixed_value_ranges)
    
    # Plot the data using the custom colormap
    plt.scatter(x, y, c=color_mapping, cmap=custom_cmap, s=30, edgecolors='none', alpha=0.7)
    
    # Draw coastlines and other map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    reference_date = datetime(1900, 1, 1)
    day_value = day_index
    
    result_date = reference_date + timedelta(days=day_value)
    today = result_date.strftime("%Y-%m-%d")
    timestamp_string = result_date.strftime("%Y-%m-%d")
    
    # Add a title
    plt.title(f'Predicted SWE in the Western US - {today}', pad=20)

    # Add labels for latitude and longitude on x and y axes with smaller font size
    plt.xlabel('Longitude', fontsize=6)
    plt.ylabel('Latitude', fontsize=6)

    # Add longitude values to the x-axis and adjust font size
    x_ticks_labels = np.arange(lon_min, lon_max + 5, 5)
    x_tick_labels_str = [f"{lon:.1f}°W" if lon < 0 else f"{lon:.1f}°E" for lon in x_ticks_labels]
    plt.xticks(*m(x_ticks_labels, [lat_min] * len(x_ticks_labels)), fontsize=6)
    plt.gca().set_xticklabels(x_tick_labels_str)

    # Add latitude values to the y-axis and adjust font size
    y_ticks_labels = np.arange(lat_min, lat_max + 5, 5)
    y_tick_labels_str = [f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S" for lat in y_ticks_labels]
    plt.yticks(*m([lon_min] * len(y_ticks_labels), y_ticks_labels), fontsize=6)
    plt.gca().set_yticklabels(y_tick_labels_str)

    # Convert map coordinates to latitude and longitude for y-axis labels
    y_tick_positions = np.linspace(lat_min, lat_max, len(y_ticks_labels))
    y_tick_positions_map_x, y_tick_positions_map_y = lat_lon_to_map_coordinates([lon_min] * len(y_ticks_labels), y_tick_positions, m)
    y_tick_positions_lat, _ = m(y_tick_positions_map_x, y_tick_positions_map_y, inverse=True)
    y_tick_positions_lat_str = [f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S" for lat in y_tick_positions_lat]
    plt.yticks(y_tick_positions_map_y, y_tick_positions_lat_str, fontsize=6)

    # Create custom legend elements using the same colormap
    legend_elements = [Patch(color=colors[i], label=f"{value_ranges[i]} - {value_ranges[i+1]-1}" if i < len(value_ranges) - 1 else f"> {value_ranges[-1]}") for i in range(len(value_ranges))]

    # Create the legend outside the map
    legend = plt.legend(handles=legend_elements, loc='upper left', title='Legend', fontsize=8)
    legend.set_bbox_to_anchor((1.01, 1)) 

    # Remove the color bar
    #plt.colorbar().remove()

    plt.text(0.98, 0.02, 'Copyright © SWE Wormhole Team',
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gcf().transFigure, fontsize=6, color='black')

    # Set the aspect ratio to 'equal' to keep the plot at the center
    plt.gca().set_aspect('equal', adjustable='box')

    # Adjust the bottom and top margins to create more white space between the title and the plot
    plt.subplots_adjust(bottom=0.15, right=0.80)  # Adjust right margin to accommodate the legend
    # Show the plot or save it to a file
    new_plot_path = f'/home/chetana/gridmet_test_run/predicted_swe-{test_start_date}.png'
    print(f"The new plot is saved to {new_plot_path}")
    plt.savefig(new_plot_path)
    # plt.show()  # Uncomment this line if you want to display the plot directly instead of saving it to a file

def convert_csvs_to_images_simple():
    """
    Convert CSV data to simple scatter plot images for predicted SWE.

    Returns:
        None
    """
    selected_date = datetime.strptime(test_start_date, "%Y-%m-%d")
    var_name = "predicted_swe"
    test_csv = "/home/chetana/gridmet_test_run/test_data_predicted.csv"
    result_var_df = pd.read_csv(test_csv)
    result_var_df.replace('--', pd.NA, inplace=True)
    result_var_df.dropna(inplace=True)
    result_var_df[var_name] = pd.to_numeric(result_var_df[var_name], errors='coerce')
    
    colormaplist, value_ranges = create_color_maps_with_value_range(result_var_df[var_name], fixed_value_ranges)

    # Create a scatter plot
    plt.scatter(result_var_df["lon"].values, 
                result_var_df["lat"].values, 
                label='Predicted SWE', 
                c=result_var_df['predicted_swe'], 
                cmap='viridis', 
                s=1, 
                edgecolor='none',
               )

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Predicted SWE')  # Label for the colorbar
    
    # Add labels and a legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'SWE Prediction Map {test_start_date}')
    plt.legend()

    res_png_path = f"{work_dir}/testing_output/{str(selected_date.year)}_{var_name}_{test_start_date}.png"
    plt.savefig(res_png_path)
    print(f"test image is saved at {res_png_path}")
    plt.close()

# Uncomment the function call you want to use:
# convert_csvs_to_images()
convert_csvs_to_images_simple()

