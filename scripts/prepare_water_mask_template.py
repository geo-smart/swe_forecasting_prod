# prepare the template tifs for the water mask from MODIS

import requests
from bs4 import BeautifulSoup
import os
import subprocess
import csv
from datetime import datetime, timedelta

output_csv_file = "/home/chetana/gridmet_test_run/mod44w_water_mask.csv"

# Function to extract snow cover value at a given lat lon
def extract_snow_cover_value(geotiff_path, lon, lat):
    gdallocationinfo_cmd = [
        "gdallocationinfo",
        "-valonly",
        geotiff_path,
        str(lon),
        str(lat)
    ]
    result = subprocess.run(gdallocationinfo_cmd, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

def generate_template():
    # Define the date range for data extraction
    start_date = datetime(2018,1, 1)
    end_date = datetime(2018, 1, 3)

    # Load the CSV file with latitude and longitude coordinates
    csv_file_path = "/home/chetana/gridmet_test_run/station_cell_mapping.csv"

    # CSV file header
    csv_header = ["Date", "Latitude", "Longitude", "Snow Cover Value"]

    # Loop through the date range
    current_date = start_date
    while current_date <= end_date:
        extracted_data = list()
        # URL and reference link with dynamic date
        date_str = current_date.strftime("%Y.%m.%d")
        url = f"https://e4ftl01.cr.usgs.gov/MOLT/MOD44W.061/{date_str}/"
        reference_link = f"https://e4ftl01.cr.usgs.gov/MOLT/MOD10A1.061/{date_str}/"
        
        print("url = ", url)
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all <a> tags (links) on the page
            links = soup.find_all("a")
            
            # Filter the links to keep only those with the .hdf extension
            all_hdf_files = [link.get("href") for link in links if link.get("href").endswith(".hdf")]
            
            # Define the sinusoidal tiles of interest
            sinusoidal_tiles = ["h08v04", "h08v05", 
                                "h09v04", "h09v05", 
                                "h10v04", "h10v05", 
                                "h11v04", "h11v05", 
                                "h12v04", "h12v05", 
                                "h13v04", "h13v05", 
                                "h15v04", "h16v03", 
                                "h16v04"]
            
            # Filter the HDF files based on sinusoidal tiles
            filtered_hdf_files = [hdf_file for hdf_file in all_hdf_files if any(tile in hdf_file for tile in sinusoidal_tiles)]
            
            # List to store the paths of converted GeoTIFF files
            geotiff_files = []
            
            # Loop through the filtered HDF files and download/convert/delete them
            for hdf_file in filtered_hdf_files:
                # Construct the complete URL for the HDF file
                hdf_url = reference_link + hdf_file
                
                # Define the local filename to save the HDF file
                local_hdf_filename = hdf_file
                
                # Send an HTTP GET request to download the HDF file
                print("hdf_url = ", hdf_url)
                hdf_response = requests.get(hdf_url)
                
                # Check if the download was successful (HTTP status code 200)
                if hdf_response.status_code == 200:
                    with open(local_hdf_filename, "wb") as f:
                        f.write(hdf_response.content)
                    print(f"Downloaded {local_hdf_filename}")
                    
                    # Construct the output GeoTIFF file path and name in the same directory
                    local_geotiff_filename = os.path.splitext(local_hdf_filename)[0] + ".tif"
                    
                    # Run the gdal_translate command to convert HDF to GeoTIFF
                    gdal_translate_cmd = [
                        "gdal_translate",
                        "-of", "GTiff",
                        f"HDF4_EOS:EOS_GRID:{local_hdf_filename}:MOD_Grid_Snow_500m:NDSI_Snow_Cover",
                        local_geotiff_filename
                    ]
                    
                    # Execute the gdal_translate command
                    subprocess.run(gdal_translate_cmd)
                    
                    # Append the path of the converted GeoTIFF to the list
                    geotiff_files.append(local_geotiff_filename)
                    
                    # Delete the original HDF file
    #                 os.remove(local_hdf_filename)
                    
                    #print(f"Converted and deleted: {local_hdf_filename}")
                else:
                    pass
                    #print(f"Failed to download {local_hdf_filename}")
            
            # Merge all the GeoTIFF files into a single GeoTIFF
            merged_geotiff = "merged_geotiff.tif"
            gdal_merge_cmd = [
                "gdal_merge.py",
                "-o", merged_geotiff,
                "-of", "GTiff"
            ] + geotiff_files
            
            subprocess.run(gdal_merge_cmd)
            
            print(f"Merged all GeoTIFF files into {merged_geotiff}")
            
            # Loop through the CSV file with latitude and longitude coordinates
            with open(csv_file_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip the header row
                
                for row in csv_reader:
                    lat = float(row[3])  # Assuming latitude is in the first column
                    lon = float(row[4])  # Assuming longitude is in the second column
                    
                    # Extract snow cover value
                    snow_cover_value = extract_snow_cover_value(merged_geotiff, lon, lat)
                    
                    # Append the extracted data to the list
                    extracted_data.append([date_str, lat, lon, snow_cover_value])
            
            # Delete the merged GeoTIFF file
    #         os.remove(merged_geotiff)
            print(f"Deleted {merged_geotiff}")
            
            # Delete individual GeoTIFF files after a successful merge
            #for geotiff_file in geotiff_files:
            #    try:
            #        os.remove(geotiff_file)
            #        print(f"Deleted {geotiff_file}")
            #    except Exception as e:
            #        pass
            
            # Append the extracted data to the CSV file after each date
            with open(output_csv_file, "a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(extracted_data)
            
            print(f"Extracted data appended to {output_csv_file}")
        
        else:
            print("Failed to fetch the HTML content.")
        
        # Move to the next date
        current_date += timedelta(days=1)


if __name__ == "__main__":
  generate_template()


