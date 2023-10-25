import distutils.dir_util
from snowcast_utils import work_dir
import os
import shutil


print("move the plots and the results into the http folder")

source_folder = f"{work_dir}/var_comparison/"
destination_folder = f"/var/www/html/swe_forecasting/plots/"

# Copy the folder with overwriting existing files/folders
distutils.dir_util.copy_tree(source_folder, destination_folder, update=1)

print(f"Folder '{source_folder}' copied to '{destination_folder}' with overwriting.")


# copy the png from testing_output to plots
source_folder = f"{work_dir}/testing_output/"

# Ensure the destination folder exists, create it if necessary
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through the files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a PNG file
    if filename.endswith('.png'):
        # Build the source and destination file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Copy the file from the source to the destination
        shutil.copy(source_file, destination_file)
        print(f'Copied: {filename}')

