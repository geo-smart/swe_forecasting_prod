#!/bin/bash

# Specify the file containing the download links
input_file="/home/chetana/gridmet_test_run/amsr/download_links.txt"

# Specify the base wget command with common options
base_wget_command="wget --http-user=<your_username> --http-password=<your_password> --load-cookies /home/chetana/gridmet_test_run/amsr/mycookies.txt --save-cookies mycookies.txt --keep-session-cookies --no-check-certificate -$

# Specify the output directory for downloaded files
output_directory="/home/chetana/gridmet_test_run/amsr"

# Ensure the output directory exists
mkdir -p "$output_directory"

# Loop through each line (URL) in the input file and download it using wget
while IFS= read -r url; do
    echo "Downloading: $url"
    $base_wget_command -P "$output_directory" "$url"
done < "$input_file"
