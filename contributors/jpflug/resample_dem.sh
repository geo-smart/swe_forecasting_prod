#!/bin/bash
# this script will reproject and resample the western US dem, clip it, to match the exact spatial extent and resolution as the template tif

cd /home/chetana/gridmet_test_run

mkdir template_shp/

cp /home/chetana/western_us_geotiff_template.tif template_shp/

# generate the template shape
gdaltindex template.shp template_shp/*.tif

gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -tr 0.036 0.036  -cutline template.shp -crop_to_cutline -overwrite output_4km.tif output_4km_clipped.tif
    
gdalinfo output_4km_clipped.tif

