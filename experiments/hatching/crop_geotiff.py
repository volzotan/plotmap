import numpy as np
import rasterio
from matplotlib import pyplot
import tifffile
import cv2

GEOTIFF_PATH = "data/GebcoToBlender/gebco_mosaic.tif"
OUTPUT_PATH = "data/gebco_crop.tif"

CENTER_LON_LAT = [139.839478, 35.652832]
CENTER_LON_LAT = [0, 0]
# CENTER_LON_LAT = [146.7996121929291, -43.512453249372804]
# CENTER_LON_LAT = [175.9266138226686, 52.354513899345434] # Aleutian trench

# CROP_SIZE = [5000, 5000]
# OUTPUT_SIZE = [2000, 2000]

CROP_SIZE = [5000, 5000]
OUTPUT_SIZE = CROP_SIZE

# CROP_SIZE = [1000, 1000]
# OUTPUT_SIZE = CROP_SIZE

with rasterio.open(GEOTIFF_PATH) as dataset:
    band = dataset.read(1)
    y, x = dataset.index(*CENTER_LON_LAT)

    crop = band[y-CROP_SIZE[1]//2:y+CROP_SIZE[1]//2, x-CROP_SIZE[0]//2:x+CROP_SIZE[0]//2]

    crop = cv2.resize(crop, OUTPUT_SIZE)

    tifffile.imwrite(OUTPUT_PATH, crop)

    pyplot.imshow(crop, cmap='pink')
    pyplot.show()