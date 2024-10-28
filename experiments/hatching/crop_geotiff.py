import numpy as np
import rasterio
from matplotlib import pyplot
import tifffile

GEOTIFF_PATH = "data/GebcoToBlender/gebco_mosaic.tif"
OUTPUT_PATH = "data/gebco_crop.tif"
CENTER_LON_LAT = 139.839478, 35.652832
CROP_SIZE = [1000, 1000]

with rasterio.open(GEOTIFF_PATH) as dataset:
    band = dataset.read(1)
    y, x = dataset.index(*CENTER_LON_LAT)

    crop = band[y-CROP_SIZE[1]//2:y+CROP_SIZE[1]//2, x-CROP_SIZE[0]//2:x+CROP_SIZE[0]//2]

    tifffile.imwrite(OUTPUT_PATH, crop)

    pyplot.imshow(crop, cmap='pink')
    pyplot.show()