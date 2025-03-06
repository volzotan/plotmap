import rasterio
from matplotlib import pyplot
import tifffile
import cv2

# GEOTIFF_PATH = "data/GebcoToBlender/gebco_mosaic.tif"
GEOTIFF_PATH = "data/GebcoToBlender/fullsize_reproject.tif"
OUTPUT_PATH = "data/gebco_crop.tif"

# CENTER_LON_LAT = [139.839478, 35.652832]
# CENTER_LON_LAT = [0, 0] # Null Island
# CENTER_LON_LAT = [-45, -25]  # North Atlantic ridge
# CENTER_LON_LAT = [-20, -40]
# CENTER_LON_LAT = [146.7996121929291, -43.512453249372804]
# CENTER_LON_LAT = [175.9266138226686, 52.354513899345434] # Aleutian trench
# CENTER_LON_LAT = [-14.373269321117954, -7.9386538081877935] # Ascension Island

CROP_SIZE = [10000, 10000]
CENTER = [0.4, 0.4]
CENTER = [0.6, 0.4]

OUTPUT_SIZE = CROP_SIZE

with rasterio.open(GEOTIFF_PATH) as dataset:
    band = dataset.read(1)
    # y, x = dataset.index(*CENTER_LON_LAT)

    x = int(CENTER[0] * band.shape[1])
    y = int(CENTER[1] * band.shape[0])

    crop = band[
        y - CROP_SIZE[1] // 2 : y + CROP_SIZE[1] // 2,
        x - CROP_SIZE[0] // 2 : x + CROP_SIZE[0] // 2,
    ]

    crop = cv2.resize(crop, OUTPUT_SIZE)

    tifffile.imwrite(OUTPUT_PATH, crop)
