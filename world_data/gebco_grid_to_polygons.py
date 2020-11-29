import rasterio
# import rasterio.features
# import rasterio.warp
# from rasterio import mask
from rasterio.transform import Affine

# import pyproj
import numpy as np
import cv2

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform

from datetime import datetime
import os

import pickle

import fiona
from fiona.crs import from_epsg
from collections import OrderedDict

""" ----------------------------------------------------------------------

Projection Info:

GEBCO data is WGS84 - Mercator (EPSG 4326)
GeoJSON output is the same

---------------------------------------------------------------------- """

BASE_DIR = "gebco_2020_geotiff"
DATASET_FILES = [
    "gebco_2020_n0.0_s-90.0_w-90.0_e0.0.tif",
    "gebco_2020_n0.0_s-90.0_w-180.0_e-90.0.tif",
    "gebco_2020_n0.0_s-90.0_w0.0_e90.0.tif",
    "gebco_2020_n0.0_s-90.0_w90.0_e180.0.tif",
    "gebco_2020_n90.0_s0.0_w-90.0_e0.0.tif",
    "gebco_2020_n90.0_s0.0_w-180.0_e-90.0.tif",
    "gebco_2020_n90.0_s0.0_w0.0_e90.0.tif",
    "gebco_2020_n90.0_s0.0_w90.0_e180.0.tif"
]

MIN_VAUE            = 0
MAX_VALUE           = 9000
NUM_ELEVATION_LINES = 90

MIN_AREA                    = 4 # in square-pixel
MAX_SIMPLIFICATION_ERROR    = 0.1 # in px
MORPH_KERNEL_SIZE           = 5

TIMER_STRING                = "{:<60s}: {:2.2f}s"

# if ALLOW_OVERLAP is true, the polygon for 2000-3000 meters will contain 
# the polygon for 3000-4000m too. (i.e. polygons overlap like pyramids)

ALLOW_OVERLAP               = True

def generate_elevation_lines(image):

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    # erosion = cv2.erode(image, kernel, iterations = 1)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    thres = closing.astype(np.uint8)
    thres[thres > 0] = 1

    # retrieve contours with CCOMP (only two levels of hierarchy, either hole or no hole)
    contours = cv2.findContours(thres, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    return contours

def transform_geometry(x, y):
    return dataset.xy(y, x) # openCVs row,col order ...

def get_holes_for_poly(contours, hierarchy, index):
    
    holes = []
    next_index = hierarchy[index][0]

    while True:
        points = []

        for coord in contours[next_index]:
            points.append(coord[0])

        holes.append(points)
        next_index = hierarchy[next_index][0]

        if next_index < 0:
            break
        
    return holes

for DATASET_FILE in DATASET_FILES:

    GEOJSON_FILE = os.path.join(BASE_DIR, DATASET_FILE[:-4] + ".{}_{}_{}".format(MIN_VAUE, MAX_VALUE, NUM_ELEVATION_LINES) + ".geojson")

    timer_start = datetime.now()

    with rasterio.open(os.path.join(BASE_DIR, DATASET_FILE)) as dataset:

        # print(dataset)
        print("dataset count: {}".format(dataset.count))
        print("dataset w,h: {},{}".format(dataset.width, dataset.height))
        print("dataset bounds: {}".format(dataset.bounds))
        print("dataset CRS: {}".format(dataset.crs))
        print("dataset transform: \n{}".format(dataset.transform))

        band = dataset.read(1)

        min_elevation = np.min(band) #[band > 0])
        max_elevation = np.max(band)

        print("min: {} | max: {}".format(min_elevation, max_elevation))

        elevation_line_height = abs((MAX_VALUE - MIN_VAUE) / NUM_ELEVATION_LINES)

        contour_layers = []

        for i in range(0, NUM_ELEVATION_LINES):

            threshold_value_low = MIN_VAUE + elevation_line_height*i
            threshold_value_high = threshold_value_low + elevation_line_height

            print("elevation line {}: {} -> {}".format(i, threshold_value_low, threshold_value_high))

            _, bin_band_lower = cv2.threshold(band, threshold_value_low, max_elevation, cv2.THRESH_BINARY)
            
            bin_band = bin_band_lower

            if not ALLOW_OVERLAP:

                _, bin_band_higher = cv2.threshold(band, threshold_value_high, max_elevation, cv2.THRESH_BINARY)

                bin_band_lower[bin_band_higher > 0] = 0
                bin_band = bin_band_lower

            contours = generate_elevation_lines(bin_band)
            contour_layers.append(contours)

            # foo = band.copy()
            # foo[foo <= value] = 0
            # foo[foo > value] = 1
            # contours = generate_elevation_lines()
            # contour_layers.append(contours)

        layers = []

        for contours, contour_hierarchy in contour_layers:

            if len(contours) == 0:
                layers.append([])
                continue

            contour_hierarchy = contour_hierarchy[0]
            polygons = []

            holes = {}

            for i in range(0, len(contour_hierarchy)):
                # print("processing contour (hole search) {}/{}".format(i, len(contour_hierarchy))) 

                parent_index = contour_hierarchy[i][3]       

                if parent_index >= 0:
                    if not parent_index in holes:
                        holes[parent_index] = []

                    points = []
                    for coord in contours[i]:
                        points.append(coord[0])   

                    holes[parent_index].append(points)

            for i in range(0, len(contour_hierarchy)):

                # print("processing contour {}/{}".format(i, len(contour_hierarchy)))

                child_index = contour_hierarchy[i][2]
                parent_index = contour_hierarchy[i][3]

                if parent_index >= 0: # is hole
                    continue

                points_exterior = []
                points_interiors = []

                if child_index >= 0: # has holes
                    points_interiors = holes[i] # get_holes_for_poly(contours, contour_hierarchy, child_index)
                else: # has no holes
                    pass
            
                for coord in contours[i]:
                    points_exterior.append(coord[0])

                if len(points_exterior) < 3:
                    print("warning: too few points for polygon: {}".format(len(points_exterior)))
                    continue

                sanitized_points_interiors = []
                for hole in points_interiors:
                    if len(hole) >= 3:
                        sanitized_points_interiors.append(hole)
                points_interiors = sanitized_points_interiors
                
                poly = Polygon(points_exterior, points_interiors)

                if poly.area < MIN_AREA:
                    print("warning: polygon too small")
                    continue

                poly = poly.simplify(MAX_SIMPLIFICATION_ERROR)

                if not poly.is_valid:

                    poly = poly.buffer(0)

                    if not poly.is_valid:
                        print("warning: polygon not valid (size: {})".format(len(poly.exterior.coords)))
                        continue

                poly = transform(transform_geometry, poly)   

                if type(poly) is Polygon:
                    polygons.append(poly)
                elif type(poly) is MultiPolygon:
                    for g in poly.geoms:
                        polygons.append(g)
                else:
                    print("warning: polygon is not a Polygon (type: {})".format(type(poly)))
                    continue

            layers.append(polygons)

        for i in range(0, len(layers)):
            print("layer {} [{} to {}]: polys: {}".format(i, MIN_VAUE + elevation_line_height*i, MIN_VAUE + elevation_line_height*(i+1), len(layers[i])))

        # remove overlaps
        # new_layers = []
        # for i in range(len(layers), 0):
        #     higher_layer = layers[i]
        #     lower_layer = layers[i-1]

        #     for higher_poly in higher_layer:

        # save GeoJSON

        schema = {
            "geometry": "Polygon",
            "properties": OrderedDict([
                ("min_height", "int"),
                ("max_height", "int"),
                ("layer", "int")
            ])
        }

        # driver needs to be GeoJSON (even though it doesn't support layers) 
        # because ESRI shapefiles have no support for polygons with holes

        options = {
            "crs": from_epsg(4326),
            "driver": "GeoJSON", 
            "schema": schema
        }

        with fiona.open(GEOJSON_FILE, "w", **options) as sink:

            for i in range(0, len(layers)):
                polys = layers[i]
                records = []

                for p in polys:

                    records.append({
                        "geometry": {
                            "coordinates": [p.exterior.coords] + [hole.coords for hole in p.interiors],
                            "type": "Polygon"
                        },
                        "properties": OrderedDict([
                            ("min_height", MIN_VAUE + elevation_line_height*i),
                            ("max_height", MIN_VAUE + elevation_line_height*(i+1)),
                            ("layer", i)
                        ])
                    })

                sink.writerecords(records)

        print(TIMER_STRING.format("processing {}".format(DATASET_FILE), (datetime.now()-timer_start).total_seconds()))

