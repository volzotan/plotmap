import sys
sys.path.append("..")
from svgwriter import SvgWriter

import rasterio
import rasterio.features
import rasterio.warp
from rasterio import mask
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

import shapely
from shapely.wkb import loads

import pyproj
from functools import partial
from shapely.ops import transform

import cv2

DB_NAME = "import"
DB_PREFIX = "osm_"

CITY = "Weimar"

NUM_ELEVATION_LINES = 50

conn = psycopg2.connect(database='osm', user='osm')
curs = conn.cursor()

elevation_lines = []

def generate_elevation_lines(image):

    lines = []

    kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(image, kernel, iterations = 1)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    thres = closing.astype(np.uint8)
    thres[thres > 0] = 1
    contours = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for item in contours[0]:

        if type(item) is list:
            for i in item:
                print(type(i))

            continue

        lines.append(item)

    return lines

curs.execute("""
    SELECT geometry 
    FROM {0}.{1}admin 
    WHERE name='{2}' 
    ORDER BY admin_level ASC
""".format(DB_NAME, DB_PREFIX, CITY))
results = curs.fetchall()

shape = loads(results[0][0], hex=True)

project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3785'), # source coordinate system
    pyproj.Proj(init='epsg:3044')) # destination coordinate system

shape = [transform(project, shape)]

with rasterio.open('thueringen_20m.tif') as dataset:

    # print("indexes: {}".format(dataset.indexes))
    # print("size: {} x {}".format(dataset.width, dataset.height))
    # print({i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)})
    # print("bounds: {}".format(dataset.bounds))
    # print(dataset.transform)
    # print(dataset.crs)

    # plt.matshow(band)
    # plt.savefig("data1.png")

    out_image, out_transform = mask.mask(dataset, shape, crop=True)
    # out_meta = dataset.meta.copy()
    # reduce the image dimensions from (1, x, y) to (x, y)
    band = np.squeeze(np.asarray(out_image))

    # band = dataset.read(1)

    band = band.clip(min=0)

    min_elevation = np.min(band[band > 0])
    max_elevation = np.max(band)

    elevation_line_height = (max_elevation - min_elevation) / NUM_ELEVATION_LINES

    for i in range(0, NUM_ELEVATION_LINES):
        foo = band.copy()
        foo[foo > min_elevation + elevation_line_height*i] = 0
        contours = generate_elevation_lines(foo)
        elevation_lines.append(contours)

    print("data range, min: {:.2f} | max: {:.2f}".format(min_elevation, max_elevation))

    # plt.imshow(band)
    # plt.savefig("data1.png")

MAP_SIZE = (1000, 1000)
svg = SvgWriter("elevation_lines.svg", MAP_SIZE)

for height_level in elevation_lines:
    for line in height_level:
        coords = []
        for pair in line:
            coords.append([pair[0][0], pair[0][1]])
        svg.add_polygon(coords, opacity=0.0)

svg.save()
