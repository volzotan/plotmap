import sys
sys.path.append("..")
from svgwriter import SvgWriter

from datetime import datetime

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

TIMER_STRING = "{:<50s}: {:2.2f}s"

CITY = "Weimar"
NUM_ELEVATION_LINES = 15

MAP_WIDTH = 100
MAP_SIZE = [1000, 1000]
MAP_SCALE = MAP_SIZE[0] / float(MAP_WIDTH)

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

def simplify_polygons(polygons):
    polygons_simplified = []

    for poly in polygons:
        new_poly = []

        if len(poly) < 20:
            continue

        if len(poly) < 30:
            polygons_simplified.append(poly)
            continue

        polygons_simplified.append(poly)

        # for i in range(0, len(poly)):
        #     if i%2 == 0:
        #         new_poly.append(poly[i])

        # polygons_simplified.append(new_poly)

    return polygons_simplified

timer_start = datetime.now()
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
print(TIMER_STRING.format("load shape data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
with rasterio.open('thueringen_50m.tif') as dataset:

    # print("indexes: {}".format(dataset.indexes))
    # print("size: {} x {}".format(dataset.width, dataset.height))
    # print({i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)})
    # print("bounds: {}".format(dataset.bounds))
    # print(dataset.transform)
    # print(dataset.crs)

    # plt.matshow(band)
    # plt.savefig("data1.png")

    # out_image, out_transform = mask.mask(dataset, shape, crop=True)
    # # out_meta = dataset.meta.copy()
    # # reduce the image dimensions from (1, x, y) to (x, y)
    # band = np.squeeze(np.asarray(out_image))

    band = dataset.read(1)

    print(TIMER_STRING.format("load raster data", (datetime.now()-timer_start).total_seconds()))
    timer_start = datetime.now()

    band = band.clip(min=0)

    MAP_SIZE = (band.shape[1], band.shape[0])
    MAP_SCALE = float(MAP_WIDTH) / MAP_SIZE[0]

    print(MAP_SCALE)

    min_elevation = np.min(band[band > 0])
    max_elevation = np.max(band)

    elevation_line_height = (max_elevation - min_elevation) / NUM_ELEVATION_LINES

    for i in range(0, NUM_ELEVATION_LINES):
        foo = band.copy()
        foo[foo > min_elevation + elevation_line_height*i] = 0
        contours = generate_elevation_lines(foo)
        elevation_lines.append(contours)

    print("data range, min: {:.2f} | max: {:.2f}".format(min_elevation, max_elevation))
    print(TIMER_STRING.format("process elevation lines", (datetime.now()-timer_start).total_seconds()))

    # plt.imshow(band)
    # plt.savefig("data1.png")

svg = SvgWriter("elevation_lines.svg", (MAP_SIZE[0]*MAP_SCALE, MAP_SIZE[1]*MAP_SCALE))

timer_start = datetime.now()

polys = []
for height_level in elevation_lines:
    for line in height_level:
        coords = []
        for pair in line:
            coords.append([pair[0][0], pair[0][1]])

        # close the polygon
        coords = coords + [[coords[0][0], coords[0][1]]]

        polys.append(coords)

polygons_simplified = simplify_polygons(polys)

for poly in polygons_simplified:
    for i in range(0, len(poly)):
        poly[i][0] *= MAP_SCALE
        poly[i][1] *= MAP_SCALE

print(TIMER_STRING.format("simplify polygons", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
for poly in polygons_simplified:
    svg.add_polygon(poly, stroke_width=0.25, opacity=0) # , opacity=0.02

print(TIMER_STRING.format("loading svgwriter", (datetime.now()-timer_start).total_seconds()))

print("map size: {:.2f} x {:.2f}".format(MAP_SIZE[0]*MAP_SCALE, MAP_SIZE[1]*MAP_SCALE))

svg.save()
