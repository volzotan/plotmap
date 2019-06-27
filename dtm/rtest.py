import sys
sys.path.append("..")
from svgwriter import SvgWriter
import maptools

from datetime import datetime
from functools import partial

import rasterio
import rasterio.features
import rasterio.warp
from rasterio import mask

import psycopg2
import pyproj
import numpy as np
import cv2
import matplotlib.pyplot as plt

import shapely
from shapely.wkb import loads
from shapely.ops import transform
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry import CAP_STYLE, JOIN_STYLE

DATASET_FILE = "thueringen_50m.tif"
# DATASET_FILE = "thueringen_20m.tif"
# DATASET_FILE = "nordrheinwestfalen_20m.tif"

DB_NAME = "import"
DB_PREFIX = "osm_"

TIMER_STRING = "{:<50s}: {:2.2f}s"

NUM_ELEVATION_LINES = 50

MAP_WIDTH = 500
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

        polygons_simplified.append(maptools.simplify_polygon(poly, epsilon=0.5))

    return polygons_simplified

def project_shape(shape):

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:3785'), # source coordinate system
        pyproj.Proj(init='epsg:3044')) # destination coordinate system

    return transform(project, shape)

def _reproject(shape, out_transform):

    m = np.linalg.inv(np.asarray(out_transform).reshape(3, 3))
    m = [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[1, 2], m[0, 2]]
    shape = shapely.affinity.affine_transform(shape, m)

    return shape

def reproject_shape(shape, out_transform, map_scale, offset=None):

    shape = _reproject(shape, out_transform)

    # # mirror the shape
    # shape = shapely.affinity.affine_transform(shape, [1, 0, 0, -1, 0, 0])

    # # downscale 50m meter to pixel factor
    # shape = shapely.affinity.scale(shape, xfact=1/50, yfact=1/50, origin=(0, 0))

    # move shape to 0,0
    if offset is None:
        shape = shapely.affinity.translate(shape, xoff=-shape.bounds[0], yoff=-shape.bounds[1])
    else:
        shape = shapely.affinity.translate(shape, xoff=-offset[0], yoff=-offset[1])

    # downscale own map scale factor
    shape = shapely.affinity.scale(shape, xfact=MAP_SCALE, yfact=MAP_SCALE, origin=(0, 0))

    return shape

def cutout(polys, cutshape):
    result = []

    invalid = 0

    for poly in polys:

        background_objects = []

        if type(poly) is MultiPolygon:
            background_objects = background_objects + unpack_multipolygon(poly)
        elif type(poly) is GeometryCollection:
            background_objects = background_objects + unpack_multipolygon(poly)
        elif type(poly) is Polygon:
            background_objects.append(poly)
        elif type(poly) is list:
            if len(poly) < 3:
                continue

            background_objects.append(Polygon(poly))
        else:
            print(type(poly))

        for background_object in background_objects:
            try:
                # cut_lines = maptools.unpack_multipolygon(pline.difference(shape))

                if not background_object.is_valid:
                    background_object = background_object.buffer(0)

                result_object = background_object.difference(cutshape)
                for obj in maptools.unpack_multipolygon(result_object):
                    result.append(obj)
            except shapely.errors.TopologicalError as tpe:
                invalid += 1

    if invalid > 0:
        print("invalid polygons: {}".format(invalid))

    return result


boundary_shape = None
cutout_shapes = []

# BOUNDARY = "Weimar"
BOUNDARY = "Thüringen"
curs.execute("""
    SELECT geometry 
    FROM {0}.{1}admin 
    WHERE name='{2}' 
    ORDER BY admin_level ASC
""".format(DB_NAME, DB_PREFIX, BOUNDARY))
results = curs.fetchall()
boundary_shape = loads(results[0][0], hex=True)

if boundary_shape is not None:
    boundary_shape = project_shape(boundary_shape)

CUTOUT = ["Weimar", "Erfurt"]
for coutout_name in CUTOUT:
    curs.execute("""
        SELECT geometry 
        FROM {0}.{1}admin 
        WHERE name='{2}' 
        ORDER BY admin_level ASC
    """.format(DB_NAME, DB_PREFIX, coutout_name))
    results = curs.fetchall()
    shape = loads(results[0][0], hex=True)
    cutout_shapes.append(shape)

shapes_projected = []
for shape in cutout_shapes:
    shapes_projected.append(project_shape(shape))
cutout_shapes = shapes_projected

timer_start = datetime.now()
with rasterio.open(DATASET_FILE) as dataset:

    # print("indexes: {}".format(dataset.indexes))
    # print("size: {} x {}".format(dataset.width, dataset.height))
    # print({i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)})
    # print("bounds: {}".format(dataset.bounds))
    # print(dataset.transform)
    # print(dataset.crs)

    # plt.matshow(band)
    # plt.savefig("data1.png")

    if boundary_shape is not None:
        out_image, out_transform = mask.mask(dataset, [boundary_shape], crop=True)
        # out_meta = dataset.meta.copy()

        # reduce the image dimensions from (1, x, y) to (x, y)
        band = np.squeeze(np.asarray(out_image))

        MAP_SIZE = (band.shape[1], band.shape[0])
        MAP_SCALE = float(MAP_WIDTH) / MAP_SIZE[0]

        reprojected_boundary = _reproject(boundary_shape, out_transform)
        offset = [reprojected_boundary.bounds[0], reprojected_boundary.bounds[1]]

        boundary_shape = reproject_shape(boundary_shape, out_transform, MAP_SCALE)

        shapes_reprojected = []
        for shape in cutout_shapes:
            shapes_reprojected.append(reproject_shape(shape, out_transform, MAP_SCALE, offset=offset))
        cutout_shapes = shapes_reprojected
 
    else:
        band = dataset.read(1)

        MAP_SIZE = (band.shape[1], band.shape[0])
        MAP_SCALE = float(MAP_WIDTH) / MAP_SIZE[0]

    print(TIMER_STRING.format("load raster data", (datetime.now()-timer_start).total_seconds()))
    timer_start = datetime.now()

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

# Problem Douglas Peucker may produce self intersecting polygons...
polygons_simplified = simplify_polygons(polys)
# polygons_simplified = polys

for poly in polygons_simplified:
    for i in range(0, len(poly)):
        poly[i][0] *= MAP_SCALE
        poly[i][1] *= MAP_SCALE

print(TIMER_STRING.format("simplify polygons", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
elevation_lines_cutout = polygons_simplified
# skipped_lines = 0
# malformed_polygons = 0
for cutshape in cutout_shapes:
    elevation_lines_cutout = cutout(elevation_lines_cutout, cutshape)
    
polygons_simplified = []
for line in elevation_lines_cutout:
    for poly in maptools.shapely_polygon_to_list(line):
        polygons_simplified.append(poly)

print(TIMER_STRING.format("cutout elevation lines", (datetime.now()-timer_start).total_seconds()))
# print("cutout elevation line errors: skipped {} | malformed {}".format(skipped_lines, malformed_polygons))

timer_start = datetime.now()

for poly in polygons_simplified:
    svg.add_polygon(poly, stroke_width=0.25, opacity=0) # , opacity=0.02

# for cutout_poly in cutout_shapes:
#     for poly in maptools.shapely_polygon_to_list(cutout_poly):
#         svg.add_polygon(poly, stroke_width=0.25, opacity=0.5)

if boundary_shape is not None:
    # shape_buffered = shape.buffer(10)
    # for poly in maptools.shapely_polygon_to_list(shape_buffered):
    #     svg.add_polygon(poly, stroke_width=0.25, opacity=0, repeat=20, wiggle=2)
    #     # svg.add_polygon(simplify_polygons([poly])[0], stroke_width=0.25, opacity=0, repeat=20, wiggle=2)

    for i in range(0, 3):
        shape_buffered = boundary_shape.buffer(i/3)
        for poly in maptools.shapely_polygon_to_list(shape_buffered):
            svg.add_polygon(poly, stroke_width=0.25, opacity=0)

    buffer_params = {
        "cap_style": CAP_STYLE.flat, 
        "join_style": JOIN_STYLE.mitre,
        "mitre_limit": 5.0
    }

    shape = boundary_shape.buffer(2, **buffer_params)

    shape_buffered1 = boundary_shape.buffer(4, **buffer_params)
    # for poly in maptools.shapely_polygon_to_list(shape_buffered1):
    #     svg.add_polygon(poly, stroke_width=0.25, opacity=0)

    shape_buffered2 = boundary_shape.buffer(6, **buffer_params)
    # for poly in maptools.shapely_polygon_to_list(shape_buffered2):
    #     svg.add_polygon(poly, stroke_width=0.25, opacity=0)

    shape_outline1 = shape_buffered1.difference(shape)
    for line in maptools.Hatching.create_hatching(shape_outline1, distance=2.5, connect=False):
        svg.add_line(maptools.shapely_linestring_to_list(line), stroke_width=0.25)

    shape_outline2 = shape_buffered2.difference(shape_buffered1)
    for line in maptools.Hatching.create_hatching(shape_outline2, distance=5.0, connect=False):
        svg.add_line(maptools.shapely_linestring_to_list(line), stroke_width=0.25)

print(TIMER_STRING.format("loading svgwriter", (datetime.now()-timer_start).total_seconds()))

print("map size: {:.2f} x {:.2f}".format(MAP_SIZE[0]*MAP_SCALE, MAP_SIZE[1]*MAP_SCALE))

svg.save()
