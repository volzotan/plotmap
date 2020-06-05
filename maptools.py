import math
from datetime import datetime

from svgwriter import SvgWriter

import shapely
from shapely.wkb import loads
from shapely import ops
from shapely.prepared import prep
from shapely.geometry import GeometryCollection, MultiLineString, LineString, Polygon, MultiPolygon

import numpy as np
import cv2

EQUATOR = 40075016.68557849
EQUATOR_HALF = EQUATOR / 2.0

class Converter(object):

    def __init__(self, map_center, map_size, map_size_scale):
        self.map_center_lat_lon = map_center

        self.map_center_m = self.convert_wgs_to_mercator(*map_center)

        self.map_center_m[0] *= EQUATOR
        self.map_center_m[1] *= EQUATOR

        self.map_size = map_size
        self.map_left = self.map_center_m[0] - self.map_size[0]/2.0
        self.map_top = self.map_center_m[1] - self.map_size[1]/2.0

        self.map_size_scale = map_size_scale

    def get_bounding_box(self): # in web mercator, i.e. meters
        return [
            self.map_center_m[0] - EQUATOR_HALF - (self.map_size[0] * map_size_scale)/2.0, 
            (self.map_center_m[1]  - EQUATOR_HALF) * -1 - (self.map_size[1] * map_size_scale)/2.0,
            self.map_center_m[0] - EQUATOR_HALF + (self.map_size[0] * map_size_scale)/2.0, 
            (self.map_center_m[1]  - EQUATOR_HALF) * -1  + (self.map_size[1] * map_size_scale)/2.0,
        ]

    # convert WGS84 (lat, lon) to (-1, 1) in WebMercator
    def convert_wgs_to_mercator(self, lat, lon):

        e = 1.0E-5

        if lon < -180:
            lon = -180 + e
        if lon > 180:
            lon = 180 - e
        if lat < -90:
            lat = -90 + e
        if lat > 90:
            lat = 90 - e

        x       = (lon + 180.0) / 360.0 

        latRad  = (lat * math.pi) / 180.0
        mercN   = math.log(math.tan((math.pi / 4.0) + (latRad / 2.0)))
        y       = 0.5 - (mercN / (2.0*math.pi))

        return [x, y]

    # convert WebMercator (meter) to (px) in map
    def convert_mercator_to_map(self, x, y):
        mapx = (x + EQUATOR_HALF) / self.map_size_scale
        mapy = ((y - EQUATOR_HALF) * -1) / self.map_size_scale

        return mapx, mapy

    # convert WGS84 (lat, lon) to (px) in map
    def convert_wgs_to_map(self, lat, lon):
        x, y = self.convert_wgs_to_mercator(lat, lon) # results in values in the range of -1 to 1
        x = x * self.map_size[0]
        y = y * self.map_size[1]
        return x, y

    def convert_mercator_to_map_list(self, xs, ys):
        mapxs = []
        mapys = []

        for i in range(0, len(xs)):
            mapx, mapy = self.convert_mercator_to_map(xs[i], ys[i])
            mapxs.append(mapx)
            mapys.append(mapy)

        return mapxs, mapys

    def convert_wgs_to_map_list(self, lats, lons):
        mapxs = []
        mapys = []

        for i in range(0, len(lats)):
            mapx, mapy = self.convert_wgs_to_map(lats[i], lons[i])
            mapxs.append(mapx)
            mapys.append(mapy)

        return mapxs, mapys

    def convert_wgs_to_map_list_lon_lat(self, lons, lats): # flipped lat/lon
        mapxs = []
        mapys = []

        for i in range(0, len(lats)):
            mapx, mapy = self.convert_wgs_to_map(lats[i], lons[i])
            mapxs.append(mapx)
            mapys.append(mapy)

        return mapxs, mapys



class Color(object):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Hatching(object):

    LEFT_TO_RIGHT   = 0x1 << 1
    RIGHT_TO_LEFT   = 0x1 << 2
    VERTICAL        = 0x1 << 3 
    HORIZONTAL      = 0x1 << 4

    @staticmethod
    def create_hatching(poly, distance=2.0, connect=False, hatching_type=RIGHT_TO_LEFT):
        bounds = poly.bounds

        # align the hatching lines on a common grid
        bounds = [
            bounds[0]-bounds[0]%distance,
            bounds[1]-bounds[1]%distance,
            bounds[2]-bounds[2]%distance + distance,
            bounds[3]-bounds[3]%distance + distance
        ]

        hatching_lines = []
        num_lines = 0 

        if hatching_type == Hatching.LEFT_TO_RIGHT or hatching_type == Hatching.RIGHT_TO_LEFT:
            num_lines = (bounds[2]-bounds[0]+bounds[3]-bounds[1]) / distance

        if hatching_type == Hatching.VERTICAL:
            num_lines = bounds[2]-bounds[0] / distance

        if hatching_type == Hatching.HORIZONTAL:
            num_lines = bounds[3]-bounds[1] / distance

        for i in range(0, int(num_lines)):
            
            line = None

            if hatching_type == Hatching.LEFT_TO_RIGHT:
                line = LineString([[bounds[0], bounds[1] + i*distance], [bounds[0] + i*distance, bounds[1]]])
            elif hatching_type == Hatching.RIGHT_TO_LEFT:
                line = LineString([[bounds[2], bounds[1] + i*distance], [bounds[2] - i*distance, bounds[1]]])
            elif hatching_type == Hatching.VERTICAL:
                line = LineString([[bounds[0] + i*distance, bounds[1]], [bounds[0] + i*distance, bounds[3]]])
            elif hatching_type == Hatching.HORIZONTAL:
                line = LineString([[bounds[0], bounds[1] + i*distance], [bounds[2], bounds[1] + i*distance]])

            line = poly.intersection(line)

            if line.length == 0:
                continue

            if line.is_empty:
                continue

            if type(line) is MultiLineString:
                hatching_lines = hatching_lines + list(line.geoms)
            elif type(line) is LineString:
                hatching_lines.append(line)
            elif type(line) is GeometryCollection:       
                # probably Point and LineString
                for geom in line.geoms:
                    if geom.length == 0:
                        continue

                    hatching_lines.append(geom)
            else:
                print("unknown Geometry: {}".format(line))

        if connect:
            connection_lines = []

            max_length = distance * 2

            flip = False

            for i in range(0, len(hatching_lines)-1):
                curr_x, curr_y = hatching_lines[i].coords.xy
                next_x, next_y = hatching_lines[i+1].coords.xy

                conn = None

                if not flip:
                    conn = LineString([[curr_x[1], curr_y[1]], [next_x[1], next_y[1]]])
                else:
                    conn = LineString([[curr_x[0], curr_y[0]], [next_x[0], next_y[0]]])

                flip = not flip

                if conn.length < max_length:
                    connection_lines.append(conn)

            hatching_lines = hatching_lines + connection_lines

        return hatching_lines

def shapely_polygon_to_list(poly):
    result_list = []

    result_list.append(list(poly.exterior.coords))
    for hole in list(poly.interiors):
        result_list.append(list(hole.coords))

    return result_list

# converts a python list to a postgres-syntax compatible value list in " WHERE foo IN ('a', 'b') "-style
def list_to_pg_str(l):
    ret = "'"
    for i in range(0, len(l)-1):
        ret += l[i]
        ret += "', '"
    ret += l[-1]
    ret += "'"

    return ret

# removes all contained-in duplicate polygons 
# modifies passed list and returns number of removed elements
def remove_duplicates(layer):

    indices = []
    for i in range(0, len(layer)):
        obj_prep = prep(layer[i])
        for j in range(i+1, len(layer)):
            obj_comp = layer[j]

            if j in indices:
                continue

            if obj_prep.contains(obj_comp):
                indices.append(j)

    for i in sorted(indices, reverse=True):
        del layer[i]   

    return len(indices)

def subtract_layer_from_layer(lower_layer, upper_layer, grow=0):
    lower_layer_copy = lower_layer.copy()
    result = []

    for i in range(0, len(upper_layer)):
        cutter = upper_layer[i]

        if grow > 0:
            cutter = cutter.buffer(grow)

        append_list = []
        cutter_prepared = prep(cutter)

        for low_layer_object in lower_layer_copy:
            if cutter_prepared.intersects(low_layer_object):
                zonk = low_layer_object.difference(cutter)
                if not zonk.is_empty:
                    append_list.append(zonk)
                    lower_layer_copy.remove(low_layer_object)

        for low_layer_object in result:
            if cutter_prepared.intersects(low_layer_object):
                zonk = low_layer_object.difference(cutter)
                if not zonk.is_empty:
                    append_list.append(zonk)
                    result.remove(low_layer_object)

        for item in append_list:
            if type(item) is MultiPolygon:
                for poly in list(item.geoms):
                    result.append(poly)
            else:
                result.append(item)

    return result

def clip_layer_by_box(layer, box):

    minx = box[0]
    miny = box[1]
    maxx = box[2]
    maxy = box[3]

    box = Polygon([
        [minx, miny], 
        [maxx, miny], 
        [maxx, maxy], 
        [minx, maxy]
    ])

    clipped = []

    for i in range(0, len(layer)):

        zonk = box.intersection(layer[i])

        if zonk.is_empty:
            continue

        if type(zonk) is MultiPolygon or type(zonk) is GeometryCollection:
            for poly in list(zonk.geoms):

                if not type(poly) is Polygon:
                    continue

                x, y = poly.exterior.coords.xy

                if min(x) < minx:
                    continue

                if max(x) > maxx: 
                    continue
                    
                if min(y) < miny:
                    continue
            
                if max(y) > maxy:
                    continue

                clipped.append(poly)
        else:
            clipped.append(zonk)

    return clipped

def unpack_multipolygon(item):
    result = []

    if type(item) in [MultiPolygon, GeometryCollection]:
        for g in list(item.geoms):
            if type(g) is Polygon:
                result.append(g)
            else:
                print("narf!")
                print(type(g))
    else:
        result.append(item)

    return result


def unpack_multipolygons(layer):
    result = []

    for item in layer:
        result = result + unpack_multipolygon(item)

    return result


"""
    Remove points in the given polygon, but only points which are not shared with the parent polygon
    (ie. parent may be an outline or a neighbouring polygon and no gaps between these should be produced)
"""
def simplify_polygon(poly, epsilon=0.1): #, parent=None):

    # nppoly = np.asarray(poly)

    # if parent is not None:
    #     npparent = np.asarray(parent)

    if type(poly) is Polygon:
        tmp = poly.exterior.coords
    else:
        tmp = poly

    tmp = np.asarray(tmp, dtype=np.float32)

    approximatedPolygon = cv2.approxPolyDP(tmp, epsilon, True)
    approximatedPolygon = np.concatenate(approximatedPolygon) 
    approximatedPolygon = approximatedPolygon.tolist()

    try:
        return Polygon(approximatedPolygon)
    except ValueError as ve:
        return approximatedPolygon
