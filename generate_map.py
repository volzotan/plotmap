import math
from datetime import datetime

from svgwriter import SvgWriter

import psycopg2
import shapely
from shapely.wkb import loads
from shapely import ops

DB_NAME = "import"
DB_PREFIX = "osm_"

conn = psycopg2.connect(database='osm', user='osm')
curs = conn.cursor()

class Converter(object):

    EQUATOR = 40075016.68557849
    EQUATOR_HALF = EQUATOR / 2.0

    def __init__(self, map_center, map_size, map_size_scale):
        self.map_center_lat_lon = map_center

        self.map_center_m = self._convert_lat_lon(*map_center)

        self.map_center_m[0] *= self.EQUATOR
        self.map_center_m[1] *= self.EQUATOR # right value in meter?

        self.map_size = [map_size[0] * map_size_scale, map_size[1] * map_size_scale]
        self.map_left = self.map_center_m[0] - self.map_size[0]/2.0
        self.map_top = self.map_center_m[1] - self.map_size[1]/2.0

        self.map_size_scale = map_size_scale

    def _convert_lat_lon(self, lat, lon):

        x       = (lon + 180.0) / 360.0 

        latRad  = (lat * math.pi) / 180.0
        mercN   = math.log(math.tan((math.pi / 4.0) + (latRad / 2.0)))
        y       = 0.5 - (mercN / (2.0*math.pi))

        return [x, y]

    def convert(self, x, y):
        mapx = (x - self.map_left) / self.map_size_scale
        mapy = (y - self.map_top) / self.map_size_scale

        return mapx, mapy

    def convert_list(self, xs, ys):
        mapxs = []
        mapys = []

        for i in range(0, len(xs)):
            mapxs.append((xs[i] + self.EQUATOR_HALF - self.map_left) / self.map_size_scale)
            mapys.append((((ys[i] - self.EQUATOR_HALF) * -1) - self.map_top) / self.map_size_scale)

        return mapxs, mapys 

def shapely_poly_to_list(poly):
    x, y = building.exterior.coords.xy

    coords = []
    for i in range(0, len(x)):
        coords.append([x[i], y[i]])

    return coords

MAP_CENTER      = [50.980467, 11.325000]
MAP_SIZE        = [297, 210] # unit: m
MAP_SIZE_SCALE  = 10.0 # increase or decrease MAP_SIZE by factor

conv = Converter(MAP_CENTER, MAP_SIZE, MAP_SIZE_SCALE)
svg = SvgWriter("test.svg", MAP_SIZE)

print("map size: {:.2f} x {:.2f} meter".format(MAP_SIZE[0]*MAP_SIZE_SCALE, MAP_SIZE[1]*MAP_SIZE_SCALE))
print("svg size: {:.2f} x {:.2f} units".format(*MAP_SIZE))

svg.add_layer("buildings")

buildings = []

timer_start = datetime.now()
# curs.execute("SELECT ST_AsEWKB(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
# curs.execute("SELECT ST_AsBinary(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
curs.execute("SELECT geometry FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
# curs.execute("SELECT ST_AsText(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
print("querying data in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))


timer_start = datetime.now()
# results = [curs.fetchone()]
results = curs.fetchall()
for item in results:
    buildings.append(ops.transform(conv.convert_list, loads(item[0], hex=True)))
    # buildings.append(loads(item[0], hex=True))

print("transforming data in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))

for building in buildings:
    # print(building)
    # exit()
    # print(building.area)

    svg.add_polygon(shapely_poly_to_list(building), stroke_width=0.2, opacity=1.0, layer="buildings")

svg.save()

curs.close()
conn.close()