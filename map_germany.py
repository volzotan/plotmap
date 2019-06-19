import math
from datetime import datetime

from svgwriter import SvgWriter
from maptools import *

import psycopg2
import shapely
from shapely.wkb import loads
from shapely import ops
from shapely.prepared import prep
from shapely.geometry import GeometryCollection, MultiLineString, LineString, Polygon, MultiPolygon

DB_NAME = "import"
DB_PREFIX = "osm_"

TIMER_STRING = "{:<50s}: {:2.2f}s"

# Area Thresholds in meter
BUILDING_AREA_THRESHOLD_FILTER  = 10.0
BUILDING_AREA_THRESHOLD_SMALL   = 1500.0 

MAP_CENTER      = [50.979858, 11.325714]
MAP_SIZE        = [210-10, 297-10]          # unit for data: m / unit for SVG elements: px or mm
MAP_SIZE_SCALE  = 10000.0                      # increase or decrease MAP_SIZE by factor

PEN_WIDTH = 0.25

conn = psycopg2.connect(database='osm', user='osm')
curs = conn.cursor()

timer_total = datetime.now()

conv = Converter(MAP_CENTER, MAP_SIZE, MAP_SIZE_SCALE)
svg = SvgWriter("germany.svg", MAP_SIZE)

print("map size: {:.2f} x {:.2f} meter".format(MAP_SIZE[0]*MAP_SIZE_SCALE, MAP_SIZE[1]*MAP_SIZE_SCALE))
print("svg size: {:.2f} x {:.2f} units".format(*MAP_SIZE))

svg.add_layer("landusages")
svg.add_layer("waterareas")
svg.add_layer("roads")
svg.add_layer("buildings")
svg.add_layer("buildings_hatching")

landusages      = []
waterareas      = []
roads_large     = []
roads_medium    = []
roads_small     = []
roads_railway   = []
buildings_large = []
buildings_small = []

filter_list = ["water", "riverbank", "basin", "reservoir", "swimming_pool"]
timer_start = datetime.now()

params = {
    "db_name"       : DB_NAME,
    "db_prefix"     : DB_PREFIX,
    "types"         : list_to_pg_str(filter_list),
    "env_0"         : conv.get_bounding_box()[0],
    "env_1"         : conv.get_bounding_box()[1],
    "env_2"         : conv.get_bounding_box()[2],
    "env_3"         : conv.get_bounding_box()[3],
    "minimum_area"  : 10000.0
}

curs.execute("""
    SELECT type, geometry, area FROM {db_name}.{db_prefix}waterareas 
    WHERE type IN ({types})
    AND area > {minimum_area}
    AND {db_name}.{db_prefix}waterareas.geometry && ST_MakeEnvelope({env_0}, {env_1}, {env_2}, {env_3}, 3857)
""".format(**params))

print(TIMER_STRING.format("querying waterarea data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
for item in results:
    waterarea = loads(item[1], hex=True)
    waterareas.append(waterarea)

print(TIMER_STRING.format("reading waterarea data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
for i in range(0, len(waterareas)):
    waterareas[i] = ops.transform(conv.convert_list, waterareas[i])
print(TIMER_STRING.format("transforming waterarea data", (datetime.now()-timer_start).total_seconds()))    

waterareas = unpack_multipolygons(waterareas)

for waterarea in waterareas:
    for poly in shapely_polygon_to_list(waterarea):
        svg.add_polygon(poly, stroke_width=0.2, opacity=1.0, layer="waterareas")

svg.save()

curs.close()
conn.close()

print(TIMER_STRING.format(Color.BOLD + "time total" + Color.END, (datetime.now()-timer_total).total_seconds()))
