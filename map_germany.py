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

SIMPLIFICATION_MAX_ERROR = 0.5

MAP_CENTER      = [50.979858, 11.325714]
MAP_SIZE        = [210-10, 148-10]          # unit for data: m / unit for SVG elements: px or mm
MAP_SIZE_SCALE  = 5.0                      # increase or decrease MAP_SIZE by factor

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

svg.add_hatching("building_hatching", stroke_width=PEN_WIDTH, distance=2.0)
svg.add_hatching("landusages_hatching", stroke_width=PEN_WIDTH, distance=4.0)

# --- WATERAREAS ---

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
    waterareas[i] = ops.transform(conv.convert_mercator_to_map_list, waterareas[i])
print(TIMER_STRING.format("transforming waterarea data", (datetime.now()-timer_start).total_seconds()))    

waterareas = unpack_multipolygons(waterareas)

for waterarea in waterareas:
    # for poly in shapely_polygon_to_list(waterarea):
    #     svg.add_polygon(poly, stroke_width=0.2, opacity=1.0, layer="waterareas")

    w = waterarea.simplify(SIMPLIFICATION_MAX_ERROR)
    svg.add_polygon(waterarea, stroke_width=0.2, opacity=1.0, layer="waterareas")

# --- BUILDINGS ---

timer_start = datetime.now()
curs.execute("""
    SELECT geometry FROM {0}.{1}buildings 
    WHERE {0}.{1}buildings.geometry && ST_MakeEnvelope({2}, {3}, {4}, {5}, 3857)
""".format(DB_NAME, DB_PREFIX, *conv.get_bounding_box()))
print(TIMER_STRING.format("querying building data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
filtered = 0
for item in results:
    building = loads(item[0], hex=True)
    
    if building.area < BUILDING_AREA_THRESHOLD_FILTER:
        filtered += 1
        continue

    if building.area < BUILDING_AREA_THRESHOLD_SMALL:
        buildings_small.append(building)
        continue

    buildings_large.append(building) 
    # buildings.append(loads(item[0], hex=True))

print("{:<50s}: {}/{}".format("filtered buildings", filtered, len(buildings_small)+len(buildings_large)+filtered))
print("{:<50s}: {}/{}".format("large buildings", len(buildings_large), len(buildings_small)+len(buildings_large)))
print(TIMER_STRING.format("reading building data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
for i in range(0, len(buildings_small)):
    buildings_small[i] = ops.transform(conv.convert_mercator_to_map_list, buildings_small[i])
for i in range(0, len(buildings_large)):
    buildings_large[i] = ops.transform(conv.convert_mercator_to_map_list, buildings_large[i])
print(TIMER_STRING.format("transforming building data", (datetime.now()-timer_start).total_seconds()))  

timer_start = datetime.now()

for building in buildings_small:
    svg.add_polygon(building, stroke_width=PEN_WIDTH, opacity=0, layer="buildings")

for building in buildings_large:
    svg.add_polygon(building, stroke_width=PEN_WIDTH, opacity=0, hatching="building_hatching", layer="buildings")
    
print("{:<50s}: {}".format("added buildings", len(buildings_small) + len(buildings_large)))

# --- LANDUSAGES ---

filter_list = [
    "park", 
    "forest", 
    "meadow", 
    "grass", 
    "scrub", 
    "village_green", 
    "nature_reserve",
    "orchard", 
    "cemetery",
    "farmland",
    "allotments",
    "recreation_ground",
    "sports_centre",
    "wetland",
    "stadium",
    "common",
    # "parking",
    "footway",
]
timer_start = datetime.now()
curs.execute("""
    SELECT type, geometry FROM {0}.{1}landusages 
    WHERE type IN ({2})
    AND {0}.{1}landusages.geometry && ST_MakeEnvelope({3}, {4}, {5}, {6}, 3857)
    ORDER BY area DESC
""".format(DB_NAME, DB_PREFIX, list_to_pg_str(filter_list), *conv.get_bounding_box()))
print(TIMER_STRING.format("querying landusage data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
for item in results:
    landusage = loads(item[1], hex=True)
    landusages.append(landusage)

print("{:<50s}: {}".format("landusages", len(landusages)))

# remove contain-duplicates from landusage
number_removed = remove_duplicates(landusages)

print("{:<50s}: {}/{}".format("filtered landusage duplicates", number_removed, number_removed+len(landusages)))
print(TIMER_STRING.format("reading landusage data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
for i in range(0, len(landusages)):
    landusages[i] = ops.transform(conv.convert_mercator_to_map_list, landusages[i])
print(TIMER_STRING.format("transforming landusage data", (datetime.now()-timer_start).total_seconds()))    

for landusage in landusages:
    svg.add_polygon(landusage, stroke_width=0, opacity=0, hatching="landusages_hatching", layer="landusages")



svg.save()

curs.close()
conn.close()

print(TIMER_STRING.format(Color.BOLD + "time total" + Color.END, (datetime.now()-timer_total).total_seconds()))
