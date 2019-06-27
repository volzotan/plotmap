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

MAP_CENTER      = [50.979858, 11.325714] # Weimar
MAP_CENTER      = [53.7404084,7.4809997] # Langeoog
MAP_SIZE        = [210-10, 297-10]       # unit for data: m / unit for SVG elements: px or mm
MAP_SIZE_SCALE  = 50.0                   # increase or decrease MAP_SIZE by factor

PEN_WIDTH = 0.25

conn = psycopg2.connect(database='osm', user='osm')
curs = conn.cursor()

timer_total = datetime.now()

conv = Converter(MAP_CENTER, MAP_SIZE, MAP_SIZE_SCALE)
svg = SvgWriter("test.svg", MAP_SIZE)

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

# buildings.append(Polygon([[10, 10], [150, 10], [120, 100], [10, 100], [50, 70]]))

# --- QUERY & READ

# curs.execute("SELECT ST_AsEWKB(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
# curs.execute("SELECT ST_AsBinary(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))
# curs.execute("SELECT ST_AsText(geometry) FROM {}.{}buildings".format(DB_NAME, DB_PREFIX))

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

# ---

timer_start = datetime.now()
curs.execute("""
    SELECT type, geometry FROM {0}.{1}roads 
    WHERE {0}.{1}roads.geometry && ST_MakeEnvelope({2}, {3}, {4}, {5}, 3857)
""".format(DB_NAME, DB_PREFIX, *conv.get_bounding_box()))
print(TIMER_STRING.format("querying road data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
for item in results:
    roadgeometry = loads(item[1], hex=True)
    roadtype = item[0]

    if roadtype in ["railway"]: 
        roads_railway.append(roadgeometry)
        continue

    if roadtype in ["residential", "unclassified", "road", "bridleway"]: 
        roads_medium.append(roadgeometry)
        continue

    if roadtype in ["motorway", "motorway_link", "tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link", "trunk"]: 
        roads_large.append(roadgeometry)
        continue

    roads_small.append(roadgeometry)

num_all_roads = len(roads_large) + len(roads_medium) + len(roads_small) + len(roads_railway)
print("{:<50s}: {:5}/{} | {:3.2f}%".format("railway roads", len(roads_railway), num_all_roads, len(roads_railway)/num_all_roads*100.0))
print("{:<50s}: {:5}/{} | {:3.2f}%".format("small roads", len(roads_small), num_all_roads, len(roads_small)/num_all_roads*100.0))
print("{:<50s}: {:5}/{} | {:3.2f}%".format("medium roads", len(roads_medium), num_all_roads, len(roads_medium)/num_all_roads*100.0))
print("{:<50s}: {:5}/{} | {:3.2f}%".format("large roads", len(roads_large), num_all_roads, len(roads_large)/num_all_roads*100.0))
print(TIMER_STRING.format("reading road data", (datetime.now()-timer_start).total_seconds()))

# ---

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

# ---

filter_list = ["water", "riverbank", "basin", "reservoir", "swimming_pool"]
timer_start = datetime.now()
curs.execute("""
    SELECT type, geometry FROM {0}.{1}waterareas 
    WHERE type IN ({2})
    AND {0}.{1}waterareas.geometry && ST_MakeEnvelope({3}, {4}, {5}, {6}, 3857)
""".format(DB_NAME, DB_PREFIX, list_to_pg_str(filter_list), *conv.get_bounding_box()))
print(TIMER_STRING.format("querying waterarea data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
for item in results:
    waterarea = loads(item[1], hex=True)
    waterareas.append(waterarea)

print(TIMER_STRING.format("reading waterarea data", (datetime.now()-timer_start).total_seconds()))

# --- OPERATIONS

timer_start = datetime.now()

# generate road polygons from lines

for i in range(0, len(roads_large)):
    roads_large[i] = roads_large[i].buffer(8, 
        cap_style=shapely.geometry.CAP_STYLE.round, 
        join_style=shapely.geometry.JOIN_STYLE.round)

for i in range(0, len(roads_medium)):
    roads_medium[i] = roads_medium[i].buffer(4, 
        cap_style=shapely.geometry.CAP_STYLE.round, 
        join_style=shapely.geometry.JOIN_STYLE.bevel)

for i in range(0, len(roads_small)):
    roads_small[i] = roads_small[i].buffer(5)

# cut out landusage

landusages = subtract_layer_from_layer(landusages, roads_small)#, grow=1)
# landusages = subtract_layer_from_layer(landusages, buildings_small, grow=1)
# landusages = subtract_layer_from_layer(landusages, buildings_large, grow=2) # TODO: bug? nearly no landusage polygons left...

number_removed = remove_duplicates(landusages)
print("{:<50s}: {}/{}".format("removed duplicates after cutting", number_removed, number_removed+len(landusages)))

print(TIMER_STRING.format("performing operations", (datetime.now()-timer_start).total_seconds()))

# --- TRANSFORM

timer_start = datetime.now()
for i in range(0, len(buildings_small)):
    buildings_small[i] = ops.transform(conv.convert_list, buildings_small[i])
for i in range(0, len(buildings_large)):
    buildings_large[i] = ops.transform(conv.convert_list, buildings_large[i])
print(TIMER_STRING.format("transforming building data", (datetime.now()-timer_start).total_seconds()))    

timer_start = datetime.now()
for i in range(0, len(roads_railway)):
    roads_railway[i] = ops.transform(conv.convert_list, roads_railway[i])
for i in range(0, len(roads_small)):
    roads_small[i] = ops.transform(conv.convert_list, roads_small[i])
for i in range(0, len(roads_medium)):
    roads_medium[i] = ops.transform(conv.convert_list, roads_medium[i])
for i in range(0, len(roads_large)):
    roads_large[i] = ops.transform(conv.convert_list, roads_large[i])
print(TIMER_STRING.format("transforming road data", (datetime.now()-timer_start).total_seconds()))    

timer_start = datetime.now()
for i in range(0, len(landusages)):
    landusages[i] = ops.transform(conv.convert_list, landusages[i])
print(TIMER_STRING.format("transforming landusage data", (datetime.now()-timer_start).total_seconds()))    

timer_start = datetime.now()
for i in range(0, len(waterareas)):
    waterareas[i] = ops.transform(conv.convert_list, waterareas[i])
print(TIMER_STRING.format("transforming waterarea data", (datetime.now()-timer_start).total_seconds()))    

# --- CLIPPING

timer_start = datetime.now()

bbox = [0, 0, MAP_SIZE[0], MAP_SIZE[1]]
buildings_small = clip_layer_by_box(buildings_small, bbox)
buildings_large = clip_layer_by_box(buildings_large, bbox)
roads_railway = clip_layer_by_box(roads_railway, bbox)
roads_medium = clip_layer_by_box(roads_medium, bbox)
roads_large = clip_layer_by_box(roads_large, bbox)
landusages = clip_layer_by_box(landusages, bbox)
waterareas = clip_layer_by_box(waterareas, bbox)

print(TIMER_STRING.format("clipping objects", (datetime.now()-timer_start).total_seconds()))   

# --- ADDING

timer_start = datetime.now()

for building in buildings_small:
    for poly in shapely_polygon_to_list(building):
        svg.add_polygon(poly, stroke_width=PEN_WIDTH, opacity=0, layer="buildings")

for building in buildings_large:

    polys = shapely_polygon_to_list(building)

    for poly in polys: 
        svg.add_polygon(poly, stroke_width=PEN_WIDTH, opacity=0, layer="buildings")
    
    # if multiple polygons (i.e. poly with holes) only first (outer) polygon should be hatched
    for line in Hatching.create_hatching(building): #, hatching_type=Hatching.VERTICAL):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH)

print("{:<50s}: {}".format("added buildings", len(buildings_small) + len(buildings_large)))

for road in roads_railway:
    for poly in shapely_polygon_to_list(road):
        svg.add_polygon(poly, stroke_width=PEN_WIDTH, opacity=0.5, layer="roads")
# for road in roads_small:
#     svg.add_polygon(shapely_polygon_to_list(road), stroke_width=PEN_WIDTH, opacity=0, layer="roads")
for road in roads_medium:
    # svg.add_polygon(shapely_polygon_to_list(road), stroke_width=PEN_WIDTH, opacity=0, layer="roads")
    for line in Hatching.create_hatching(road, distance=0.5, connect=True):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="roads")
for road in roads_large:
    # svg.add_polygon(shapely_polygon_to_list(road), stroke_width=0, opacity=0, layer="roads")
    for line in Hatching.create_hatching(road, distance=0.5, connect=True):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="roads")

for landusage in landusages:
    # svg.add_polygon(shapely_linestring_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    # svg.add_polygon(shapely_polygon_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    for line in Hatching.create_hatching(landusage, distance=1.0):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="landusages")

for waterarea in waterareas:
    # svg.add_polygon(shapely_linestring_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    # svg.add_polygon(shapely_polygon_to_list(waterarea), stroke_width=0.2, opacity=1.0, layer="waterareas")

    hatch1 = Hatching.create_hatching(waterarea, distance=1.0, connect=True, hatching_type=Hatching.LEFT_TO_RIGHT)
    hatch2 = Hatching.create_hatching(waterarea, distance=1.0, connect=True, hatching_type=Hatching.RIGHT_TO_LEFT)

    for line in hatch1 + hatch2:
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="waterareas")

print(TIMER_STRING.format("added data to svgwriter", (datetime.now()-timer_start).total_seconds()))

# --- WRITING

svg.save()

curs.close()
conn.close()

print(TIMER_STRING.format(Color.BOLD + "time total" + Color.END, (datetime.now()-timer_total).total_seconds()))