import math
from datetime import datetime

from svgwriter import SvgWriter

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

MAP_CENTER      = [50.980467, 11.325000]
 # unit for data: m / unit for SVG elements: px or mm
# MAP_SIZE        = [297, 210]
MAP_SIZE        = [297, 420]
# increase or decrease MAP_SIZE by factor
MAP_SIZE_SCALE  = 10.0 

PEN_WIDTH = 0.25

conn = psycopg2.connect(database='osm', user='osm')
curs = conn.cursor()

class Converter(object):

    EQUATOR = 40075016.68557849
    EQUATOR_HALF = EQUATOR / 2.0

    def __init__(self, map_center, map_size, map_size_scale):
        self.map_center_lat_lon = map_center

        self.map_center_m = self._convert_lat_lon(*map_center)

        self.map_center_m[0] *= self.EQUATOR
        self.map_center_m[1] *= self.EQUATOR

        self.map_size = [map_size[0] * map_size_scale, map_size[1] * map_size_scale]
        self.map_left = self.map_center_m[0] - self.map_size[0]/2.0
        self.map_top = self.map_center_m[1] - self.map_size[1]/2.0

        self.map_size_scale = map_size_scale

    def get_bounding_box(self): # in web mercator
        return [
            self.map_center_m[0] - self.EQUATOR_HALF - self.map_size[0]/2.0, 
            (self.map_center_m[1]  - self.EQUATOR_HALF) * -1 - self.map_size[1]/2.0,
            self.map_center_m[0] - self.EQUATOR_HALF + self.map_size[0]/2.0, 
            (self.map_center_m[1]  - self.EQUATOR_HALF) * -1  + self.map_size[1]/2.0,
        ]

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

class color:
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

def shapely_polygon_to_list(poly):
    x, y = poly.exterior.coords.xy

    coords = []
    for i in range(0, len(x)):
        coords.append([x[i], y[i]])

    return coords

def shapely_linestring_to_list(linestring):
    x, y = linestring.coords.xy

    coords = []
    for i in range(0, len(x)):
        coords.append([x[i], y[i]])

    return coords

# converts a python list to a postgres-syntax compatible value list in " WHERE foo IN ('a', 'b') "-style
def list_to_pg_str(l):
    ret = "'"
    for i in range(0, len(l)-1):
        ret += l[i]
        ret += "', '"
    ret += l[-1]
    ret += "'"

    return ret

def create_hatching(poly, distance=2.0):
    bounds = poly.bounds

    hatching_lines = []
    num_lines = (bounds[2]-bounds[0]+bounds[3]-bounds[1]) / distance

    for i in range(0, int(num_lines)):
        line = LineString([[bounds[0], bounds[1] + i*distance], [bounds[0] + i*distance, bounds[1]]])
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

    return hatching_lines

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

    if roadtype in ["residential"]: 
        roads_medium.append(roadgeometry)
        continue

    if roadtype in ["tertiary", "secondary", "primary"]: 
        roads_large.append(roadgeometry)
        continue

    roads_small.append(roadgeometry)

num_all_roads = len(roads_large) + len(roads_medium) + len(roads_small)
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
    "stadium"
] # residential
timer_start = datetime.now()
curs.execute("""
    SELECT type, geometry FROM {0}.{1}landusages 
    WHERE type IN ({2})
    AND {0}.{1}landusages.geometry && ST_MakeEnvelope({3}, {4}, {5}, {6}, 3857)
""".format(DB_NAME, DB_PREFIX, list_to_pg_str(filter_list), *conv.get_bounding_box()))
print(TIMER_STRING.format("querying landusage data", (datetime.now()-timer_start).total_seconds()))

timer_start = datetime.now()
results = curs.fetchall()
for item in results:
    landusage = loads(item[1], hex=True)
    landusages.append(landusage)


print("{:<50s}: {}".format("landusages", len(landusages)))
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
    roads_large[i] = roads_large[i].buffer(6)

for i in range(0, len(roads_medium)):
    roads_medium[i] = roads_medium[i].buffer(3)

for i in range(0, len(roads_small)):
    roads_small[i] = roads_small[i].buffer(5)

# cut out landusage

landusages_copy = landusages.copy()
landusages_result = []

for i in range(0, len(roads_small)):
    road = roads_small[i]
    append_list = []
    road_prepared = prep(road)

    for land in landusages_copy:
        if road_prepared.intersects(land):
            zonk = land.difference(road)
            if not zonk.is_empty:
                append_list.append(zonk)
                landusages_copy.remove(land)

    for land in landusages_result:
        if road_prepared.intersects(land):
            zonk = land.difference(road)
            if not zonk.is_empty:
                append_list.append(zonk)
                landusages_result.remove(land)

    for item in append_list:
        if type(item) is MultiPolygon:
            for poly in list(item.geoms):
                landusages_result.append(poly)
        else:
            landusages_result.append(item)

landusages = landusages_result

print(TIMER_STRING.format("performing operations", (datetime.now()-timer_start).total_seconds()))

# --- TRANSFORM

timer_start = datetime.now()
for i in range(0, len(buildings_small)):
    buildings_small[i] = ops.transform(conv.convert_list, buildings_small[i])
for i in range(0, len(buildings_large)):
    buildings_large[i] = ops.transform(conv.convert_list, buildings_large[i])
print(TIMER_STRING.format("transforming building data", (datetime.now()-timer_start).total_seconds()))    

timer_start = datetime.now()
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

# --- ADDING

timer_start = datetime.now()

for building in buildings_small:
    svg.add_polygon(shapely_polygon_to_list(building), stroke_width=PEN_WIDTH, opacity=0, layer="buildings")
for building in buildings_large:
    svg.add_polygon(shapely_polygon_to_list(building), stroke_width=PEN_WIDTH, opacity=0, layer="buildings")
    for line in create_hatching(building):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH)
print("{:<50s}: {}".format("added buildings", len(buildings_small) + len(buildings_large)))

# for road in roads_small:
#     svg.add_polygon(shapely_polygon_to_list(road), stroke_width=0.2, fill=[255, 0, 0], opacity=0.5, layer="roads")
for road in roads_medium:
    svg.add_polygon(shapely_polygon_to_list(road), stroke_width=PEN_WIDTH, opacity=0, layer="roads")
for road in roads_large:
    svg.add_polygon(shapely_polygon_to_list(road), stroke_width=0, opacity=0, layer="roads")
    for line in create_hatching(road, distance=0.5):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="roads")

for landusage in landusages:
    # svg.add_polygon(shapely_linestring_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    # svg.add_polygon(shapely_polygon_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    for line in create_hatching(landusage, distance=1.0):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="landusages")

for waterarea in waterareas:
    # svg.add_polygon(shapely_linestring_to_list(landusage), stroke_width=0.2, opacity=1.0, layer="landusages")
    # svg.add_polygon(shapely_polygon_to_list(waterarea), stroke_width=0.2, opacity=1.0, layer="waterareas")
    for line in create_hatching(waterarea, distance=0.5):
        svg.add_line(shapely_linestring_to_list(line), stroke_width=PEN_WIDTH, layer="waterareas")

print(TIMER_STRING.format("added data to svgwriter", (datetime.now()-timer_start).total_seconds()))

# --- WRITING

svg.save()

curs.close()
conn.close()

print(TIMER_STRING.format(color.BOLD + "time total" + color.END, (datetime.now()-timer_total).total_seconds()))