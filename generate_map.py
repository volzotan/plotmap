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

MAP_CENTER      = [50.979858, 11.325714]
MAP_SIZE        = [210-10, 297-10]          # unit for data: m / unit for SVG elements: px or mm
MAP_SIZE_SCALE  = 8.0                       # increase or decrease MAP_SIZE by factor

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

    x, y = poly.exterior.coords.xy

    coords_outer = []
    for i in range(0, len(x)):
        coords_outer.append([x[i], y[i]])

    result_list.append(coords_outer)

    holes = list(poly.interiors)
    for ring in holes:
        xr, yr = ring.coords.xy
        ring_coords = []
        if xr is not None and yr is not None:
            for i in range(0, len(xr)):
                ring_coords.append([xr[i], yr[i]])
        result_list.append(ring_coords)

    return result_list

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