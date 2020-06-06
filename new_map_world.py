from datetime import datetime
import os

from svgwriter import SvgWriter
import maptools

import psycopg2
import shapely
from shapely.wkb import dumps, loads
from shapely import ops
from shapely.prepared import prep
from shapely.geometry import GeometryCollection, MultiLineString, LineString, Polygon, MultiPolygon
from shapely.geometry import shape

import fiona

TIMER_STRING                    = "{:<60s}: {:2.2f}s"

DB_NAME                         = "import"
DB_PREFIX                       = "osm_"

MAP_CENTER                      = [0, 0]
MAP_SIZE                        = [500, 500] # [210-10, 297-10]   # unit for data: m / unit for SVG elements: px or mm

MAP_SIZE_SCALE                  = maptools.EQUATOR/MAP_SIZE[0]      # increase or decrease MAP_SIZE by factor

SIMPLIFICATION_MAX_ERROR        = 0.1                               # unit in map coordinates (px or mm)

THRESHOLD_CITY_POPULATION       = 500000

CONNECT_DATABASE                = False

DRAW_COASTLINE                  = True
DRAW_PLACES                     = False
DRAW_BORDERS                    = False
DRAW_BATHYMETRY                 = False

""" ------------------------------

Projection:

Map is using OSMs Web-Mercator (3857) projection
Natural Earth Shapefiles are encoded in WGS84 (4326)

------------------------------ """

timer_total = datetime.now()

if CONNECT_DATABASE:
    conn = psycopg2.connect(database='osm', user='osm')
    curs = conn.cursor()

conv = maptools.Converter(MAP_CENTER, MAP_SIZE, MAP_SIZE_SCALE)
svg = SvgWriter("world.svg", MAP_SIZE, background_color="gray")

print("map size: {:.2f} x {:.2f} meter".format(MAP_SIZE[0]*MAP_SIZE_SCALE, MAP_SIZE[1]*MAP_SIZE_SCALE))
print("svg size: {:.2f} x {:.2f} units".format(*MAP_SIZE))

svg.add_layer("coastlines")
svg.add_layer("places")
svg.add_layer("borders")
svg.add_layer("bathymetry")

svg.add_hatching("bathymetry_hatching_6",   distance=16)
svg.add_hatching("bathymetry_hatching_5",   distance=8, stroke_dasharray="4 4")
svg.add_hatching("bathymetry_hatching_4",   distance=4)
svg.add_hatching("bathymetry_hatching_3",   distance=2, stroke_dasharray="4 4")
svg.add_hatching("bathymetry_hatching_2",   distance=1)
svg.add_hatching("bathymetry_hatching_1",   distance=0.5, stroke_dasharray="4 4")

coastlines = []
places = []
borders = []
bathymetry = []

# ---------------------------------------- COASTLINES / LAND POLYS ----------------------------------------

# ---------------------------------------- COASTLINES / WATER POLYS ----------------------------------------

def save_pickle_geom(filename, geom):

    if os.path.exists(filename):
        raise Exception("file does already exist: {}".format(filename))

    f = open(filename, "wb")
    f.write(dumps(geom))
    f.close()

def load_pickle_geom(filename):

    if not os.path.exists(filename):
        raise Exception("file does not exist: {}".format(filename))

    f = open(filename, "r")
    wkt_data = f.read()
    f.close()

    return loads(wkt_data)

def load_coastline():

    coastlines = []

    COASTLINE_FILE = "world_data/simplified-water-polygons-split-3857/simplified_water_polygons.shp"

    timer_start = datetime.now()
    shapefile = fiona.open(COASTLINE_FILE)

    for item in shapefile:
        shp_geom = shape(item['geometry'])

        coastlines.append(shp_geom)

    coastlines = coastlines[0:3000]

    print(TIMER_STRING.format("loading coastline data", (datetime.now()-timer_start).total_seconds()))   

    timer_start = datetime.now()
    for i in range(0, len(coastlines)):
        coastlines[i] = ops.transform(conv.convert_mercator_to_map_list, coastlines[i])
    print(TIMER_STRING.format("transforming coastline data", (datetime.now()-timer_start).total_seconds()))   

    timer_start = datetime.now()
    coastlines_simplified = []
    errors_occured = 0
    skipped = 0
    for i in range(0, len(coastlines)):

        poly = coastlines[i]

        # if len(poly.exterior.coords) <= 8: # empty coastline tile
        #     skipped += 1
        #     continue

        # simplified_polygon = simplify_polygon(coastlines[i], epsilon=SIMPLIFICATION_MAX_ERROR)
        simplified_polygon = poly.simplify(SIMPLIFICATION_MAX_ERROR)
        if not type(simplified_polygon) is Polygon:
            errors_occured += 1
            continue
        coastlines_simplified.append(simplified_polygon)
    coastlines = coastlines_simplified

    print(TIMER_STRING.format("simplifing coastline data ({} errors, {} skipped)".format(errors_occured, skipped), (datetime.now()-timer_start).total_seconds())) 

    return coastlines

if DRAW_COASTLINE:

    PICKLE_FILE = "coastlines.pickle"
    complete_poly = None

    if os.path.exists(PICKLE_FILE):
        master_poly = load_pickle_geom(PICKLE_FILE)
        print("loaded coastlines from pickle: {}".format(PICKLE_FILE))
    else:
        coastlines = load_coastline()

        # merge water polygon to single block
        complete_poly = ops.unary_union(coastlines)
        save_pickle_geom("coastlines.pickle", complete_poly)

    timer_start = datetime.now()

    coastlines = [] 
    if type(master_poly) is MultiPolygon:
        for g in master_poly.geoms:
            coastlines.append(g)
    else:
        coastlines.append(master_poly)

    print(TIMER_STRING.format("merging coastline data", (datetime.now()-timer_start).total_seconds()))   


# --------------------------------------------------------------------------------

if DRAW_PLACES:

    filter_list = ["city", "town"]
    timer_start = datetime.now()

    params = {
        "db_name"       : DB_NAME,
        "db_prefix"     : DB_PREFIX,
        "db_table"      : "places",
        "types"         : maptools.list_to_pg_str(filter_list),
        "env_0"         : conv.get_bounding_box()[0],
        "env_1"         : conv.get_bounding_box()[1],
        "env_2"         : conv.get_bounding_box()[2],
        "env_3"         : conv.get_bounding_box()[3],
        "minimum_population" : THRESHOLD_CITY_POPULATION
    }

    curs.execute("""
        SELECT name, type, geometry, population FROM {db_name}.{db_prefix}{db_table} 
        WHERE type IN ({types})
        AND population > {minimum_population}
    """.format(**params))

    print(TIMER_STRING.format("querying places data", (datetime.now()-timer_start).total_seconds()))

    timer_start = datetime.now()
    results = curs.fetchall()
    for item in results:
        place = loads(item[2], hex=True)
        places.append(place)

    print(TIMER_STRING.format("reading places data", (datetime.now()-timer_start).total_seconds()))

    timer_start = datetime.now()
    for i in range(0, len(places)):
        places[i] = ops.transform(conv.convert_mercator_to_map_list, places[i])
    print(TIMER_STRING.format("transforming places data", (datetime.now()-timer_start).total_seconds()))  

    print("found places: {}".format(len(places)))

# --------------------------------------------------------------------------------

if DRAW_BORDERS:

    BORDER_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_admin_0_boundary_lines_land.shp"

    timer_start = datetime.now()
    shapefile = fiona.open(BORDER_FILE)

    for item in shapefile:
        shp_geom = shape(item['geometry'])
        borders.append(shp_geom)

    print(TIMER_STRING.format("loading border data", (datetime.now()-timer_start).total_seconds()))  

    timer_start = datetime.now()
    for i in range(0, len(borders)):
        borders[i] = ops.transform(conv.convert_wgs_to_map_list_lon_lat, borders[i])
    print(TIMER_STRING.format("transforming border data", (datetime.now()-timer_start).total_seconds()))  


# --------------------------------------------------------------------------------

def parse_shapefile(filename, min_area=None, latlons_flipped=False):

    geometries = []

    shapefile = fiona.open(filename)

    if latlons_flipped:
        func = conv.convert_wgs_to_map_list_lon_lat
    else:
        func = conv.convert_wgs_to_map_list

    for item in shapefile:
        shapefile_geom = shape(item['geometry'])
        geom = ops.transform(func, shapefile_geom)
        geom = geom.simplify(SIMPLIFICATION_MAX_ERROR)

        if type(geom) is Polygon:

            if not geom.is_valid:
                geom = geom.buffer(0.01)

            if min_area is not None and geom.area < min_area:
                pass
            else:
                geometries.append(geom)
        elif type(geom) is MultiPolygon:
            for g in geom.geoms:

                if not geom.is_valid:
                    geom = geom.buffer(0.01)

                if min_area is not None and geom.area < min_area:
                    pass
                else:
                    geometries.append(g)
        else:
            raise Exception("bathymetry: unexpected type: {}".format(b))

    return geometries

def add_to_writer(polys, hatching_name):

    options = {
        "stroke_width": 0,
        "opacity": 0.0,
        "layer": "bathymetry"
    }

    for i in range(0, len(polys)):
        p = polys[i]
        if not p.is_valid:
            print("error: polygon {}/{} not valid".format(i, len(polys)))
            continue
        svg.add_polygon(p, **options, hatching=hatching_name)

if DRAW_BATHYMETRY:

    timer_start = datetime.now()

    bathymetry = []

    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_A_10000.shp", min_area=1.0, latlons_flipped=True), "bathymetry_hatching_1")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_B_9000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_1")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_C_8000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_2")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_D_7000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_2")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_E_6000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_2")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_F_5000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_3")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_G_4000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_3")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_H_3000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_4")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_I_2000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_4")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_J_1000.shp",  min_area=1.0, latlons_flipped=True), "bathymetry_hatching_5")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_K_200.shp",   min_area=1.0, latlons_flipped=True), "bathymetry_hatching_5")
    add_to_writer(parse_shapefile("world_data/10m_physical/ne_10m_bathymetry_L_0.shp",     min_area=1.0, latlons_flipped=True), "bathymetry_hatching_6")

    print(TIMER_STRING.format("parsing sea data", (datetime.now()-timer_start).total_seconds()))  

# --------------------------------------------------------------------------------

timer_start = datetime.now()

for coastline in coastlines:
    svg.add_polygon(coastline, stroke_width=0.2, opacity=0.0, stroke=[255, 255, 255], layer="coastlines") #, hatching="coastlines_hatching")
    # for poly in maptools.shapely_polygon_to_list(coastline):
    #     svg.add_polygon(poly, stroke_width=0.2, opacity=0.0, layer="coastlines")

for place in places:
    svg.add_circles(list(place.coords), radius=0.5, layer="places")

for border in borders:
    lines = []

    if type(border) is MultiLineString:
        lines += list(border.geoms)
    else:
        lines += [border]

    for lineString in lines:
        svg.add_polygon(list(lineString.coords), stroke_width=0.2, opacity=0.0, layer="borders")


print(TIMER_STRING.format("preparing SVG writer", (datetime.now()-timer_start).total_seconds())) 

svg.save()

print(TIMER_STRING.format(maptools.Color.BOLD + "time total" + maptools.Color.END, (datetime.now()-timer_total).total_seconds()))