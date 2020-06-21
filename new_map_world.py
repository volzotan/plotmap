from datetime import datetime
import os
import pickle

from svgwriter import SvgWriter
import maptools

import pickle

import psycopg2
import shapely
from shapely.wkb import dumps, loads
from shapely import ops
from shapely.prepared import prep
from shapely.geometry import Point, GeometryCollection, MultiLineString, LineString, Polygon, MultiPolygon
from shapely.geometry import shape

import fiona

TIMER_STRING                    = "{:<60s}: {:2.2f}s"

DB_NAME                         = "import"
DB_PREFIX                       = "osm_"

CACHE_DIRECTORY                 = "cache"

MAP_CENTER                      = [0, 0]
MAP_SIZE                        = [2000, 2000] #[2000, 2000] # [210-10, 297-10]   # unit for data: m / unit for SVG elements: px or mm

MAP_SIZE_SCALE                  = maptools.EQUATOR/MAP_SIZE[0]      # increase or decrease MAP_SIZE by factor

SIMPLIFICATION_MAX_ERROR        = 1.0 # 0.2                              # unit in map coordinates (px or mm)

THRESHOLD_CITY_POPULATION       = 1000000

CONNECT_DATABASE                = False

DRAW_COASTLINE                  = True
DRAW_PLACES                     = False
DRAW_CITIES                     = True
DRAW_URBAN_AREAS                = False
DRAW_BORDERS                    = False
DRAW_ADMIN_REGIONS              = False
DRAW_BATHYMETRY                 = False
DRAW_TERRAIN                    = True

# --------------------------------

CITY_CIRCLE_RADIUS              = 3

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

svg.add_layer("urban")
svg.add_layer("borders")
svg.add_layer("bathymetry")
svg.add_layer("terrain")
svg.add_layer("coastlines")
svg.add_layer("places")
svg.add_layer("meta")

for i in range(0, 15):
    svg.add_hatching("bathymetry_hatching_{}".format(i), stroke_width=0.5, distance=1+0.5*i) #1.1**i) #1+0.5*i)

coastlines = []
places = []
cities = []
urban = []
borders = []
admin = []
bathymetry = []
terrain = []

def simplify_polygon(polys, min_area=None):

    polys_simplyfied = []
    errors_occured = 0
    skipped = 0

    for i in range(0, len(polys)):
        poly = polys[i]

        # simplified_polygon = simplify_polygon(coastlines[i], epsilon=SIMPLIFICATION_MAX_ERROR)
        poly = poly.simplify(SIMPLIFICATION_MAX_ERROR)

        if not type(poly) is Polygon:
            errors_occured += 1
            continue

        if min_area is not None and poly.area < min_area:
            skipped += 1
            continue

        polys_simplyfied.append(poly)

    return polys_simplyfied, errors_occured, skipped

def simplify_linestring(lines):

    lines_simplyfied = []
    errors_occured = 0
    skipped = 0

    for i in range(0, len(lines)):
        line = lines[i]

        line = line.simplify(SIMPLIFICATION_MAX_ERROR)

        if not type(line) is LineString:
            errors_occured += 1
            continue

        lines_simplyfied.append(line)

    return lines_simplyfied, errors_occured, skipped

def get_poly_layer_from_geojson(filename, min_area=None, latlons_flipped=False):

    layers = {}

    shapefile = fiona.open(filename)

    if latlons_flipped:
        func = conv.convert_wgs_to_map_list_lon_lat
    else:
        func = conv.convert_wgs_to_map_list

    for item in shapefile:
        shapefile_geom = shape(item["geometry"])        
        geom = ops.transform(func, shapefile_geom)
        geom = geom.simplify(SIMPLIFICATION_MAX_ERROR)

        prop = item["properties"]
        layer_number = prop["layer"]
        if layer_number not in layers:
            layers[layer_number] = []

        if type(geom) is Polygon:

            if min_area is not None and geom.area < min_area:
                pass
            else:
                layers[layer_number].append(geom)

        elif type(geom) is MultiPolygon:
            for g in geom.geoms:

                if not geom.is_valid:
                    geom = geom.buffer(0.01)

                if min_area is not None and geom.area < min_area:
                    pass
                else:
                    layers[layer_number].append(g)
        else:
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))


    # [
    #   [poly, poly, ...],
    #   [poly, poly, ...],
    # ] 

    # problem: if the lowest layers are not populated, they
    # should not be discard altogether but be empty lists

    flat_list = []
    for i in range(0, max(layers.keys())+1):

        if i not in layers:
            flat_list.append([])
        else:
            flat_list.append(layers[i])

    return flat_list

    # [poly, poly, poly, ...]

    # flat_list = []
    # for sublist in list(layers.values()):
    #     for item in sublist:
    #         flat_list.append(item)

    # return flat_list

def get_polys_from_shapefile(filename, min_area=None, latlons_flipped=False):

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
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))

    return geometries

def get_lines_from_shapefile(filename, latlons_flipped=False):

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

        if type(geom) is LineString:
            geometries.append(geom)
        elif type(geom) is MultiLineString:
            for g in geom.geoms:
                geometries.append(g)
        else:
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))

    return geometries

def write_geometries_to_file(filename, geoms):

    with open(filename, "wb") as handle:
        pickle.dump(geoms, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_geometries_from_file(filename):

    with open(filename, "rb") as handle:
        return pickle.load(handle)

def polygons_to_linestrings(polygons):

    linestrings = []

    flat_list = []
    if len(polygons) > 0 and type(polygons[0]) is list: # polygons is a list of layers
        for layer in polygons:
            flat_list += layer
    else: # polygons is a list of polygons
        flat_list = polygons

    for poly in flat_list:

        if not poly.is_valid:
            poly = poly.buffer(0.01)

        outline = poly.boundary

        if type(outline) is MultiLineString:
            for g in outline.geoms:
                linestrings.append(g)
        else:
            linestrings.append(outline)

    return linestrings

def cut_linestrings_inplace(linestrings, centers):

    circles = []
    for center in centers:
        circles.append(center.buffer(CITY_CIRCLE_RADIUS + 3).simplify(SIMPLIFICATION_MAX_ERROR))

    for i in range(0, len(linestrings)):

        if i%10 == 0:
            print("cut {:10}/{:10} ({:5.2f})".format(i, len(linestrings), (i/len(linestrings))*100), end="\r")
        
        poly = linestrings[i]
        poly = poly.simplify(SIMPLIFICATION_MAX_ERROR)
        poly_bounds = poly.bounds
        
        for circle in circles:    

            if maptools.check_polygons_for_overlap(poly_bounds, circle.bounds):
                poly = poly.difference(circle)
        
        linestrings[i] = poly

    print("") # newline to counter \r

    return linestrings

# ---------------------------------------- COASTLINES / LAND POLYS ----------------------------------------

def load_coastline():

    coastlines = []

    # COASTLINE_FILE = "world_data/simplified-water-polygons-split-3857/simplified_water_polygons.shp"
    COASTLINE_FILE = "world_data/simplified-land-polygons-complete-3857/simplified_land_polygons.shp"

    timer_start = datetime.now()
    shapefile = fiona.open(COASTLINE_FILE)

    for item in shapefile:
        shp_geom = shape(item["geometry"])

        coastlines.append(shp_geom)

    # coastlines = coastlines[0:3000]

    print(TIMER_STRING.format("loading coastline data", (datetime.now()-timer_start).total_seconds()))   

    timer_start = datetime.now()
    for i in range(0, len(coastlines)):
        coastlines[i] = ops.transform(conv.convert_mercator_to_map_list, coastlines[i])
    print(TIMER_STRING.format("transforming coastline data", (datetime.now()-timer_start).total_seconds()))   

    timer_start = datetime.now()
    coastlines, errors_occured, skipped = simplify_polygon(coastlines, min_area=5.0)

    print(TIMER_STRING.format("simplifing coastline data ({} errors, {} skipped)".format(errors_occured, skipped), (datetime.now()-timer_start).total_seconds())) 

    return coastlines

if DRAW_COASTLINE:

    coastlines = load_coastline()

    timer_start = datetime.now()

    for i in range(0, len(coastlines)):
        c = coastlines[i]

        if c.area < 10:
            continue

        # c = c.buffer(0.3).buffer(-0.3).simplify(SIMPLIFICATION_MAX_ERROR)

        if c.is_valid:
            coastlines[i] = c
        else:
            print("error during postprocessing. coastline {}/{}".format(i, len(coastlines)))

    print(TIMER_STRING.format("postprocessing coastline data", (datetime.now()-timer_start).total_seconds())) 

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

if DRAW_CITIES:

    CITIES_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_populated_places.shp"
    geometries = []
    shapefile = fiona.open(CITIES_FILE)

    latlons_flipped = True

    if latlons_flipped:
        func = conv.convert_wgs_to_map_list_lon_lat
    else:
        func = conv.convert_wgs_to_map_list

    for item in shapefile:
        shapefile_geom = shape(item["geometry"])
        geom = ops.transform(func, shapefile_geom)
        geom = geom.simplify(SIMPLIFICATION_MAX_ERROR)

        if not type(geom) is Point:
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))

        # for key in item["properties"]:
        #     print("{} : {}".format(key, item["properties"][key]))
        # exit()

        if int(item["properties"]["POP_MAX"]) >= THRESHOLD_CITY_POPULATION or int(item["properties"]["POP_MIN"]) >= THRESHOLD_CITY_POPULATION:  
            cities.append(geom)


# --------------------------------------------------------------------------------

if DRAW_URBAN_AREAS:

    URBAN_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_urban_areas.shp"

    timer_start = datetime.now()
    
    urban = get_polys_from_shapefile(URBAN_FILE, latlons_flipped=True)

    print(TIMER_STRING.format("loading urban areas data", (datetime.now()-timer_start).total_seconds())) 

    timer_start = datetime.now()

    areas_postprocessed = []
    for i in range(0, len(urban)):
        a = urban[i]
        a = a.buffer(0.1)

        # if a.area < 2:
        #     continue

        areas_postprocessed.append(a)

    urban = []
    unified_poly = ops.unary_union(areas_postprocessed)
    for poly in unified_poly.geoms:
        if poly.area > 8:
            urban.append(poly.buffer(1.0).buffer(1.0))

    print(TIMER_STRING.format("postprocessing urban areas data", (datetime.now()-timer_start).total_seconds())) 
# --------------------------------------------------------------------------------

if DRAW_BORDERS:

    BORDER_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_admin_0_boundary_lines_land.shp"

    timer_start = datetime.now()
    
    borders = get_lines_from_shapefile(BORDER_FILE, latlons_flipped=True)

    print(TIMER_STRING.format("loading border region data", (datetime.now()-timer_start).total_seconds())) 


# --------------------------------------------------------------------------------

if DRAW_ADMIN_REGIONS:

    ADMIN_REGIONS_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_admin_1_states_provinces_lines.shp"

    timer_start = datetime.now()
    
    admin = get_lines_from_shapefile(ADMIN_REGIONS_FILE, latlons_flipped=True)

    print(TIMER_STRING.format("loading admin region data", (datetime.now()-timer_start).total_seconds())) 


# --------------------------------------------------------------------------------

def add_layers_to_writer(poly_layers):

    options = {
        "stroke_width": 0,
        "opacity": 0,
        "layer": "bathymetry",
        "stroke": [11, 72, 107] # [0, 198, 189]
    }

    for layer_i in range(0, len(poly_layers)):

        polys = poly_layers[layer_i]
        hatching_name = "bathymetry_hatching_{}".format(layer_i)

        # if layer_i > 8:
        #     options["stroke"] = [11, 72, 107]

        for i in range(0, len(polys)):
            p = polys[i]

            save_polys = []
            if type(p) is MultiPolygon:
                for g in p.geoms:
                    save_polys.append(g)
            elif type(p) is GeometryCollection:
                for g in p.geoms:
                    if type(g) is Polygon:
                        save_polys.append(g)
                    else:
                        print("encountered unknown geom: {}".format(type(g)))
            else:
                save_polys.append(p)

            for save_poly in save_polys:
                if not save_poly.is_valid:
                    save_poly = save_poly.buffer(0.01)
                    if not save_poly.is_valid:
                        print("error: polygon {}/{} not valid".format(i, len(polys)))
                        continue
                svg.add_polygon(save_poly, **options, hatching=hatching_name)

def cut_bathymetry_inplace(layers, tool_polys):

    current_poly = None
    tool_poly = None

    for i in range(0, len(layers)):
        layer = layers[i]
        for j in range(0, len(layer)):
            current_poly = layer[j]

            if current_poly.area < 1.0:
                    continue

            for k in range(0, len(tool_polys)):
                tool_poly = tool_polys[k]

                if not current_poly.is_valid:
                    current_poly = current_poly.buffer(0.1)

                if not tool_poly.is_valid:
                    continue

                # if maptools.check_polygons_for_overlap(current_poly, tool_poly):
                current_poly = current_poly.difference(tool_poly)

            layer[j] = current_poly.simplify(SIMPLIFICATION_MAX_ERROR)

    return layers

# def cut_bathymetry(layers, tool_polys):

#     for i in range(0, len(layers)):
#         layer = layers[i]

#         mpoly = MultiPolygon(layer)
  
#         for tool_poly in tool_polys:

#             if not mpoly.is_valid:
#                 mpoly = mpoly.buffer(0.1)

#             mpoly = mpoly.difference(tool_poly)

#         layers[i] = mpoly.simplify(SIMPLIFICATION_MAX_ERROR)

#     return layers

def load_bathymetry_file(filename, difference=None):

    cache_file = os.path.join(CACHE_DIRECTORY, filename)

    data = None

    if os.path.exists(cache_file):
        data = load_geometries_from_file(cache_file)
        print("loaded from cache: {} [{} layers]".format(filename, len(data)))
    else:
        data = get_poly_layer_from_geojson(os.path.join(BATHYMETRY_DIRECTORY, filename), latlons_flipped=True, min_area=5.0)
        if difference is not None:
            data = cut_bathymetry_inplace(data, difference)

        write_geometries_to_file(cache_file, data)
        print("processed and written to cache: {}".format(filename))

    return data

BATHYMETRY_DIRECTORY = "world_data/gebco_2020_geotiff"

if DRAW_BATHYMETRY:

    timer_start = datetime.now()

    bathymetry = []

    filenames = [
        "gebco_2020_n0.0_s-90.0_w-90.0_e0.0.",
        # "gebco_2020_n0.0_s-90.0_w-180.0_e-90.0.",
        # "gebco_2020_n0.0_s-90.0_w0.0_e90.0.",
        # "gebco_2020_n0.0_s-90.0_w90.0_e180.0.",
        # "gebco_2020_n90.0_s0.0_w-90.0_e0.0.",
        # "gebco_2020_n90.0_s0.0_w-180.0_e-90.0.",
        # "gebco_2020_n90.0_s0.0_w0.0_e90.0.",
        # "gebco_2020_n90.0_s0.0_w90.0_e180.0."
    ]

    num_layers = 15
    min_height = -9000
    max_height = 0
    format_options = [min_height, max_height, num_layers]

    for i in range(0, num_layers):
        bathymetry.append([])

    for filename in filenames:
        data = load_bathymetry_file(filename + "{}_{}_{}.geojson".format(*format_options), difference=coastlines)

        for i in range(0, len(data)):
            bathymetry[i] = bathymetry[i] + data[i]


    print(TIMER_STRING.format("parsing sea data", (datetime.now()-timer_start).total_seconds()))     
    timer_start = datetime.now()

    add_layers_to_writer(bathymetry)

    print(TIMER_STRING.format("preparing sea data", (datetime.now()-timer_start).total_seconds()))  


# --------------------------------------------------------------------------------

if DRAW_TERRAIN:

    TERRAIN_DIRECTORY = "world_data/gebco_2020_geotiff"

    timer_start = datetime.now()

    terrain = []

    filenames = [
        "gebco_2020_n0.0_s-90.0_w-90.0_e0.0.",
        # "gebco_2020_n0.0_s-90.0_w-180.0_e-90.0.",
        # "gebco_2020_n0.0_s-90.0_w0.0_e90.0.",
        # "gebco_2020_n0.0_s-90.0_w90.0_e180.0.",
        # "gebco_2020_n90.0_s0.0_w-90.0_e0.0.",
        # "gebco_2020_n90.0_s0.0_w-180.0_e-90.0.",
        # "gebco_2020_n90.0_s0.0_w0.0_e90.0.",
        # "gebco_2020_n90.0_s0.0_w90.0_e180.0."
    ]

    num_layers = 90
    min_height = 0
    max_height = 9000
    format_options = [min_height, max_height, num_layers]

    for i in range(0, num_layers):
        terrain.append([])

    for filename in filenames:
        data = load_bathymetry_file(filename + "{}_{}_{}.geojson".format(*format_options))

        for i in range(0, len(data)):
            terrain[i] = terrain[i] + data[i]

    print(TIMER_STRING.format("parsing terrain data", (datetime.now()-timer_start).total_seconds()))   

    timer_start = datetime.now()  

    # polygons to linestrings
    terrain = polygons_to_linestrings(terrain)

    # cutouts for city circles
    terrain = cut_linestrings_inplace(terrain, cities)

    print(TIMER_STRING.format("converting terrain data", (datetime.now()-timer_start).total_seconds()))  

# --------------------------------------------------------------------------------

timer_start = datetime.now()

# ---

coastlines_line = polygons_to_linestrings(coastlines)
coastlines_line = cut_linestrings_inplace(coastlines_line, cities)
for coastline in coastlines_line:
    lines = []

    if type(coastline) is MultiLineString:
        for g in coastline.geoms:
            lines.append(g)
    else:
        lines.append(coastline)

    for line in lines:
        svg.add_poly_line(list(line.coords), stroke_width=1.0, stroke=[255, 255, 255], layer="coastlines")

# ---

for place in places:
    svg.add_circles(list(place.coords), radius=0.5, layer="places")

# ---

for city in cities:
    svg.add_polygon(city.buffer(CITY_CIRCLE_RADIUS), stroke_width=1.0, opacity=0, stroke=[255, 255, 255], layer="places")

# ---

for u in urban:
    if not u.is_valid:
        continue
    svg.add_polygon(u, stroke_width=0.2, opacity=0.3, stroke=[255, 255, 255], layer="urban")

# ---

# for border in borders:
#     lines = []

#     if type(border) is MultiLineString:
#         lines += list(border.geoms)
#     else:
#         lines += [border]

#     for lineString in lines:
#         svg.add_polygon(list(lineString.coords), stroke_width=0.2, opacity=0.0, layer="borders")

for border in borders:
    svg.add_poly_line(list(border.coords), stroke_width=0.6, stroke=[255, 255, 255], layer="borders")

# ---

for a in admin:
    svg.add_poly_line(list(a.coords), stroke_width=0.2, stroke=[255, 255, 255], layer="borders")

# ---

# for i in range(0, len(bathymetry)):
#     options = {
#         "stroke_width": 0.2,
#         "opacity": 0.0,
#         "stroke": [11, 72, 107],
#         "layer": "bathymetry"
#     }
#     p = bathymetry[i]
#     if not p.is_valid:
#         print("error: polygon {}/{} not valid".format(i, len(bathymetry)))
#         continue
#     svg.add_polygon(p, **options)

# ---

for terrain_line in terrain:
    lines = []

    if type(terrain_line) is MultiLineString:
        for g in terrain_line.geoms:
            lines.append(g)

    for line in lines:
        svg.add_poly_line(list(line.coords), stroke_width=0.25, layer="terrain")

# ---

print(TIMER_STRING.format("preparing SVG writer", (datetime.now()-timer_start).total_seconds())) 

# add fiducial

options = {
    "stroke_width": 1.0,
    "layer": "meta"
}

for r in [10, 15, 20]:
    svg.add_polygon(Point(100, 100).buffer(r), **options, opacity=0)
svg.add_line([[100, 100-50], [100, 100+50]], **options)
svg.add_line([[100-50, 100], [100+50, 100]], **options)

svg.save()

print(TIMER_STRING.format(maptools.Color.BOLD + "time total" + maptools.Color.END, (datetime.now()-timer_total).total_seconds()))