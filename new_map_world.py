from datetime import datetime
import os
import pickle

from svgwriter import SvgWriter
import maptools

import lxml.etree as ET

import shapely
from shapely.wkb import dumps, loads
from shapely import ops
from shapely.prepared import prep
from shapely.geometry import Point, GeometryCollection, MultiLineString, LineString, Polygon, MultiPolygon
from shapely.geometry import shape

import fiona

from HersheyFonts import HersheyFonts

TIMER_STRING                    = "{:<60s}: {:2.2f}s"

DB_NAME                         = "import"
DB_PREFIX                       = "osm_"

CACHE_DIRECTORY                 = "cache"

MAP_CENTER                      = [0, 0]

# MAP_SIZE                        = [2000, 2000] # unit for data: m / unit for SVG elements: px or mm
# VIEWPORT_SIZE                   = [MAP_SIZE[0], 1000]
# VIEWPORT_OFFSET                 = [0, 300]

MAP_SIZE                        = [3000, 3000] # unit for data: m / unit for SVG elements: px or mm
VIEWPORT_SIZE                   = [750, 1000]
VIEWPORT_OFFSET                 = [(MAP_SIZE[0]-VIEWPORT_SIZE[0])//2, 900]

# FULL MAP
VIEWPORT_SIZE                   = [750*4, 1000*2]
VIEWPORT_OFFSET                 = [0, 500]
MAP_FRAGMENT_OFFSET             = VIEWPORT_OFFSET
MAP_FRAGMENT_SIZE               = VIEWPORT_SIZE

# TILE MAP
# tile_x = 1
# tile_y = 0
# MAP_FRAGMENT_OFFSET             = [0, 500]
# MAP_FRAGMENT_SIZE               = [750*4, 1000*2] # the full map comprising all tiles
# VIEWPORT_SIZE                   = [750-4, 1000-4]
# VIEWPORT_OFFSET                 = [MAP_FRAGMENT_OFFSET[0]+750*tile_x+4/2, MAP_FRAGMENT_OFFSET[1]+1000*tile_y+4/2]

# ---

MAP_SIZE_SCALE                  = maptools.EQUATOR/MAP_SIZE[0]      # increase or decrease MAP_SIZE by factor

SIMPLIFICATION_MAX_ERROR        = 0.1 #1.0 # 0.2                    # unit in map coordinates (px or mm)

THRESHOLD_CITY_POPULATION       = 1000000

CONNECT_DATABASE                = False

FONT_SIZE                       = 5
FONT_SIZE_LARGE                 = 12

DRAW_META                       = True
DRAW_LARGE_LABELS               = True
DRAW_COASTLINE                  = True
DRAW_PLACES                     = False
DRAW_CITIES                     = False
DRAW_CITIES_WTIH_LABELS         = True
DRAW_URBAN_AREAS                = False
DRAW_BORDERS                    = False
DRAW_ADMIN_REGIONS              = False
DRAW_BATHYMETRY                 = True
DRAW_TERRAIN                    = True

# --------------------------------

CITY_CIRCLE_RADIUS              = 2.0

DARK_MODE                       = False

""" ------------------------------

Projection:

Map is using OSMs Web-Mercator (3857) projection
Natural Earth Shapefiles are encoded in WGS84 (4326)

------------------------------ """

timer_total = datetime.now()

if CONNECT_DATABASE:
    import psycopg2
    conn = psycopg2.connect(database='osm', user='osm')
    curs = conn.cursor()

conv = maptools.Converter(MAP_CENTER, MAP_SIZE, MAP_SIZE_SCALE)

bg_color = "gray"
if DARK_MODE:
    bg_color = "black"

# svg = SvgWriter("world.svg", dimensions=MAP_SIZE, background_color="gray")
svg = SvgWriter("world.svg", dimensions=VIEWPORT_SIZE, offset=VIEWPORT_OFFSET, background_color=bg_color)

print("map size: {:.2f} x {:.2f} meter".format(MAP_SIZE[0]*MAP_SIZE_SCALE, MAP_SIZE[1]*MAP_SIZE_SCALE))
print("svg size: {:.2f} x {:.2f} units".format(*MAP_SIZE))

viewport_polygon = Polygon([
    [VIEWPORT_OFFSET[0],                    VIEWPORT_OFFSET[1]],
    [VIEWPORT_OFFSET[0]+VIEWPORT_SIZE[0],   VIEWPORT_OFFSET[1]],
    [VIEWPORT_OFFSET[0]+VIEWPORT_SIZE[0],   VIEWPORT_OFFSET[1]+VIEWPORT_SIZE[1]],
    [VIEWPORT_OFFSET[0],                    VIEWPORT_OFFSET[1]+VIEWPORT_SIZE[1]]
])

svg.add_layer("urban")
svg.add_layer("borders")
svg.add_layer("bathymetry")
svg.add_layer("terrain")
svg.add_layer("coastlines")
svg.add_layer("coastlines_hatching")
svg.add_layer("places")
svg.add_layer("places_circles")
svg.add_layer("large_labels")
svg.add_layer("meta")
svg.add_layer("meta_text")

hfont = HersheyFonts()
hfont.load_default_font("futural")
hfont.normalize_rendering(FONT_SIZE)

hfont_large = HersheyFonts()
hfont_large.load_default_font("futuram")
# print(hfont_large.default_font_names)
# exit()
hfont_large.normalize_rendering(FONT_SIZE_LARGE)

svg.add_hatching("coastline_hatching", stroke_width=0.5, distance=2.0)

# for i in range(0, 15):
#     svg.add_hatching("bathymetry_hatching_{}".format(i), stroke_width=0.25, distance=1.1**i) #1+0.5*i)
#     # svg.add_hatching("bathymetry_hatching_{}".format(i), stroke_width=0.5, distance=(2**i)/2.0)

if DARK_MODE:
    for i in range(0, 15):

        if i < 5:
            distance = 6
        else:
            distance = 1.5 + 1.2**(14-4) - (1.2**(i-4) - 1.2**(5-4))

        svg.add_hatching("bathymetry_hatching_{}".format(i), stroke_width=0.5, stroke_opacity=0.5, distance=distance)
else:
    for i in range(0, 15):

        if i < 5:
            distance = 1
        else:
            distance = 1.15**(i-4)

        svg.add_hatching("bathymetry_hatching_{}".format(i), stroke_width=0.5, stroke_opacity=0.5, distance=distance)

# for i in range(0, 15):

#     dist = 1.4

#     if i % 2 == 0:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45, stroke_width=0.5, distance=dist**(i-i%2)) 
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**(i-i%2))
#     else:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45, stroke_width=0.5, distance=dist**(i-i%2))


# for i in range(0, 15):

#     dist = 1.1

#     if i % 2 == 0:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**i)
#     else:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45, stroke_width=0.5, distance=dist**i)

# for i in range(0, 15):

#     dist = 1.2

#     if i % 2 == 0:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=(dist**i)/1.0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=(dist**i)/1.0)
#     else:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=(dist**(i-1))/1.0)


# for i in range(0, 15):

#     dist = 1.5

#     if i == 0:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**0)
    
#     if i == 1:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**1)
    
#     if i == 2:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**2)

#     if i == 3:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**3)

#     if i == 4:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**0)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 5:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**1)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 6:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**2)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 7:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**3)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 8:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**4)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 9:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**5)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 10:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**6)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 11:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**7)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**4)

#     if i == 12:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**7)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**5)

#     if i == 13:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**7)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**6)

#     if i == 14:
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45,     stroke_width=0.5, distance=dist**7)
#         svg.add_hatching("bathymetry_hatching_{}".format(i), orientation=SvgWriter.HATCHING_ORIENTATION_45_REV, stroke_width=0.5, distance=dist**7)

exclusion_zones = []

coastlines = []
places = []
urban = []
borders = []
admin = []
bathymetry = []
terrain = []

def get_text(font, text):

    lines_raw = font.lines_for_text(text)
    lines_restructured = []
    for (x1, y1), (x2, y2) in lines_raw:
        lines_restructured.append([[x1, y1], [x2, y2]])
    lines = MultiLineString(lines_restructured)

    return lines


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

    counter = 0
    num_items = len(shapefile)
    for item in shapefile:
        counter += 1
        print("processing item {}/{} in {}".format(counter, num_items, filename), end="\r")
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

    print("")

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

# returns list of polys
def _smooth_poly(p, factor):

    p = p.buffer(factor).buffer(-factor)

    if type(p) is Polygon:

        if not p.is_valid:
            p = p.buffer(0.01)

        return [p]
    else:
        res = []

        for g in p.geoms:

            if not g.is_valid:
                g = g.buffer(0.01)

            res.append(g)

        return res 


def _polygons_to_linestrings(polygons):
    linestrings = []
    for poly in polygons:

        subpolys = []

        if not poly.is_valid:
            poly = poly.buffer(0.01)
        subpolys.append(poly)

        for p in subpolys:

            outline = p.boundary

            if type(outline) is MultiLineString:
                for g in outline.geoms:
                    linestrings.append(g)
            else:
                linestrings.append(outline)

    return linestrings

def polygons_to_linestrings(polygons, flatten=True):

    linestrings = []

    # polygons is a list of layers
    if len(polygons) > 0 and type(polygons[0]) is list: 
        for layer in polygons:
            if flatten:
                linestrings += _polygons_to_linestrings(layer)
            else:
                linestrings.append(_polygons_to_linestrings(layer))
          
    # polygons is a list of polygons  
    else: 
        linestrings = _polygons_to_linestrings(polygons)

    return linestrings


def _unpack_geometrycollection(m):

    polys = []

    if type(m) is GeometryCollection:
        for g in m.geoms:
            polys += _unpack_geometrycollection(g)
    elif type(m) is MultiPolygon:
        for g in m.geoms:
            polys += _unpack_geometrycollection(g)
    elif type(m) is Polygon:
        polys.append(m)
    else:
        print("unknown geometry: {}".format(type(m)))

    return polys



def polygons_merge_tiles(polys):

    unified_poly = ops.unary_union(polys)
    return _unpack_geometrycollection(unified_poly)


# expects city centers
# def cut_linestrings_inplace(linestrings, centers):

#     circles = []
#     for center in centers:
#         circles.append(center.buffer(CITY_CIRCLE_RADIUS + 3).simplify(SIMPLIFICATION_MAX_ERROR))

#     for i in range(0, len(linestrings)):

#         if i%10 == 0:
#             print("cut {:10}/{:10} ({:5.2f})".format(i, len(linestrings), (i/len(linestrings))*100), end="\r")
        
#         poly = linestrings[i]
#         poly = poly.simplify(SIMPLIFICATION_MAX_ERROR)
#         poly_bounds = poly.bounds
        
#         for circle in circles:    

#             if maptools.check_polygons_for_overlap(poly_bounds, circle.bounds):
#                 poly = poly.difference(circle)
        
#         linestrings[i] = poly

#     print("") # newline to counter \r

#     return linestrings

def cut_linestrings(linestrings, cut_polys): # may return [MultiLineString, LineString, ...]

    linestrings_processed = []

    cut_poly = ops.unary_union(cut_polys)
    cut_poly = cut_poly.simplify(SIMPLIFICATION_MAX_ERROR)

    for i in range(0, len(linestrings)):

        if i%10 == 0:
            print("cut {:10}/{:10} ({:5.2f})".format(i, len(linestrings), (i/len(linestrings))*100)) #, end="\r")
        
        line = linestrings[i]
        line = line.simplify(SIMPLIFICATION_MAX_ERROR)

        for l in validate_linestring(line):   
            linestrings_processed.append(l.difference(cut_poly))

    return linestrings_processed

"""
input: [Polygon, ...]
output: [Polygon, ...]

"""
def cut_polygons(polygons, cut_polys):

    polygons_processed = []

    for i in range(0, len(polygons)):

        if i%10 == 0:
            print("cut {:10}/{:10} ({:5.2f})".format(i, len(polygons), (i/len(polygons))*100), end="\r")
        
        poly = polygons[i]
        poly = poly.simplify(SIMPLIFICATION_MAX_ERROR)
        poly_bounds = poly.bounds
        
        for cut_poly in cut_polys:    

            if maptools.check_polygons_for_overlap(poly_bounds, cut_poly.bounds):
                poly = poly.difference(cut_poly)
        
        polygons_processed.append(poly)

    print("") # newline to counter \r

    return polygons_processed

def recalculate_exclusion_zones(zones):

    zones_simplified = []

    for z in zones:
        zones_simplified.append(z.simplify(SIMPLIFICATION_MAX_ERROR))

    zones_simplified = [ops.unary_union(zones_simplified)]

    # TODO: check for validity

    return zones_simplified

def validate_polygon(poly):

    p = poly

    if p.area <= 1:
        return []

    if not p.is_valid:

        p = p.buffer(0.01)

        if not p.is_valid:
            print("error: polygon not valid")
            return []

    if type(p) is MultiPolygon:
        return list(p.geoms)
    else:
        return [p]

def validate_linestring(line):

    l = line

    if not l.is_valid:
        print("error: LineString not valid")
        return []

    if l.is_empty:
        print("error: LineString is empty")
        return []

    if not len(l.bounds) == 4:
        print("error: LineString is empty (bounds)")
        return []

    if type(l) is MultiLineString:
        lines = []
        for g in l.geoms:
            lines += validate_linestring(g)
        return lines
    else:
        return [l]

# --------------------------------------------------------------------------------

# add fiducial

if DRAW_META:

    options = {
        "stroke_width": 2.0,
        "layer": "meta",
        "stroke": [0, 0, 0],
        "opacity": 0
    }

    options_text = {
        "stroke_width": 0.5,
        "layer": "meta_text",
        "stroke": [0, 0, 0],
    }

    if DARK_MODE:
        options["stroke"] = [255, 255, 255]
        options_text["stroke"] = [255, 255, 255]

    # ROSE_OFFSET = [0, 0]
    # rose_center = [VIEWPORT_OFFSET[0] + VIEWPORT_SIZE[0] + ROSE_OFFSET[0], VIEWPORT_OFFSET[1] + VIEWPORT_SIZE[1] - ROSE_OFFSET[1]] # GCODE coordinate system is bottom-left

    # exclusion_zones.append(Point(rose_center).buffer(15+2))
    # exclusion_zones.append(LineString([(rose_center[0], rose_center[1]-20), (rose_center[0], rose_center[1]+20)]).buffer(2))
    # exclusion_zones.append(LineString([(rose_center[0]-20, rose_center[1]), (rose_center[0]+20, rose_center[1])]).buffer(2))

    # for r in [5, 10, 15]:
    #     svg.add_polygon(Point(rose_center).buffer(r), **options)
    # svg.add_line([[rose_center[0], rose_center[1]-20], [rose_center[0], rose_center[1]+20]], **options)
    # svg.add_line([[rose_center[0]-20, rose_center[1]], [rose_center[0]+20, rose_center[1]]], **options)

    # add tiles

    options_tile = {
        "stroke_width": 4.0,
        "layer": "meta",
        "stroke": [0, 0, 0],
        "opacity": 0
    }

    svg.add_line([[0, 500+1000*1], [3000, 500+1000*1]], **options_tile)
    svg.add_line([[0, 500+1000*2], [3000, 500+1000*2]], **options_tile)
    svg.add_line([[0, 500+1000*3], [3000, 500+1000*3]], **options_tile)

    svg.add_line([[750*1, 500], [750*1, 2500]], **options_tile)
    svg.add_line([[750*2, 500], [750*2, 2500]], **options_tile)
    svg.add_line([[750*3, 500], [750*3, 2500]], **options_tile)
    svg.add_line([[750*4, 500], [750*4, 2500]], **options_tile)

    # screw holes

    TILE_SIZE = [750, 1000]
    TILE_NUMBERS = [4, 2]
    HOLE_DIST = 15
    positions = [
        [MAP_FRAGMENT_OFFSET[0]+HOLE_DIST,                MAP_FRAGMENT_OFFSET[1]+HOLE_DIST],
        [MAP_FRAGMENT_OFFSET[0]+TILE_SIZE[0]-HOLE_DIST,   MAP_FRAGMENT_OFFSET[1]+HOLE_DIST],
        [MAP_FRAGMENT_OFFSET[0]+TILE_SIZE[0]-HOLE_DIST,   MAP_FRAGMENT_OFFSET[1]+TILE_SIZE[1]-HOLE_DIST],
        [MAP_FRAGMENT_OFFSET[0]+HOLE_DIST,                MAP_FRAGMENT_OFFSET[1]+TILE_SIZE[1]-HOLE_DIST],
    ]

    for col in range(0, TILE_NUMBERS[0]):
        for row in range(0, TILE_NUMBERS[1]):

            tile_origin = [TILE_SIZE[0]*col, TILE_SIZE[1]*row]

            for pos in positions:

                x = tile_origin[0]+pos[0]
                y = tile_origin[1]+pos[1]

                p = Point([x, y])

                if not viewport_polygon.contains(p):
                    continue

                svg.add_line([[x-2, y-2], [x+2, y+2]], **options_text)
                svg.add_line([[x+2, y-2], [x-2, y+2]], **options_text)

                # svg.add_polygon(p.buffer(2), **options)
                svg.add_polygon(p.buffer(3+1), **options)
                exclusion_zones.append(p.buffer(7))

    # lat lon lines

    # TODO: cut with viewport

    latlonlines = []

    color = [0, 0, 0]
    if DARK_MODE:
        color = [255, 255, 255]

    NUM_LINES_LAT = 24*2

    for i in range(1, NUM_LINES_LAT): # lat 

        deg = 90 - (180/NUM_LINES_LAT)*i

        text_lines = get_text(hfont, "{:5.1f}".format(deg))
        text_lines = shapely.affinity.scale(text_lines, xfact=1, yfact=-1, origin=Point(0, 0))

        text_lines1 = shapely.affinity.translate(text_lines, xoff=2, yoff=MAP_SIZE[1]*i/NUM_LINES_LAT-1)
        for line in text_lines1.geoms:
            l = list(line.coords)

            if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                continue

            svg.add_line(l, stroke=color, layer="meta_text")
        
        exclusion_zones.append(text_lines1.buffer(3).simplify(SIMPLIFICATION_MAX_ERROR))

        text_lines2 = shapely.affinity.translate(text_lines, xoff=MAP_FRAGMENT_SIZE[0]-20, yoff=MAP_SIZE[1]*i/NUM_LINES_LAT-1)
        for line in text_lines2.geoms:
            l = list(line.coords)

            if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                continue

            svg.add_line(l, stroke=color, layer="meta_text")
        
        exclusion_zones.append(text_lines2.buffer(3).simplify(SIMPLIFICATION_MAX_ERROR))

        # exclusion_zones.append(Polygon())

        line = LineString([[0, MAP_SIZE[1]*i/NUM_LINES_LAT],                                        [1+20, MAP_SIZE[1]*i/NUM_LINES_LAT]])
        exclusion_zones.append(line.buffer(2))
        latlonlines.append(line)

        line = LineString([[MAP_SIZE[0]-20, MAP_SIZE[1]*i/NUM_LINES_LAT],                           [MAP_SIZE[0], MAP_SIZE[1]*i/NUM_LINES_LAT]])
        exclusion_zones.append(line.buffer(2))
        latlonlines.append(line)

    NUM_LINES_LON = 24*4

    for i in range(1, NUM_LINES_LON): # lon

        line_length = 10

        if i % 2 == 0:
            line_length = 20

            deg = (180/NUM_LINES_LON)*i

            text_lines = get_text(hfont, "{:5.1f}".format(deg))
            text_lines = shapely.affinity.scale(text_lines, xfact=1, yfact=-1, origin=Point(0, 0))

            text_lines1 = shapely.affinity.translate(text_lines, xoff=MAP_SIZE[0]*i/NUM_LINES_LON+2, yoff=MAP_FRAGMENT_OFFSET[1]+20)
            for line in text_lines1.geoms:
                l = list(line.coords)

                if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                    continue

                svg.add_line(l, stroke=color, layer="meta_text")

            exclusion_zones.append(text_lines1.buffer(3).simplify(SIMPLIFICATION_MAX_ERROR))

            text_lines2 = shapely.affinity.translate(text_lines, xoff=MAP_SIZE[0]*i/NUM_LINES_LON+2, yoff=MAP_FRAGMENT_OFFSET[1]+MAP_FRAGMENT_SIZE[1]-14)
            for line in text_lines2.geoms:
                l = list(line.coords)

                if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                    continue

                svg.add_line(l, stroke=color, layer="meta_text")

            exclusion_zones.append(text_lines2.buffer(3).simplify(SIMPLIFICATION_MAX_ERROR))

        line = LineString([[MAP_SIZE[0]*i/NUM_LINES_LON, MAP_FRAGMENT_OFFSET[1]],                               [MAP_SIZE[0]*i/NUM_LINES_LON, MAP_FRAGMENT_OFFSET[1]+line_length]])
        exclusion_zones.append(line.buffer(2))
        latlonlines.append(line)

        line = LineString([[MAP_SIZE[0]*i/NUM_LINES_LON, MAP_FRAGMENT_OFFSET[1]+MAP_FRAGMENT_SIZE[1]-line_length],  [MAP_SIZE[0]*i/NUM_LINES_LON, MAP_FRAGMENT_OFFSET[1]+MAP_FRAGMENT_SIZE[1]]])
        exclusion_zones.append(line.buffer(2))
        latlonlines.append(line)

    for line in latlonlines:

        save_line = line.intersection(viewport_polygon)
        if save_line.is_empty:
            continue

        svg.add_line(save_line.coords, **options)   

# --------------------------------------------------------------------------------

if DRAW_LARGE_LABELS:

    # LABEL_FILESNAMES = ["pacificocean.svg"]

    # for filename in LABEL_FILESNAMES:

    #     tree = ET.parse(filename)  
    #     root = tree.getroot()

    #     DEFAULT_NS = "{" + root.nsmap[None] + "}"

    #     lines = []

    #     for layer in root.findall("g", root.nsmap):

    #         for e in layer:
                    
    #             if e.tag == DEFAULT_NS + "path":
    #                 d = e.attrib["d"]
    #                 d = d[1:] # cut off the M
    #                 segments = d.split("L")

    #                 l = []

    #                 for s in segments:
    #                     pairs = s.split(" ")
    #                     l.append([float(pairs[0]), float(pairs[1])])

    #                 for i in range(1, len(l)):
    #                     lines.append([l[i-1][0], l[i-1][1], l[i][0], l[i][1]])

    #     print(lines)
    #     exit()


    labels = [
        [[40.5,     -150],  "Nordpazifik"],
        [[57,       168],   "Beringmeer"],
        [[-40.6,    -148],  "Pazifik"],
        # [[24.201523, -93.767498], "Golf von Mexiko"],
        [[14.2,     -78],   "Karibisches Meer"],
        [[59.7,     -88],   "Hudson"],
        [[58.7,     -86],   "Bay"],
        # [[66.004326, -80.944026], "Nordwestpassage"],
        [[74,       -71],   "Baffin Bay"],
        [[56,       -53],   "Labradorsee"],
        [[33,       -42],   "Nordatlantik"],
        [[-38,      -30],   "Atlantik"],
        [[68,       -15],   "Nordmeer"],
        # [[77.532554, -9.455168], "Groenlandsee"],
        [[72,       35],    "Barentssee"],
        [[74,       63],    "Karasee"],
        [[34.2,     13.2],  "Mittelmeer"],
        # [[0.754504, -2.021039], "Golf von Guinea"],
        [[12,       58.2],  "Arabisches Meer"],
        [[-20,      73.5],  "Indischer Ozean"],
        # [[11,       82],    "Golf von Bengalen"],
        [[-40,      154],   "Tasmanische See"],
        [[-16,      151],   "Korallenmeer"],
        [[25.5,     125],   "Chinesisches Meer"],
        [[39.3,     130.5], "Jap. Meer"],
        [[56,       142],   "Ochotskisches"],
        [[55,       146],   "Meer"],
        [[73,       160],   "Ostsibirische See"],
    ]

    for l in labels:

        pos, label = l

        if len(pos) == 0:
            continue

        c = conv.convert_wgs_to_map(*pos)

        text_lines = get_text(hfont_large, label)
        text_lines = shapely.affinity.scale(text_lines, xfact=1, yfact=-1, origin=Point(0, 0))
        text_lines = shapely.affinity.translate(text_lines, xoff=c[0], yoff=c[1])

        for line in text_lines.geoms:
            l = list(line.coords)
            if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                continue

            color = [0, 0, 0]
            if DARK_MODE:
                color = [255, 255, 255]

            svg.add_line(l, stroke=color, stroke_width=1.5, layer="large_labels")
        
        exclusion_zones.append(text_lines.buffer(4+1).buffer(-1).simplify(SIMPLIFICATION_MAX_ERROR)) 


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

cities = []
cities_names = []

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

            cities_names.append(item["properties"]["NAMEASCII"])


for i in range(0, len(cities)):
    city_pos = cities[i]
    city_name = cities_names[i]
    svg.add_polygon(city_pos.buffer(CITY_CIRCLE_RADIUS), stroke_width=0.5, opacity=0, stroke=[255, 255, 255], layer="places_circles")

    c = list(city_pos.coords)[0]
    # for (x1, y1), (x2, y2) in thefont.lines_for_text(city_name):
    #     svg.add_line([
    #         [x1 + c[0], y1 + c[1]], 
    #         [x2 + c[0], y2 + c[1]]
    #     ])

    text_lines = get_text(thefont, city_name)

    # height = 0
    # minx, miny, maxx, maxy = text_lines.bounds
    # height = maxy - miny

    text_lines = shapely.affinity.scale(text_lines, xfact=1, yfact=-1, origin=Point(0, 0))
    text_lines = shapely.affinity.translate(text_lines, xoff=c[0]+CITY_CIRCLE_RADIUS*2, yoff=c[1]-1.0)

    for line in text_lines.geoms:
        l = list(line.coords)
        svg.add_line(l)

# --------------------------------------------------------------------------------

if DRAW_CITIES_WTIH_LABELS:

    CITIES_FILE = "labeltest/Dymo/geojson/world-townspots-z5.json"
    LABELS_FILE = "labeltest/Dymo/geojson/world-labels-z5.json"

    city_name   = []
    city_pos    = []
    city_label  = []

    for item in fiona.open(CITIES_FILE):
        shapefile_geom = shape(item["geometry"])
        geom = ops.transform(conv.convert_wgs_to_map_list_lon_lat, shapefile_geom)
        geom = geom.simplify(SIMPLIFICATION_MAX_ERROR)

        if not type(geom) is Point:
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))

        # for key in item["properties"]:
        #     print("{} : {}".format(key, item["properties"][key]))
        # exit()

        city_pos.append(geom)
        city_name.append(item["properties"]["name"]) # label placement computation is done with name, not asciiname
        # city_name.append(item["properties"]["asciiname"])

    for item in fiona.open(LABELS_FILE):
        shapefile_geom = shape(item["geometry"])
        geom = ops.transform(conv.convert_wgs_to_map_list_lon_lat, shapefile_geom)
        # geom = geom.buffer(-1).buffer(+1) # rounded corners
        # geom = geom.simplify(SIMPLIFICATION_MAX_ERROR)

        if not type(geom) is Polygon:
            raise Exception("parsing shapefile: unexpected type: {}".format(geom))

        city_label.append(geom)

    for i in range(0, len(city_name)):

        if not viewport_polygon.contains(city_pos[i]):
            continue

        svg.add_polygon(city_pos[i].buffer(CITY_CIRCLE_RADIUS), stroke_width=0.5, opacity=0, stroke=[255, 255, 255], layer="places_circles")
        # svg.add_polygon(city_label[i], stroke_width=0.5, opacity=0, stroke=[0, 0, 0], layer="places")

        minx, _, _, maxy = city_label[i].bounds
        c = [minx, maxy]
        text_lines = get_text(hfont, city_name[i])
        text_lines = shapely.affinity.scale(text_lines, xfact=1, yfact=-1, origin=Point(0, 0))
        text_lines = shapely.affinity.translate(text_lines, xoff=c[0]+CITY_CIRCLE_RADIUS-0.75, yoff=c[1]+0.4)

        for line in text_lines.geoms:
            l = list(line.coords)
            if not viewport_polygon.contains(Point(l[0])) or not viewport_polygon.contains(Point(l[1])):
                continue

            color = [0, 0, 0]
            if DARK_MODE:
                color = [255, 255, 255]

            svg.add_line(l, stroke=color, layer="places")

        exclusion_zones.append(city_pos[i].buffer(CITY_CIRCLE_RADIUS + 2))
        
        exclusion_zones.append(text_lines.buffer(3).simplify(SIMPLIFICATION_MAX_ERROR)) # use buffered hershey text instead of the label rect from the shapefile
        # exclusion_zones.append(city_label[i].buffer(1))

    exclusion_zones = recalculate_exclusion_zones(exclusion_zones)

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

    for border in borders:
        exclusion_zones.append(border.buffer(1).simplify(SIMPLIFICATION_MAX_ERROR))

    borders = []

    print(TIMER_STRING.format("loading border region data", (datetime.now()-timer_start).total_seconds())) 


# --------------------------------------------------------------------------------

if DRAW_ADMIN_REGIONS:

    ADMIN_REGIONS_FILE = "world_data/10m_cultural/10m_cultural/ne_10m_admin_1_states_provinces_lines.shp"

    timer_start = datetime.now()
    
    admin = get_lines_from_shapefile(ADMIN_REGIONS_FILE, latlons_flipped=True)

    print(TIMER_STRING.format("loading admin region data", (datetime.now()-timer_start).total_seconds())) 


# --------------------------------------------------------------------------------

def add_layers_to_writer(poly_layers, cut_polys=[]):

    options = {
        "stroke_width": 0, #0.5
        "opacity": 0,
        "layer": "bathymetry",
        "stroke": [11, 72, 107] # [0, 198, 189]
    }

    cut_poly = ops.unary_union(cut_polys)

    if DARK_MODE:
        options["stroke"] = [255, 255, 255]

    for layer_i in range(0, len(poly_layers)):

        polys = poly_layers[layer_i]
        hatching_name = "bathymetry_hatching_{}".format(layer_i)

        # if layer_i > 8:
        #     options["stroke"] = [11, 72, 107]

        for i in range(0, len(polys)):

            print("\33[2K   adding layer {}/{} | poly {}/{}".format(layer_i, len(poly_layers), i, len(polys)), end="\r")

            p = polys[i]
            p = p.difference(cut_poly)

            save_polys = validate_polygon(p)

            for save_poly in save_polys:
                
                save_poly = save_poly.intersection(viewport_polygon)
      
                for poly in validate_polygon(save_poly):
                    svg.add_polygon(poly, **options, hatching=hatching_name)
                    # svg.add_polygon(poly, **options)

                    # dist = 1 + 0.2*layer_i

                    # outline_polys = [poly]
                    # for d in range(0, 100):
                    #     for p in outline_polys: 
                    #         svg.add_polygon(p, **options)
                        
                    #     new_polys = []
                    #     for p in outline_polys:
                    #         p_smaller = p.buffer(-dist).simplify(SIMPLIFICATION_MAX_ERROR)
                    #         p_smaller_validated = validate_polygon(p_smaller)

                    #         for pv in p_smaller_validated:
                    #             if pv.area > 1.0:
                    #                 new_polys.append(pv)

                    #     outline_polys = new_polys
                    #     if len(outline_polys) == 0:
                    #         break




def cut_bathymetry_inplace(layers, tool_polys):

    current_poly = None
    tool_poly = None

    for i in range(0, len(layers)):
        layer = layers[i]
        for j in range(0, len(layer)):
            current_poly = layer[j]

            if current_poly.area < 1.0:
                continue

            print("cut bathymetry layer {}/{} || geometry {}/{}".format(i, len(layers), j, len(layer)), end="\r")

            for k in range(0, len(tool_polys)):
                tool_poly = tool_polys[k]

                if not current_poly.is_valid:
                    current_poly = current_poly.buffer(0.1)

                if not tool_poly.is_valid:
                    continue

                # if maptools.check_polygons_for_overlap(current_poly, tool_poly):
                current_poly = current_poly.difference(tool_poly)

            layer[j] = current_poly.simplify(SIMPLIFICATION_MAX_ERROR)

    print("")

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
        "gebco_2020_n0.0_s-90.0_w-180.0_e-90.0.",
        "gebco_2020_n0.0_s-90.0_w0.0_e90.0.",
        "gebco_2020_n0.0_s-90.0_w90.0_e180.0.",
        "gebco_2020_n90.0_s0.0_w-90.0_e0.0.",
        "gebco_2020_n90.0_s0.0_w-180.0_e-90.0.",
        "gebco_2020_n90.0_s0.0_w0.0_e90.0.",
        "gebco_2020_n90.0_s0.0_w90.0_e180.0."
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

    add_layers_to_writer(bathymetry, exclusion_zones)

    print(TIMER_STRING.format("preparing sea data", (datetime.now()-timer_start).total_seconds()))  


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

    coastlines_processed = []
    for i in range(0, len(coastlines)):
        c = coastlines[i]

        c = c.intersection(viewport_polygon)

        if c.area < 10:
            continue

        c = c.buffer(0.3).buffer(-0.3).simplify(SIMPLIFICATION_MAX_ERROR)

        if c.is_valid:
            coastlines_processed.append(c)
        else:
            print("error during postprocessing. coastline {}/{}".format(i, len(coastlines)))

    coastlines = coastlines_processed

    print(TIMER_STRING.format("postprocessing coastline data", (datetime.now()-timer_start).total_seconds())) 

# --------------------------------------------------------------------------------

if DRAW_TERRAIN:

    TERRAIN_DIRECTORY = "world_data/gebco_2020_geotiff"

    timer_start = datetime.now()

    terrain = []

    filenames = [
        "gebco_2020_n0.0_s-90.0_w-90.0_e0.0.",
        "gebco_2020_n0.0_s-90.0_w-180.0_e-90.0.",
        "gebco_2020_n0.0_s-90.0_w0.0_e90.0.",
        "gebco_2020_n0.0_s-90.0_w90.0_e180.0.",
        "gebco_2020_n90.0_s0.0_w-90.0_e0.0.",
        "gebco_2020_n90.0_s0.0_w-180.0_e-90.0.",
        "gebco_2020_n90.0_s0.0_w0.0_e90.0.",
        "gebco_2020_n90.0_s0.0_w90.0_e180.0."
    ]

    # num_layers = 90
    # min_height = 0
    # max_height = 9000

    num_layers = 30
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

    # Tile border smoothing
    # for i in range(0, len(terrain)):

    #     # Expand every polygon by a bit so polygon merging (to remove tile-borders) works
    #     # at the same time this smoothes the rather rough lines a bit

    #     for j in range(0, len(terrain[i])):
    #         terrain[i][j] = terrain[i][j].buffer(+1.0).buffer(-0.8)

    #     terrain[i] = polygons_merge_tiles(terrain[i])

    # polygons to linestrings
    terrain = polygons_to_linestrings(terrain, flatten=False)

    print(TIMER_STRING.format("converting terrain data", (datetime.now()-timer_start).total_seconds()))  
    timer_start = datetime.now()  

    for i in range(0, len(terrain)):
        layer = terrain[i]
        for j in range(0, len(layer)):
            layer[j] = layer[j].intersection(viewport_polygon)

    print(TIMER_STRING.format("postprocessing terrain data", (datetime.now()-timer_start).total_seconds())) 

# --------------------------------------------------------------------------------

timer_start = datetime.now()

# ---

coastlines_combined = ops.unary_union(coastlines)
coastlines_extended = coastlines_combined.buffer(4).difference(coastlines_combined)
coastlines_extended = coastlines_extended.intersection(viewport_polygon)
coastlines_extended = coastlines_extended.simplify(SIMPLIFICATION_MAX_ERROR)
coastlines_extended = cut_polygons(coastlines_extended, exclusion_zones)
for poly in coastlines_extended:
    if poly.area < 3:
        continue
    if type(poly) not in [Polygon, MultiPolygon]:
        print("warning: unknown geometry: {}".format(poly))
        continue
    svg.add_polygon(poly, stroke_width=0, hatching="coastline_hatching", layer="coastlines_hatching")

coastlines_line = polygons_to_linestrings(coastlines)
coastlines_line = cut_linestrings(coastlines_line, exclusion_zones)
for coastline in coastlines_line:
    lines = []

    if coastline.is_empty:
        continue

    if type(coastline) is MultiLineString:
        for g in coastline.geoms:
            lines.append(g)
    elif type(coastline) is LineString:
        lines.append(coastline)
    else:
        print("error: unknown geometry: {}".format(coastline))

    for line in lines:

        color = [0, 0, 0]
        if DARK_MODE:
            color = [255, 255, 255]

        # svg.add_poly_line(list(line.coords), stroke_width=1.5, stroke=[255, 255, 255], layer="coastlines")
        svg.add_poly_line(list(line.coords), stroke_width=1.5, stroke=color, layer="coastlines")

# ---

for place in places:
    svg.add_circles(list(place.coords), radius=0.5, layer="places")

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

for i in range(0, len(terrain)):
    layer = terrain[i]
    color = 0 # 0 + i * 3

    if DARK_MODE:
        color = 255

    layer_cut = cut_linestrings(layer, exclusion_zones)

    for terrain_line in layer_cut:
        lines = validate_linestring(terrain_line)

        for line in lines:
            svg.add_poly_line(list(line.coords), stroke=[color, color, color], stroke_width=0.5, layer="terrain")

# ---

print(TIMER_STRING.format("preparing SVG writer", (datetime.now()-timer_start).total_seconds())) 

# ---

svg.save()

print(TIMER_STRING.format(maptools.Color.BOLD + "time total" + maptools.Color.END, (datetime.now()-timer_total).total_seconds()))