# import xml.etree.ElementTree as ET  
import lxml.etree as ET

import sys
import math
from datetime import datetime

# Theaterplatz
# MAP_UP_LEFT     = (50.9808, 11.3248)
# MAP_DOWN_RIGHT  = (50.9790, 11.3271)

# Weimar
# MAP_UP_LEFT     = (50.9185, 11.4048)
# MAP_DOWN_RIGHT  = (51.0395, 11.2166)

MAP_CENTER      = (50.980467, 11.325000)

MAP_SIZE        = 250 # units. may be px, may be mm

class SvgWriter(object):

    def __init__(self, filename, dimensions=None, image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename   = filename
        self.dimensions = dimensions
        self.image      = image

        self.layers     = {}
        self.add_layer("default")

    def add_layer(self, layer_id):
        self.layers[layer_id] = {}
        self.layers[layer_id]["hexagons"]   = []
        self.layers[layer_id]["circles"]    = []
        self.layers[layer_id]["rectangles"] = []
        self.layers[layer_id]["polygons"]   = []
        self.layers[layer_id]["lines"]      = []
        self.layers[layer_id]["raw"]        = []

    def add_hexagons(self, hexagons, fills, layer="default"):
        for i in range(0, len(hexagons)):
            self.layers[layer]["hexagons"].append([
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                [fills[i][0]*255, fills[i][1]*255, fills[i][2]*255, fills[i][3]]
            ]) 

    def add_circles(self, circles, radius=3, fill=[255, 0, 0], layer="default"):
        for item in circles:
            self.layers[layer]["circles"].append([item, radius, fill])

    # coords: [[x, y], [width, height]]
    def add_rectangle(self, coords, stroke_width=1, stroke=[255, 0, 0], opacity=1.0, layer="default"):
        self.layers[layer]["rectangles"].append([*coords, stroke_width, stroke, opacity])

    def add_polygon(self, coords, stroke_width=1, stroke=[0, 0, 0], fill=[120, 120, 120], opacity=1.0, layer="default"):
        options = {}
        options["stroke-width"]     = stroke_width
        options["stroke"]           = stroke
        options["fill"]             = fill
        options["opacity"]          = opacity
        self.layers[layer]["polygons"].append((coords, options))

    def add_line(self, coords, stroke_width=1, stroke=[255, 0, 0], stroke_opacity=1.0, layer="default"):
        options = {}
        options["stroke-width"]     = stroke_width
        options["stroke"]           = stroke
        options["stroke-opacity"]   = stroke_opacity
        self.layers[layer]["lines"].append((coords, options))

    def add_raw_element(self, text, layer="default"):
        self.layers[layer]["raw"].append(text)

    def save(self):

        timer_start = datetime.now()

        with open(self.filename, "w") as out:

            out.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?>")
            out.write("<?xml-stylesheet href=\"style.css\" type=\"text/css\" title=\"main_stylesheet\" alternate=\"no\" media=\"screen\" ?>")
            
            if self.dimensions is not None:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" width=\"{}px\" height=\"{}px\" ".format(self.dimensions[0], self.dimensions[1]))
            else:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" ")
            out.write("xmlns=\"http://www.w3.org/2000/svg\" ")
            out.write("xmlns:ev=\"http://www.w3.org/2001/xml-events\" ")
            out.write("xmlns:xlink=\"http://www.w3.org/1999/xlink\" ")
            out.write("xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\" ")
            out.write(">")
            out.write("<defs />")

            if self.image is not None:
                out.write("<image x=\"0\" y=\"0\" xlink:href=\"{}\" />".format(self.image))

            # for h in self.hexagons:
            #     out.write("<path d=\"")
            #     for cmd in h[0]:
            #         out.write(cmd[0])
            #         if (len(cmd) > 1):
            #             out.write(str(cmd[1]))
            #             out.write(" ")
            #             out.write(str(cmd[2]))
            #             out.write(" ")
            #     # out.write("\" fill=\"rgba({},{},{},{})\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), int(h[1][3])))
            #     out.write("\" fill=\"rgb({},{},{})\" fill-opacity=\"{}\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), h[1][3]))
 
            # for c in self.circles:
            #     out.write("<circle cx=\"{}\" cy=\"{}\" fill=\"rgb({},{},{})\" r=\"{}\" />".format(c[0][0], c[0][1], c[2][0], c[2][1], c[2][2], c[1]))

            # for r in self.rectangles:
            #     out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], r[1], *r[2], r[3]))

            for layerid in self.layers.keys():

                if layerid == "default":
                    continue

                layer = self.layers[layerid]
                
                out.write("<g inkscape:groupmode=\"layer\" id=\"{0}\" inkscape:label=\"{0}\">".format(layerid))

                for c in layer["circles"]:
                    out.write("<circle cx=\"{}\" cy=\"{}\" fill=\"rgb({},{},{})\" r=\"{}\" />".format(c[0][0], c[0][1], c[2][0], c[2][1], c[2][2], c[1]))

                for r in layer["rectangles"]:
                    out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], *r[1], r[2], *r[3], r[4]))

                for poly in layer["polygons"]:
                    p = poly[0]
                    options = poly[1]
                    out.write("<path d=\"")
                    out.write("M{} {} ".format(float(p[0][0]), float(p[0][1])))
                    for point in p[1:]:
                        out.write("L")
                        out.write(str(float(point[0])))
                        out.write(" ")
                        out.write(str(float(point[1])))
                        out.write(" ")
                    out.write("Z\" ")
                    out.write("stroke-width=\"{}\" ".format(options["stroke-width"]))
                    out.write("stroke=\"rgb({},{},{})\" ".format(*options["stroke"]))
                    out.write("fill=\"rgb({},{},{})\" ".format(*options["fill"]))
                    out.write("fill-opacity=\"{}\" />".format(options["opacity"]))

                for line in layer["lines"]:
                    l = line[0]
                    options = line[1]
                    out.write("<path d=\"")
                    out.write("M{} {} ".format(float(l[0][0]), float(l[0][1])))
                    for point in l[1:]:
                        out.write("L")
                        out.write(str(float(point[0])))
                        out.write(" ")
                        out.write(str(float(point[1])))
                        out.write(" ")
                    out.write("\" ")
                    out.write("stroke-width=\"{}\" ".format(options["stroke-width"]))
                    out.write("stroke=\"rgb({},{},{})\" ".format(*options["stroke"]))
                    out.write("stroke-opacity=\"{}\" ".format(options["stroke-opacity"]))
                    out.write("fill=\"rgb({},{},{})\" ".format(0, 0, 0))
                    out.write("fill-opacity=\"{}\" />".format(0))
                    out.write("/>")

                for r in layer["raw"]:
                    out.write(r)

                out.write("</g>")

            out.write("</svg>")

        print("writing SVG in {0:.2f}s".format((datetime.now()-timer_start).microseconds/1000000))


class Converter(object):

    def __init__(self, map_up_left, map_down_right, map_size):

        self.north      = map_up_left[0]
        self.south      = map_down_right[0]
        self.west       = map_up_left[1]
        self.east       = map_down_right[1]

        self.north_px   = self._convert_lat(self.north)
        self.south_px   = self._convert_lat(self.south)
        self.west_px    = self._convert_lon(self.west)
        self.east_px    = self._convert_lon(self.east)

        # northern hemisphere, ...

        self.lat_diff = self.north - self.south
        self.lat_diff_px = self.south_px - self.north_px
        self.lon_diff = self.east - self.west
        self.lon_diff_px = self.east_px - self.west_px

        self.map_size_x = map_size
        self.map_size_y = map_size * (self.lat_diff_px / self.lon_diff_px) 

    def _convert_lat(self, lat):

        latRad  = (lat * math.pi) / 180.0
        mercN   = math.log(math.tan((math.pi / 4.0) + (latRad / 2.0)))
        y       = 0.5 - (mercN / (2.0*math.pi))

        return y

    def _convert_lon(self, lon):
        return (lon + 180.0) / 360.0

    def convert(self, lat, lon):

        x = ((self._convert_lon(lon) - self.west_px) / self.lon_diff_px) * self.map_size_x
        y = ((self._convert_lat(lat) - self.north_px) / self.lat_diff_px) * self.map_size_y

        return (x, y)

    def get_map_size(self):
        return (self.map_size_x, self.map_size_y)

    def all_elements_inside_boundary(self, coords):
        for x, y in coords:
            if x < 0:
                return False
            if x > self.map_size_x:
                return False
            if y < 0:
                return False
            if y > self.map_size_y:
                return False

        return True

    def all_elements_outside_boundary(self, coords):

        # for lat, lon in coords:
        #     if lat > self.north:
        #         return False
        #     if lat < self.south:
        #         return False
        #     if lon < self.west:
        #         return False
        #     if lon > self.east:
        #         return False

        for x, y in coords:
            if x > 0 and x < self.map_size_x:
                if y > 0 and y < self.map_size_y:
                    return False

        return True

    @staticmethod
    def get_bounding_box_in_latlon(center_point, width, height):
        map_up_left     = (center_point[0] + Converter._m_to_latlon(height/2), center_point[1] - Converter._m_to_latlon(width/2))
        map_down_right  = (center_point[0] - Converter._m_to_latlon(height/2), center_point[1] + Converter._m_to_latlon(width/2))

        return (map_up_left, map_down_right)

    @staticmethod
    def _m_to_latlon(m):
        return (m / 1.1) * 0.00001


bounding_box = Converter.get_bounding_box_in_latlon(MAP_CENTER, 2000, 1000)

MAP_UP_LEFT = bounding_box[0]
MAP_DOWN_RIGHT = bounding_box[1]

conv = Converter(MAP_UP_LEFT, MAP_DOWN_RIGHT, MAP_SIZE)
svg = SvgWriter("test.svg", conv.get_map_size())

svg.add_layer("meta")
svg.add_layer("buildings")
svg.add_layer("streets")
svg.add_layer("whatever")
svg.add_layer("marker")

tree = ET.parse("weimar.osm")  
# tree = ET.parse("weimar_theaterplatz.osm")  
root = tree.getroot()

# parse document

nodes = {}
for node in root.findall("./node"):
    nodes[node.attrib["id"]] = [float(node.attrib["lat"]), float(node.attrib["lon"])]

print("indexed {} nodes".format(len(nodes.keys())))

# --- MISC

svg.add_circles([conv.convert(*MAP_CENTER)], layer="meta")

xy1 = conv.convert(*MAP_UP_LEFT)
xy2 = conv.convert(*MAP_DOWN_RIGHT)
svg.add_rectangle([xy1, [xy2[0] - xy1[0], xy2[1] - xy1[1]]], stroke_width=0.2, layer="meta")

# --- STREETS

for way in root.findall("./way"):

    if len(way.findall("./tag[@k='building']")) > 0:
        continue

    if len(way.findall("./tag[@k='building:part']")) > 0:
        continue

    if len(way.findall("./tag[@k='landuse']")) > 0:
        continue

    # print(ET.tostring(way).decode())

    way_nodes = []

    for node_id in way.findall("./nd[@ref]"):
        way_nodes.append(conv.convert(*nodes[node_id.attrib["ref"]]))

    if not conv.all_elements_inside_boundary(way_nodes):
        continue

    # if conv.all_elements_outside_boundary(way_nodes):
    #     continue

    if way_nodes[0] == way_nodes[-1]: # closed loop
        svg.add_polygon(way_nodes, layer="whatever", stroke_width=0.2, opacity=0.5)
    else:
        svg.add_line(way_nodes, stroke_width=0.2, layer="streets")

print("processing streets finished.")

# --- BUILDINGS

for way in root.findall("./way/tag[@k='building']/.."):

    polygon = []

    for node_id in way.findall("./nd[@ref]"):
        polygon.append(conv.convert(*nodes[node_id.attrib["ref"]]))

    if not conv.all_elements_inside_boundary(polygon):
        continue

    # if conv.all_elements_outside_boundary(polygon):
    #     continue

    svg.add_polygon(polygon, layer="buildings", stroke_width=0.2, fill=[255, 0, 0], opacity=0.5)

print("processing buildings finished.")


# --- MARKER

img_tag = "<image xlink:href=\"{}\" x=\"{}\" y=\"{}\" height=\"{}\" width=\"{}\"/>"

viewport_size = conv.get_map_size()

markersize = 30

svg.add_raw_element(img_tag.format("marker_1.jpg", 0, 0, markersize, markersize), layer="marker")
svg.add_raw_element(img_tag.format("marker_2.jpg", 0, viewport_size[1]-markersize, markersize, markersize), layer="marker")
svg.add_raw_element(img_tag.format("marker_3.jpg", viewport_size[0]-markersize, 0, markersize, markersize), layer="marker")
svg.add_raw_element(img_tag.format("marker_4.jpg", viewport_size[0]-markersize, viewport_size[1]-markersize, markersize, markersize), layer="marker")

svg.save()
