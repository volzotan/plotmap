import xml.etree.ElementTree as ET  

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

MAP_SIZE        = 1000 # px

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
    def add_rectangle(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0, layer="default"):
        self.layers[layer]["rectangles"].append([*coords, strokewidth, stroke, opacity])

    def add_polygon(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0, layer="default"):
        self.layers[layer]["polygons"].append(coords)

    def add_line(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0, layer="default"):
        self.layers[layer]["lines"].append(coords)

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

                for p in layer["polygons"]:
                    out.write("<path d=\"")
                    out.write("M{} {} ".format(int(p[0][0]), int(p[0][1])))
                    for point in p[1:]:
                        out.write("L")
                        out.write(str(int(point[0])))
                        out.write(" ")
                        out.write(str(int(point[1])))
                        out.write(" ")
                    out.write("\" ")
                    out.write("stroke=\"rgb({},{},{})\" ".format(0, 0, 0))
                    out.write("fill=\"rgb({},{},{})\" ".format(125, 125, 125))
                    out.write("fill-opacity=\"{}\" />".format(0.5))

                for l in layer["lines"]:
                    out.write("<path d=\"")
                    out.write("M{} {} ".format(int(l[0][0]), int(l[0][1])))
                    for point in l[1:]:
                        out.write("L")
                        out.write(str(int(point[0])))
                        out.write(" ")
                        out.write(str(int(point[1])))
                        out.write(" ")
                    out.write("\" ")
                    out.write("stroke=\"rgb({},{},{})\" ".format(0, 0, 0))
                    out.write("fill=\"rgb({},{},{})\" ".format(0, 0, 0))
                    out.write("fill-opacity=\"{}\" ".format(0))
                    out.write("/>")

                out.write("</g>")

            out.write("</svg>")

        print("writing SVG in {0:.2f}s".format((datetime.now()-timer_start).microseconds/1000000))


class Converter(object):

    def __init__(self, map_up_left, map_down_right, map_size):

        # self.map_size_virtual = 1000.0 # size of the virtual map

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
        # self.map_size_y = map_size * (self.lon_diff / self.lat_diff)
        # self.map_size_y = map_size 

        print(self.map_size_x)
        print(self.map_size_y)

    # def _mercN(self, lat):

    #     return math.log( math.tan((math.pi / 4.0) + ((lat*math.pi / 180.0) / 2.0)) );

    # def convert_alt(self, lat, lon):

    #     x = ((lon - self.west) / self.lon_diff) * self.map_size_x
    #     y = ((self._mercN(self.north) - self._mercN(lat)) * self.map_size_y) / (self._mercN(self.north) - self._mercN(self.south))   # somethings wrong

    #     return (x, y)

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

    @staticmethod
    def get_bounding_box_in_latlon(center_point, width, height):
        map_up_left     = (center_point[0] + Converter._m_to_latlon(height/2), center_point[1] - Converter._m_to_latlon(width/2))
        map_down_right  = (center_point[0] - Converter._m_to_latlon(height/2), center_point[1] + Converter._m_to_latlon(width/2))

        return (map_up_left, map_down_right)

    @staticmethod
    def _m_to_latlon(m):
        return (m / 1.1) * 0.00001

    # https://stackoverflow.com/a/14457180
    # def convert(lat, lon):

    #     # get x value
    #     x = (lon+180) * (self.width_px/360.0)

    #     # convert from degrees to radians
    #     latRad = lat * math.PI / 180.0

    #     # get y value
    #     mercN = ln(tan((math.PI / 4.0) + (latRad/2.0)));

    #     x = ($x - $west) * $width/($east - $west)
    #     y = (($ymax - $y) / diff) * $height/($ymax - $ymin)

    #     y     = (self.height_px/2.0) - (self.width_px * mercN / (2.0 * math.PI))

    #     return (x, y)

bounding_box = Converter.get_bounding_box_in_latlon(MAP_CENTER, 1000, 500)
print(bounding_box)

MAP_UP_LEFT = bounding_box[0]
MAP_DOWN_RIGHT = bounding_box[1]

# MAP_UP_LEFT     = (50.9808, 11.3248)
# MAP_DOWN_RIGHT  = (50.9790, 11.3271)

# MAP_UP_LEFT     = (50.980192, 11.324840)
# MAP_DOWN_RIGHT  = (50.979645, 11.327615)

# square
# MAP_UP_LEFT     = (50.980467, 11.325000)
# MAP_DOWN_RIGHT  = (50.979248, 11.327623)

print((MAP_UP_LEFT, MAP_DOWN_RIGHT))

conv = Converter(MAP_UP_LEFT, MAP_DOWN_RIGHT, MAP_SIZE)
svg = SvgWriter("test.svg", conv.get_map_size())

svg.add_layer("meta")
svg.add_layer("buildings")
svg.add_layer("streets")
svg.add_layer("whatever")

xy1 = conv.convert(*MAP_UP_LEFT)
xy2 = conv.convert(*MAP_DOWN_RIGHT)
svg.add_rectangle([xy1, [xy2[0] - xy1[0], xy2[1] - xy1[1]]], strokewidth=2)

#tree = ET.parse("weimar.osm")  
tree = ET.parse("weimar_theaterplatz.osm")  
root = tree.getroot()

# parse document

nodes = {}
for node in root.findall("./node"):
    nodes[node.attrib["id"]] = [float(node.attrib["lat"]), float(node.attrib["lon"])]

print("indexed {} nodes".format(len(nodes.keys())))

# --- MISC

svg.add_circles([conv.convert(*MAP_CENTER)], layer="meta")

# --- BUILDINGS

for way in root.findall("./way/tag[@k='building']/.."):

    polygon = []

    for node_id in way.findall("./nd[@ref]"):
        polygon.append(conv.convert(*nodes[node_id.attrib["ref"]]))

    svg.add_polygon(polygon, layer="buildings")

print("processing buildings finished.")

# --- STREETS

for way in root.findall("./way"):

    if len(way.findall("./tag[@k='building']")):
        continue

    # print(ET.tostring(way).decode())

    way_nodes = []

    for node_id in way.findall("./nd[@ref]"):
        way_nodes.append(conv.convert(*nodes[node_id.attrib["ref"]]))

    if way_nodes[0] == way_nodes[-1]: # closed loop
        svg.add_polygon(way_nodes, layer="whatever")
    else:
        svg.add_line(way_nodes, layer="streets")

print("processing streets finished.")

svg.save()
