import xml.etree.ElementTree as ET  
import sys
import math

MAP_UP_LEFT     = (50.9808, 11.3248)
MAP_DOWN_RIGHT  = (50.9790, 11.3271)
MAP_SIZE        = 1000 # px

class SvgWriter(object):

    def __init__(self, filename, dimensions=None, image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename = filename
        self.dimensions = dimensions
        self.image = image

        self.hexagons = []
        self.circles = []
        self.rectangles = []
        self.polygons = []

    def add_hexagons(self, hexagons, fills):
        for i in range(0, len(hexagons)):
            self.hexagons.append([
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                [fills[i][0]*255, fills[i][1]*255, fills[i][2]*255, fills[i][3]]
            ]) 

    def add_circles(self, circles, radius=3, fill=[255, 0, 0]):
        for item in circles:
            self.circles.append([item, radius, fill])

    # coords: [[x, y], [width, height]]
    def add_rectangle(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0):
        self.rectangles.append([*coords, strokewidth, stroke, opacity])

    def add_polygon(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0):
        self.polygons.append(coords)

    def save(self):
        with open(self.filename, "w") as out:

            out.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?>")
            out.write("<?xml-stylesheet href=\"style.css\" type=\"text/css\" title=\"main_stylesheet\" alternate=\"no\" media=\"screen\" ?>")
            
            if self.dimensions is not None:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" width=\"{}px\" height=\"{}px\" ".format(self.dimensions[0], self.dimensions[1]))
            else:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" ")
            out.write("xmlns=\"http://www.w3.org/2000/svg\" xmlns:ev=\"http://www.w3.org/2001/xml-events\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" >")
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

            for r in self.rectangles:
                out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], *r[1], r[2], *r[3], r[4]))

            for p in self.polygons:
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

            out.write("</svg>")


class Converter(object):

    def __init__(self, map_up_left, map_down_right, map_size):
        self.north = map_up_left[0]
        self.south = map_down_right[0]
        self.west = map_up_left[1]
        self.east = map_down_right[1]

        # northern hemisphere, ...

        self.lat_diff = self.north - self.south
        self.lon_diff = self.east - self.west

        self.map_size_x = map_size
        self.map_size_y = map_size * (self.lon_diff / self.lat_diff)

    def _mercN(self, lat):

        return math.log( math.tan((math.pi / 4.0) + ((lat*math.pi / 180.0) / 2.0)) );

    def convert(self, lat, lon):

        x = ((lon - self.west) / self.lon_diff) * self.map_size_x
        y = ((self._mercN(self.north) - self._mercN(lat)) * self.map_size_y) / (self._mercN(self.north) - self._mercN(self.south))

        return (x, y)

    def get_map_size(self):
        return (self.map_size_x, self.map_size_y)

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


conv = Converter(MAP_UP_LEFT, MAP_DOWN_RIGHT, MAP_SIZE)
svg = SvgWriter("test.svg", conv.get_map_size())

xy1 = conv.convert(*MAP_UP_LEFT)
xy2 = conv.convert(*MAP_DOWN_RIGHT)
svg.add_rectangle([xy1, [xy2[0] - xy1[0], xy2[1] - xy1[1]]], strokewidth=2)

tree = ET.parse("weimar_theaterplatz.osm")  
root = tree.getroot()

for way in root.findall("./way/tag[@k='building']/.."):
    # print(ET.tostring(way).decode())
    # print("---------------------------------------")

    polygon = []

    for node_id in way.findall("./nd[@ref]"):
        # print(node_id.attrib["ref"])   
        for node in root.findall("./node[@id='{}']".format(node_id.attrib["ref"])):
            lat = node.attrib["lat"]
            lon = node.attrib["lon"]

            # print(ET.tostring(node).decode())
            # print("{}, {}".format(lat, lon))
            # print("{}, {}".format(*conv.convert(float(lat), float(lon))))

            polygon.append(conv.convert(float(lat), float(lon)))

    svg.add_polygon(polygon)

    # break

svg.save()


# for node in root.findall("./node/tag[@k='building']/.."):
#     print(ET.tostring(node).decode())
#     print("---------------------------------------")
#     sys.exit(0)