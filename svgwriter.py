import math
from datetime import datetime

class SvgWriter(object):

    def __init__(self, filename, dimensions=None, image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename   = filename
        self.dimensions = dimensions
        self.image      = image

        self.hatchings          = {}
        self.hatching_options   = {}

        self.layers     = {}
        self.add_layer("default")

    def add_layer(self, layer_id):
        self.layers[layer_id] = {}
        self.layers[layer_id]["hexagons"]   = []
        self.layers[layer_id]["circles"]    = []
        self.layers[layer_id]["rectangles"] = []
        self.layers[layer_id]["polygons"]   = []
        self.layers[layer_id]["lines"]      = []
        self.layers[layer_id]["poly_lines"] = []
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

    # coords: [[x1, y1], [x2, x2]]
    def add_line(self, coords, stroke_width=1, stroke=[255, 0, 0], stroke_opacity=1.0, layer="default"):
        options = {}
        options["stroke-width"]     = stroke_width
        options["stroke"]           = stroke
        options["stroke-opacity"]   = stroke_opacity
        self.layers[layer]["lines"].append((coords, options))

    def add_lines(self, coords, **kwargs):
        for coord in coords:
            self.add_line(coord, **kwargs)

    def add_polygon(self, coords, stroke_width=1, stroke=[0, 0, 0], fill=[120, 120, 120], opacity=1.0, layer="default", hatching=None):
        options = {}
        options["stroke-width"]     = stroke_width
        options["stroke"]           = stroke
        options["fill"]             = fill
        options["opacity"]          = opacity
        self.layers[layer]["polygons"].append((coords, options))

        if hatching is not None:
            self._add_hatching_for_polygon(coords, hatching)

    def add_poly_line(self, coords, stroke_width=1, stroke=[255, 0, 0], stroke_opacity=1.0, layer="default"):
        options = {}
        options["stroke-width"]     = stroke_width
        options["stroke"]           = stroke
        options["stroke-opacity"]   = stroke_opacity
        self.layers[layer]["poly_lines"].append((coords, options))

    def add_raw_element(self, text, layer="default"):
        self.layers[layer]["raw"].append(text)

    @staticmethod
    def _line_intersection(line1, line2): 

        A = line1[0]
        B = line1[1]
        C = line2[0]
        D = line2[1]

        Bx_Ax = B[0] - A[0] 
        By_Ay = B[1] - A[1] 
        Dx_Cx = D[0] - C[0] 
        Dy_Cy = D[1] - C[1] 
        
        determinant = (-Dx_Cx * By_Ay + Bx_Ax * Dy_Cy) 
        
        if abs(determinant) < 1e-20: 
            return None 

        s = (-By_Ay * (A[0] - C[0]) + Bx_Ax * (A[1] - C[1])) / determinant 
        t = ( Dx_Cx * (A[1] - C[1]) - Dy_Cy * (A[0] - C[0])) / determinant 

        if s >= 0 and s <= 1 and t >= 0 and t <= 1: 
            return (A[0] + (t * Bx_Ax), A[1] + (t * By_Ay)) 

        return None

    # rotation: 45-90 degrees
    # TODO: 0-45 degrees
    def add_hatching(self, name, rotation=45, distance=2, stroke_width=0.2):
        self.hatchings[name] = []
        self.hatching_options[name] = {}
        self.hatching_options[name]["stroke_width"] = stroke_width

        rot_rad = rotation * math.pi / 180.0

        num_lines = (self.dimensions[0] + self.dimensions[1])/float(distance)

        north = [[0, 0], [self.dimensions[0], 0]]
        south = [[0, self.dimensions[1]], [self.dimensions[0], self.dimensions[1]]]
        west  = [[0, 0], [0, self.dimensions[1]]]
        east  = [[self.dimensions[0], 0], [self.dimensions[0], self.dimensions[1]]]

        for i in range(0, int(num_lines)):
            y = i * distance
            x = y * math.tan(rot_rad)

            hatching_line = [[0, y], [x, 0]]
            cropped_line = []

            north_intersect = SvgWriter._line_intersection(hatching_line, north)
            south_intersect = SvgWriter._line_intersection(hatching_line, south)
            west_intersect = SvgWriter._line_intersection(hatching_line, west)
            east_intersect = SvgWriter._line_intersection(hatching_line, east)

            if west_intersect is not None:
                cropped_line.append(west_intersect)

            if south_intersect is not None:
                cropped_line.append(south_intersect)

            if north_intersect is not None:
                cropped_line.append(north_intersect)

            if east_intersect is not None:
                cropped_line.append(east_intersect)

            if len(cropped_line) == 2:
                self.hatchings[name].append(cropped_line)
                # self.add_line(hatching_line, stroke_width=stroke_width)

    def _polygon_to_lines(self, polygon_coords):
        lines = []
        for i in range(0, len(polygon_coords)-1):
            cur = polygon_coords[i]
            nxt = polygon_coords[i+1]
            lines.append([[cur[0], cur[1]], [nxt[0], nxt[1]]])

        return lines

    def _add_hatching_for_polygon(self, coords, hatching_name):
        lines = self._polygon_to_lines(coords)
        hatchlines_in_poly = []

        if (len(self.hatchings[hatching_name])) <= 0:
            raise Exception("missing hatching: {}".format(hatching_name))

        for hline in self.hatchings[hatching_name]:
            intersection_points = []
            for line in lines:
                intersection_point = self._line_intersection(hline, line)

                if intersection_point is not None:
                    intersection_points.append(intersection_point)

            if len(intersection_points) >= 2:
                hatchlines_in_poly.append([intersection_points[0], intersection_points[1]])
            if len(intersection_points) >= 4:
                hatchlines_in_poly.append([intersection_points[2], intersection_points[3]])

            # TODO ...

        for hip in hatchlines_in_poly:
            self.add_line(hip, **self.hatching_options[hatching_name])


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

                # if layerid == "default":
                #     continue

                layer = self.layers[layerid]
                
                out.write("<g inkscape:groupmode=\"layer\" id=\"{0}\" inkscape:label=\"{0}\">".format(layerid))

                for c in layer["circles"]:
                    out.write("<circle cx=\"{}\" cy=\"{}\" fill=\"rgb({},{},{})\" r=\"{}\" />".format(c[0][0], c[0][1], c[2][0], c[2][1], c[2][2], c[1]))

                for r in layer["rectangles"]:
                    out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], *r[1], r[2], *r[3], r[4]))

                for line in layer["lines"]:
                    l = line[0]
                    options = line[1]
                    out.write("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke-width=\"{}\" stroke=\"rgb(1.0, 1.0, 1.0)\" />".format(*l[0], *l[1], options["stroke-width"]))

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

                for line in layer["poly_lines"]:
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

        print("writing SVG in {0:.2f}s".format((datetime.now()-timer_start).total_seconds()))