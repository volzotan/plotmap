from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import Any

from loguru import logger
from shapely import Geometry, MultiLineString, LineString, Polygon


class SvgWriter():
    type SvgOptions = dict[str, str | int | float]

    filename: Path
    dimensions: list[int | float]
    image: str | None = None
    background_color: str | None = None
    offset: list[int | float] = [0, 0]

    layers: dict[str, list[tuple[Geometry, SvgOptions]]] = {}

    styles: dict[str, SvgOptions] = {}

    def __init__(self, filename: Path | str, dimensions: list[float]):
        self.filename = Path(filename)
        self.dimensions = dimensions

    def add_style(self, layer: str, options: SvgOptions):
        if layer not in self.styles:
            self.styles[layer] = {}

        self.styles[layer] = {**self.styles[layer], **options}

    def add(self, layer: str, geom: Geometry | list[Geometry], options: SvgOptions = {}) -> None:

        logger.debug(f"layer {layer}: adding {len(geom) if type(geom) is list else 1} object(s)")

        if layer not in self.layers:
            self.layers[layer] = []

        if isinstance(geom, list):
            for i in range(len(geom)):
                if geom[i].is_empty:
                    continue
                self.layers[layer].append((geom[i], options))
        else:
            self.layers[layer].append((geom, options))

    def write_layer(self, out: TextIOWrapper, layer_name: str):
        layer = self.layers[layer_name]
        out.write(f"<g inkscape:groupmode=\"layer\" id=\"{layer_name}\" inkscape:label=\"{layer_name}\">")
        for geom, options in layer:
            match geom:

                # case Point():
                #     pass

                case Polygon():
                    self._write_polygon(out, geom, options)

                case LineString():
                    self._write_lineString(out, geom, options)

                case MultiLineString():
                    for ls in geom.geoms:
                        self._write_lineString(out, ls, options)

                case _:
                    logger.warning(f"unknown geometry object in layer {layer_name}: {type(geom)}")

        out.write("</g>")
        out.write("\n")

    def write(self):

        with open(self.filename, "w") as out:

            out.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n")
            out.write(
                "<?xml-stylesheet href=\"style.css\" type=\"text/css\" title=\"main_stylesheet\" alternate=\"no\" media=\"screen\" ?>\n")

            if self.dimensions is not None:
                out.write(
                    f"<svg baseProfile=\"tiny\" version=\"1.2\" width=\"{self.dimensions[0]}px\" height=\"{self.dimensions[1]}px\" ")
            else:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" ")
            out.write("xmlns=\"http://www.w3.org/2000/svg\" ")
            out.write("xmlns:ev=\"http://www.w3.org/2001/xml-events\" ")
            out.write("xmlns:xlink=\"http://www.w3.org/1999/xlink\" ")
            out.write("xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\" ")
            out.write(">\n")
            out.write("<defs />\n")
            out.write("\n")

            out.write("<style>\n")

            out.write("path, line { stroke-linecap: round; stroke-linejoin: round; }\n")

            if self.background_color is not None:
                out.write(f"svg {{ background-color: {self.background_color}; }}\n")

            for style_selector, style_attributes in self.styles.items():
                out.write(f"#{style_selector} {{\n")
                for k, v in style_attributes.items():
                    out.write(f"\t{k}: {v};\n")
                out.write("}\n")

            out.write("</style>\n")

            if self.image is not None:
                out.write(f"<image x=\"0\" y=\"0\" xlink:href=\"{self.image}\" />")

            out.write("\n")

            for layerid in self.layers.keys():
                self.write_layer(out, layerid)

            out.write("</svg>")

    def _write_polygon(self, out: StringIO, p: Polygon, options: SvgOptions) -> None:
        self._write_path(out, p.exterior.coords, options, holes=[hole.coords for hole in p.interiors])

    def _write_lineString(self, out: StringIO, l: LineString, options: SvgOptions) -> None:
        self._write_path(out, l.coords, options, close=False)

    def _write_path(self, out: StringIO, p: Any, options: SvgOptions, holes: list[Any] = [],
                    close: bool = True) -> None:

        out.write("<path d=\"")

        x = float(p[0][0] - self.offset[0])
        y = float(p[0][1] - self.offset[1])
        out.write(f"M{x:.2f} {y:.2f} ")

        for point in p[1:]:
            x = float(point[0] - self.offset[0])
            y = float(point[1] - self.offset[1])
            out.write(f"L{x:.2f} {y:.2f} ")

        for h in holes:
            out.write("Z ")

            x = float(h[0][0] - self.offset[0])
            y = float(h[0][1] - self.offset[1])
            out.write(f"M{x:.2f} {y:.2f} ")

            for point in h[1:]:
                x = float(point[0] - self.offset[0])
                y = float(point[1] - self.offset[1])
                out.write(f"L{x:.2f} {y:.2f} ")

        if close:
            out.write("Z\"")
        else:
            out.write("\"")

        for k, v in options.items():
            out.write(f" {k}=\"{v}\"")

        out.write("/>")
        out.write("\n")
