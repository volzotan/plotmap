import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine

from core import maptools
from core.svgwriter import SvgWriter
from layers import contour
import subprocess

if __name__ == "__main__":

    engine = create_engine("postgresql+psycopg://localhost:5432/lineworld")

    document_info = maptools.DocumentInfo()
    document_info.width = 3000
    document_info.height = 2000

    parameters = [
        ["contour_1", [0, 9000]],
        ["contour_2", [0, 800, 3000, 9000]],
        ["contour_3", [0, 500, 2000, 9000]],
        ["contour_4", [0, 1000, 9000]],
        # ["1", [0, -12_000]],
    ]

    for param_name, param_value in parameters:
        timer_start = datetime.datetime.now()

        options_bathymetry = {
            "fill": "none",
            "stroke": "blue",
            "stroke-width": "0.5",
            "fill-opacity": "0.1"
        }

        options_contour = {
            "fill": "none",
            "stroke": "black",
            "stroke-width": "0.5",
        }

        svg_filename = Path(f"comparison_{param_name}.svg")
        svg = SvgWriter(svg_filename, document_info)

        # layer_bathymetry = bathymetry.Bathymetry("Bathymetry", param_value, 15, engine)
        # layer_bathymetry.extract()
        # polygons = layer_bathymetry.transform()
        # layer_bathymetry.load(polygons)
        # layer_bathymetry.convert(document_info)
        # draw_bathymetry, exclude = layer_bathymetry.out([], document_info)
        # svg.add("bathymetry", draw_bathymetry, options=options_bathymetry)

        layer_contour = contour.Contour("Contour", param_value, 24, engine)
        # layer_contour.extract()
        polygons = layer_contour.transform()
        layer_contour.load(polygons)
        layer_contour.convert(document_info)
        draw_contour, exclude = layer_contour.out([], document_info)
        svg.add("contour", draw_contour, options=options_contour)

        svg.write()

        logger.debug("[{}] running took {:5.2f}s".format(
            param_name,
            (datetime.datetime.now() - timer_start).total_seconds()
        ))

        png_filename = Path(svg_filename, ".png")
        # subprocess.run(f"inkscape --export -width=3000 --export -type=png --export -filename=\"{png_filename}\" \"{svg_filename}\"", shell=True, check=True)
        subprocess.run(f"/Applications/Inkscape.app/Contents/MacOS/inkscape --export-type=\"png\" {svg_filename}", shell=True, check=True)