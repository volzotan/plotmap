import datetime

from loguru import logger
from sqlalchemy import create_engine

from core import maptools
from layers import bathymetry, contour
from lineworld.core.svgwriter import SvgWriter
from lineworld.layers import coastlines, grid

from shapely.geometry import MultiPolygon

import cProfile as profile

from lineworld.util.scales import Colorscale

if __name__ == "__main__":

    pr = profile.Profile()
    pr.disable()

    # engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)
    engine = create_engine("postgresql+psycopg://localhost:5432/lineworld", echo=True)

    # logging.basicConfig()
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    document_info = maptools.DocumentInfo()

    layer_bathymetry = bathymetry.Bathymetry("Bathymetry", [0, -11_000], 15, engine)
    layer_contour = contour.Contour("Contour", [0, 500, 2000, 9000], 24, engine)
    layer_coastlines = coastlines.Coastlines("Coastlines", engine)
    layer_grid_bathymetry = grid.GridBathymetry("Grid Bathymetry", engine)
    layer_grid_labels = grid.GridLabels("Grid Labels", engine)

    active_layers = [
        layer_bathymetry,
        layer_contour,
        layer_coastlines,
        layer_grid_bathymetry,
        layer_grid_labels
    ]

    for layer in active_layers:

        # l.extract()

        timer_start = datetime.datetime.now()
        polygons = layer.transform_to_world()
        logger.debug("transform in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        timer_start = datetime.datetime.now()
        layer.load(polygons)
        logger.debug("load in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        timer_start = datetime.datetime.now()
        polygons = layer.transform_to_map(document_info)
        logger.debug("project in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        timer_start = datetime.datetime.now()
        layer.load(polygons)
        logger.debug("load in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        timer_start = datetime.datetime.now()
        lines = layer.transform_to_lines(document_info)
        logger.debug("draw in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        timer_start = datetime.datetime.now()
        layer.load(lines)
        logger.debug("load in {:5.2f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

        # pr.enable()
        # pr.disable()

    # pr.dump_stats('profile.pstat')

    exclude = MultiPolygon()

    draw_grid_labels, exclude = layer_grid_labels.out(exclude, document_info)
    draw_coastlines, exclude = layer_coastlines.out(exclude, document_info)
    draw_contour, exclude = layer_contour.out(exclude, document_info)
    _, exclude = layer_grid_bathymetry.out(exclude, document_info)
    # draw_bathymetry, exclude = layer_bathymetry.out(exclude, document_info)

    svg = SvgWriter("test.svg", [document_info.width, document_info.height])
    svg.background_color = "white"

    options_bathymetry = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0",
        "fill-opacity": "0.9"
    }

    scale = Colorscale([0, layer_bathymetry.NUM_ELEVATION_LINES])
    for i in range(layer_bathymetry.NUM_ELEVATION_LINES):
        polys, _ = layer_bathymetry.out_polygons(exclude, document_info, select_elevation_level=i)
        color = "rgb({},{},{})".format(*[int(x * 255) for x in scale.get_color(layer_bathymetry.NUM_ELEVATION_LINES-1-i)])
        svg.add(f"bathymetry_polys_{i}", polys, options=options_bathymetry | {"fill": color})

    # options_contour = {
    #     "fill": "none",
    #     "stroke": "black",
    #     "stroke-width": "0",
    #     "fill-opacity": "0.5"
    # }
    #
    # scale = Colorscale([0, layer_contour.NUM_ELEVATION_LINES])
    # for i in range(layer_contour.NUM_ELEVATION_LINES):
    #     polys, _ = layer_contour.out_polygons([], document_info, select_elevation_level=i)
    #     color = "rgb({},{},{})".format(*[int(x * 255) for x in scale.get_color(i)])
    #     svg.add(f"contour_polys_{i}", polys, options=options_contour | {"fill": color})

    options_bathymetry = {
        "fill": "none",
        "stroke": "blue",
        "stroke-width": "0.5",
        "fill-opacity": "0.1"
    }

    options_contour = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.4",
    }

    options_coastlines = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.5",
    }

    options_grid = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "1.0",
    }

    svg.add_style("coastlines", options_coastlines)

    # svg.add("bathymetry", draw_bathymetry, options=options_bathymetry)
    svg.add("contour", draw_contour, options=options_contour)
    svg.add("coastlines", draw_coastlines, options=options_coastlines)
    svg.add("grid_labels", draw_grid_labels, options=options_grid)



    svg.write()

