import argparse
import cProfile as profile
import datetime

import numpy as np
from loguru import logger
from shapely.geometry import MultiPolygon
from sqlalchemy import create_engine

import lineworld
from core import map
from layers import contour
from lineworld.core.map import DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.layers import (
    coastlines,
    grid,
    labels,
    cities,
    bflowlines,
    bathymetry,
    cities,
    contour2,
    meta,
    oceancurrents,
)
from lineworld.util.export import convert_svg_to_png


def run() -> None:
    timer_total_runtime = datetime.datetime.now()

    pr = profile.Profile()
    pr.disable()

    # parser = argparse.ArgumentParser(description="...")
    # parser.add_argument("--set", metavar="KEY=VALUE", nargs='+')
    # args = vars(parser.parse_args())
    #
    # print(args["set"])
    # exit()

    config = lineworld.get_config()
    engine = create_engine(config["main"]["db_connection"])  # , echo=True)
    document_info = map.DocumentInfo(config)

    layer_grid_bathymetry = grid.GridBathymetry("GridBathymetry", engine, config)
    layer_grid_labels = grid.GridLabels("GridLabels", engine, config)

    layer_bathymetry = bflowlines.BathymetryFlowlines(
        "BathymetryFlowlines", engine, config, tile_boundaries=layer_grid_bathymetry.get_polygons(document_info)
    )
    layer_bathymetry2 = bathymetry.Bathymetry("Bathymetry", engine, config)

    layer_oceancurrents = oceancurrents.OceanCurrents(
        "OceanCurrents", engine, config, tile_boundaries=layer_grid_bathymetry.get_polygons(document_info)
    )

    layer_contour = contour.Contour("Contour", engine, config)
    layer_contour2 = contour2.Contour2("Contour2", engine, config)

    layer_coastlines = coastlines.Coastlines("Coastlines", engine, config)

    layer_cities_labels = cities.CitiesLabels("CitiesLabels", engine, config)
    layer_cities_circles = cities.CitiesCircles("CitiesCircles", engine, config)

    layer_labels = labels.Labels("Labels", engine, config)
    layer_meta = meta.Meta("Meta", engine, config)

    compute_layers = [
        # layer_oceancurrents,
        # layer_bathymetry,
        # # layer_bathymetry2,
        # layer_contour2,
        # layer_coastlines,
        # layer_cities_labels,
        # layer_cities_circles,
        # layer_labels,
        # layer_grid_bathymetry,
        # layer_grid_labels,
        # layer_meta,
    ]

    for layer in compute_layers:
        layer.extract()

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

    visible_layers = [
        # layer_meta,
        layer_cities_circles,
        layer_cities_labels,
        layer_grid_labels,
        layer_labels,
        layer_coastlines,
        layer_contour2,
        # # layer_contour,
        layer_grid_bathymetry,
        layer_bathymetry,
        # layer_bathymetry2,
        # layer_oceancurrents,
    ]

    exclude = []
    draw_objects = {}

    for layer in visible_layers:
        timer_start = datetime.datetime.now()
        draw, exclude = layer.out(exclude, document_info)
        draw_objects[layer.layer_id] = draw
        logger.debug(
            "{} layer subtraction in {:5.2f}s".format(
                layer.layer_id, (datetime.datetime.now() - timer_start).total_seconds()
            )
        )

    svg_filename = config.get("name", "output")
    if not svg_filename.endswith(".svg"):
        svg_filename += ".svg"

    svg = SvgWriter(svg_filename, document_info.get_document_size())
    # svg.background_color = config.get("svg_background_color", "#333333")

    # options_bathymetry = {
    #     "fill": "none",
    #     "stroke": "black",
    #     "stroke-width": "0",
    #     "fill-opacity": "0.9"
    # }
    #
    # scale = Colorscale([0, layer_bathymetry.NUM_ELEVATION_LINES])
    # for i in range(layer_bathymetry.NUM_ELEVATION_LINES):
    #     polys, _ = layer_bathymetry.out_polygons(exclude, document_info, select_elevation_level=i)
    #     color = "rgb({},{},{})".format(*[int(x * 255) for x in scale.get_color(layer_bathymetry.NUM_ELEVATION_LINES-1-i)])
    #     svg.add(f"bathymetry_polys_{i}", polys, options=options_bathymetry | {"fill": color})

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

    layer_styles = {}

    layer_styles[layer_oceancurrents.layer_id] = {
        "fill": "none",
        "stroke": "blue",
        "stroke-width": "0.40",
        "fill-opacity": "0.1",
    }

    layer_styles[layer_bathymetry.layer_id] = {
        "fill": "none",
        "stroke": "blue",
        "stroke-width": "0.40",
        "fill-opacity": "0.1",
    }

    layer_styles[layer_contour.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.40",
    }

    layer_styles[layer_coastlines.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.5",
    }

    layer_styles[layer_grid_labels.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.4",
    }

    layer_styles[layer_labels.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.4",
    }

    layer_styles[layer_cities_labels.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.4",
    }

    layer_styles[layer_cities_circles.layer_id] = {
        "fill": "none",
        "stroke": "red",
        "stroke-width": "0.4",
    }

    layer_styles[layer_meta.layer_id] = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "0.4",
    }

    layer_styles[layer_bathymetry2.layer_id] = layer_styles[layer_bathymetry.layer_id]
    layer_styles[layer_contour2.layer_id] = layer_styles[layer_contour.layer_id]

    for k, v in layer_styles.items():
        svg.add_style(k, v)

    for k, v in draw_objects.items():
        svg.add(k, v)  # , options=layer_styles.get(k.lower(), {}))

    # tanaka_style = {
    #     "fill": "none",
    #     "stroke-width": "0.40",
    #     "fill-opacity": "1.0",
    # }
    # svg.add(
    #     "Contours2_High",
    #     layer_contour2.out_tanaka(exclude, document_info, highlights=True)[0],
    #     {**tanaka_style, "stroke": "#999999"},
    # )
    # svg.add(
    #     "Contours2_Low",
    #     layer_contour2.out_tanaka(exclude, document_info, highlights=False)[0],
    #     {**tanaka_style, "stroke": "black"},
    # )

    svg.write()
    try:
        convert_svg_to_png(svg, svg.dimensions[0] * 10)
    except Exception as e:
        logger.warning(f"SVG to PNG conversion failed: {e}")

    logger.info(f"total time: {(datetime.datetime.now() - timer_total_runtime).total_seconds():5.2f}s")


if __name__ == "__main__":
    run()
