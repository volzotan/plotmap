from pathlib import Path
import cv2

import numpy as np
from shapely import LineString, Polygon, MultiPolygon

from lineworld.core.hatching import HatchingOptions, HatchingDirection, create_hatching
from lineworld.core.maptools import DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds
from lineworld.util.geometrytools import unpack_multipolygon

INPUT_FILE = Path("experiments/hatching_test_pattern.png")

img = cv2.imread(str(INPUT_FILE), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, [1000, 1000])

hatchings = []

LEVELS = 10
DISTANCES = [2.5 + x*0.9 for x in range(LEVELS)]
BOUNDS = get_elevation_bounds([0, 255], LEVELS)


def standard_hatching():
    hatchings = []

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(img, *BOUNDS[i], False)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        hatching_options = HatchingOptions()
        hatching_options.distance = DISTANCES[i]
        # hatching_options.direction = HatchingDirection.ANGLE_135 if i % 2 == 0 else HatchingDirection.ANGLE_45
        hatching_options.direction = HatchingDirection.ANGLE_45

        for p in polygons:
            hatchings += [create_hatching(p, None, hatching_options)]
            # hatchings += create_hatching_2(p, None, hatching_options)

    return hatchings

def standard_hatching2():

    def create_hatching2(g: Polygon | MultiPolygon, bbox: list[float], options: HatchingOptions) -> list[LineString]:
        lines = []

        if g.is_empty:
            return lines

        polys = unpack_multipolygon(g)

        for p in polys:
            lines.append(LineString(p.exterior))

            for hole in p.interiors:
                lines.append((LineString(hole)))

            lines += create_hatching2(p.buffer(-options.distance), None, options)

        return lines

    hatchings = []

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(img, *BOUNDS[i], False)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        hatching_options = HatchingOptions()
        hatching_options.distance = DISTANCES[i]
        # hatching_options.direction = HatchingDirection.ANGLE_135 if i % 2 == 0 else HatchingDirection.ANGLE_45
        hatching_options.direction = HatchingDirection.ANGLE_45

        for p in polygons:
            hatchings += create_hatching2(p, None, hatching_options)

    return hatchings


hatchings = standard_hatching()
# hatchings = standard_hatching2()

doc = DocumentInfo()
doc.width = 1000
doc.height = 1000

svg = SvgWriter("hatching.svg", [doc.width, doc.height])
svg.background_color = "white"

options = {
    "fill": "none",
    "stroke": "black",
    "stroke-width": "2.0",
}

svg.add("contour", hatchings, options=options)

svg.write()