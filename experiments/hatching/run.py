from pathlib import Path

import cv2
import numpy as np
import shapely
from rasterio.features import rasterize
from shapely import LineString, Polygon, MultiPolygon, MultiLineString
from shapelysmooth import taubin_smooth

from experiments.hatching.flowlines import FlowlineHatcher, FlowlineHatcherConfig
from experiments.hatching.slope import get_slope
from lineworld.core.hatching import HatchingOptions, HatchingDirection, create_hatching
from lineworld.core.maptools import DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds
from lineworld.util.geometrytools import unpack_multipolygon

from loguru import logger

MIN_RING_LENGTH = 50
POST_SMOOTHING_SIMPLIFY_TOLERANCE = 0.5

# INPUT_FILE = Path("experiments/hatching/data/hatching_dem.tif")
INPUT_FILE = Path("experiments/hatching/data/gebco_crop.tif")
# INPUT_FILE = Path("experiments/hatching/data/slope_test_3.tif")
# INPUT_FILE = Path("experiments/hatching/data/slope_test_5.tif")

OUTPUT_PATH = Path("experiments/hatching/output")

LEVELS = 20
DISTANCES = [3.0 + x * 0.9 for x in range(LEVELS)]
# BOUNDS = get_elevation_bounds([0, 20], LEVELS)


def read_data(input_path: Path) -> np.ndarray:
    data = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    # data = cv2.resize(img, [30, 30])

    # data = np.flipud(data)
    # data = (data * 120/20).astype(np.int8)
    # data = np.rot90(data)

    return data


def standard_hatching(data: np.ndarray, **kwargs) -> list[MultiLineString | LineString]:
    output = []
    bounds = get_elevation_bounds([np.min(data), np.max(data)], LEVELS)

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(data, *bounds[i], False)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        hatching_options = HatchingOptions()
        hatching_options.distance = DISTANCES[i]
        # hatching_options.direction = HatchingDirection.ANGLE_135 if i % 2 == 0 else HatchingDirection.ANGLE_45
        hatching_options.direction = HatchingDirection.ANGLE_45

        for p in polygons:
            output += [create_hatching(p, None, hatching_options)]

    return output


def standard_hatching_concentric(data: np.ndarray, **kwargs) -> list[MultiLineString | LineString]:
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

    output = []
    bounds = get_elevation_bounds([np.min(data), np.max(data)], LEVELS)

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(data, *bounds[i], False)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        hatching_options = HatchingOptions()
        hatching_options.distance = DISTANCES[i]
        # hatching_options.direction = HatchingDirection.ANGLE_135 if i % 2 == 0 else HatchingDirection.ANGLE_45
        # hatching_options.direction = HatchingDirection.ANGLE_45

        for p in polygons:
            output += create_hatching2(p, None, hatching_options)

    return output

def standard_hatching_slope_orientation(data: np.ndarray, angles: np.ndarray, **kwargs) -> list[MultiLineString | LineString]:
    output = []
    bounds = get_elevation_bounds([np.min(data), np.max(data)], LEVELS)

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(data, *bounds[i], False)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        for p in polygons:

            # mask = rasterize([p.buffer(-10)], out_shape=angles.shape)
            mask = rasterize([p], out_shape=angles.shape)

            # angles_debug = angles*(255/np.max(angles))
            # angles_debug[mask <= 0] = 0
            # cv2.imwrite("test.png", angles_debug)

            angle = np.degrees(np.mean(angles[mask > 0]))

            hatching_options = HatchingOptions()
            hatching_options.distance = DISTANCES[i]
            hatching_options.angle = angle

            output += [create_hatching(p, None, hatching_options)]

    return output


def _cut_linestring(ls: LineString) -> np.array:
    """
    returns NumPy array [x1, y1, x2, y2]
    """

    coordinate_pairs = np.zeros([len(ls.coords)-1, 4], dtype=float)

    coordinate_pairs[:, 0] = ls.xy[0][:-1]
    coordinate_pairs[:, 1] = ls.xy[1][:-1]
    coordinate_pairs[:, 2] = ls.xy[0][1:]
    coordinate_pairs[:, 3] = ls.xy[1][1:]

    return coordinate_pairs

def illuminated_contours(data: np.ndarray, **kwargs) -> list[list[MultiLineString | LineString]]:
    """
    correct results if (and only if) bounds are supplied in the right order, from lower to higher, ie.
    BOUNDS = get_elevation_bounds([-20, 0], LEVELS)
    """


    angle = 135
    width = 90

    all_ls = []
    output_bright = []
    output_dark = []

    bounds = get_elevation_bounds([np.min(data), np.max(data)], LEVELS)

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(data, *bounds[i], True)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(shapely.segmentize(g, 10))

        # smoothing
        polygons = [taubin_smooth(x, steps=10, factor=0.7, mu=-0.2) for x in polygons]
        polygons = [shapely.simplify(x, POST_SMOOTHING_SIMPLIFY_TOLERANCE) for x in polygons]

        for p in polygons:

            # area filtering
            # if p.area < 100.0:
            #     continue

            if p.exterior.length > MIN_RING_LENGTH:
                all_ls.append(p.exterior)

            for hole in p.interiors:
                if hole.length > MIN_RING_LENGTH:
                    all_ls.append(hole)
            # all_ls += p.interiors

    # cut linestrings to single lines
    for ls in all_ls:
        lines = _cut_linestring(ls)

        # compute orientation of lines
        theta = np.degrees(np.arctan2((lines[:, 3] - lines[:, 1]), (lines[:, 2] - lines[:, 0])))

        bright_mask = (theta > (angle - width)) & (theta < (angle + width))

        for line in lines[bright_mask]:
            output_bright.append(LineString([line[:2], line[2:]]))

        for line in lines[~bright_mask]:
            output_dark.append(LineString([line[:2], line[2:]]))


    # detect if falling or rising slope works implicit due to reversed hole coordinate order

    # reassemble connected lines of same color to linestrings

    return [output_bright, output_dark]


def flowline_hatching(data: np.ndarray, **kwargs) -> list[MultiLineString | LineString]:

    c = FlowlineHatcherConfig()

    density_data = data

    density_normalized = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))
    density = np.full(density_data.shape, c.LINE_DISTANCE[0], dtype=float) + (
            density_normalized * (c.LINE_DISTANCE[1] - c.LINE_DISTANCE[0]))

    X, Y, dX, dY, angles, inclination = get_slope(data, 10)

    hatcher = FlowlineHatcher(
        shapely.box(0, 0, data.shape[1], data.shape[0]),
        data, angles, inclination, density, c
    )

    linestrings = hatcher.hatch()

    return linestrings

if __name__ == "__main__":

    # print(_cut_linestring(LineString([
    #     [0, 1],
    #     [10, 2],
    #     [100, 3]
    # ])))
    #
    # exit()

    data = read_data(INPUT_FILE)

    logger.info(f"data {INPUT_FILE} min: {np.min(data)} / max: {np.max(data)}")

    X, Y, dX, dY, angles, inclination = get_slope(data, 10)

    experiments_table = {
        "hatching_a": standard_hatching,
        "hatching_a_concentric": standard_hatching_concentric,
        "hatching_c": standard_hatching_slope_orientation,
        "hatching_tanaka": illuminated_contours,
        "hatching_flowlines": flowline_hatching,
    }

    land_polys = _extract_polygons(data, *get_elevation_bounds([0, 10_000], 1)[0], True)

    for k, v in experiments_table.items():

        logger.info(f"running: {k}")

        hatchings = v(data, angles=angles)

        doc = DocumentInfo()
        doc.width = data.shape[1]
        doc.height = data.shape[0]

        svg = SvgWriter(Path(OUTPUT_PATH, f"{k}.svg"), [doc.width, doc.height])
        svg.debug = True
        svg.background_color = "white"

        if k == "hatching_tanaka":
            svg.background_color = "grey"

            # options_bright = {
            #     "fill": "none",
            #     "stroke": "white",
            #     "stroke-width": "2.0",
            # }
            #
            # options_dark = {
            #     "fill": "none",
            #     "stroke": "black",
            #     "stroke-width": "2.0",
            # }


            options_bright = {
                "fill": "none",
                "stroke": "skyblue",
                "stroke-width": "2.0",
            }

            options_dark = {
                "fill": "none",
                "stroke": "darkblue",
                "stroke-width": "2.0",
            }

            svg.add("contour_bright", hatchings[0], options=options_bright)
            svg.add("contour_dark", hatchings[1], options=options_dark)

        else:

            options = {
                "fill": "none",
                "stroke": "black",
                "stroke-width": "2.0",
            }

            svg.add("contour", hatchings, options=options)

        options_land = {
            "fill": "green",
            "stroke": "none",
            "fill-opacity": "0.5"
        }

        svg.add("land", land_polys, options=options_land)

        svg.write()
