from pathlib import Path
import cv2
import numpy as np
from lineworld.core.maptools import DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds
from lineworld.util.geometrytools import unpack_multipolygon

import shapely

from shapelysmooth import taubin_smooth

# INPUT_FILE = Path("data/hatching_dem.tif")
# INPUT_FILE = Path("data/slope_test_2.tif")
# INPUT_FILE = Path("data/slope_test_4.tif")

INPUT_FILE = Path("experiments/hatching/data/gebco_crop.tif")

OUTPUT_PATH = Path("experiments/hatching/output")

LEVELS = 10
DISTANCES = [3.0 + x * 0.9 for x in range(LEVELS)]
BOUNDS = get_elevation_bounds([0, -5700], LEVELS)

MIN_AREA = 10
SEGMENT_MAX_LENGTH = 10
SIMPLIFY_TOLERANCE = 1.0


def _read_data(input_path: Path) -> np.ndarray:
    data = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    # data = cv2.resize(img, [30, 30])

    # data = np.flipud(data)
    # data = (data * 120/20).astype(np.int8)
    # data = np.rot90(data)

    return data


if __name__ == "__main__":
    data = _read_data(INPUT_FILE)

    print(f"data {INPUT_FILE} min: {np.min(data)} / max: {np.max(data)}")

    doc = DocumentInfo()
    doc.width = data.shape[1]
    doc.height = data.shape[0]

    output = []

    for i in range(LEVELS):
        extracted_geometries = _extract_polygons(data, *BOUNDS[i], True)

        polygons = []
        for g in extracted_geometries:
            polygons += unpack_multipolygon(g)

        output += polygons

    for i in range(len(output)):
        output[i] = shapely.segmentize(output[i], SEGMENT_MAX_LENGTH)

    output_filename = "smoothing"

    svg = SvgWriter(Path(OUTPUT_PATH, f"{output_filename}.svg"), [doc.width, doc.height])
    svg.background_color = "white"

    options = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "2.0",
        "opacity": "0.5",
    }

    svg.add("original", output, options=options)

    # ----------

    output_taubin = []
    for poly in output:
        p = taubin_smooth(poly, steps=5, factor=0.7, mu=-0.2)
        p = shapely.simplify(p, SIMPLIFY_TOLERANCE)
        if p.area < MIN_AREA:
            continue
        output_taubin.append(p)

    options_taubin = {
        "fill": "none",
        "stroke": "red",
        "stroke-width": "2.0",
        "opacity": "0.5",
    }

    svg.add("taubin_5", output_taubin, options=options_taubin)

    # ----------

    output_taubin = []
    for poly in output:
        p = taubin_smooth(poly, steps=30)
        if p.area < MIN_AREA:
            continue
        output_taubin.append(p)

    options_taubin = {
        "fill": "none",
        "stroke": "green",
        "stroke-width": "2.0",
        "opacity": "0.5",
    }

    svg.add("taubin_30", output_taubin, options=options_taubin)

    # ----------

    output_taubin = []
    for poly in output:
        p = taubin_smooth(poly, steps=100)
        if p.area < MIN_AREA:
            continue
        output_taubin.append(p)

    options_taubin = {
        "fill": "none",
        "stroke": "blue",
        "stroke-width": "2.0",
        "opacity": "0.5",
    }

    svg.add("taubin_100", output_taubin, options=options_taubin)

    # ----------

    # output_chaikin = []
    # for poly in output:
    #     output_chaikin.append(chaikin_smooth(poly, iters=5))
    #
    # options_chaikin = {
    #     "fill": "none",
    #     "stroke": "red",
    #     "stroke-width": "2.0",
    #     "opacity": "0.5"
    # }
    #
    # svg.add("chaikin_5", output_chaikin, options=options_chaikin)

    # ----------

    svg.write()
