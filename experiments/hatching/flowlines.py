import datetime
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import shapely
from loguru import logger
from matplotlib import pyplot as plt
from shapely import Point, LineString

from experiments.hatching.slope import get_slope
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.gebco_grid_to_polygon import _extract_polygons, get_elevation_bounds

INPUT_FILE = Path("experiments/hatching/data/slope_test_3.tif")
INPUT_FILE = Path("experiments/hatching/data/gebco_crop.tif")

OUTPUT_PATH = Path("experiments/hatching/output")

LINE_DISTANCE = [4, 28]  # distance between lines
LINE_STEP_DISTANCE = 2.0  # distance between points constituting a line
MAX_ANGLE_DISCONTINUITY = 1.0  # max difference (in radians) in slope between line points
MIN_INCLINATION = 0.1  # 50.0

SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS = 10  # How many line segments should be skipped before the next seedpoint is extracted
LINE_MAX_SEGMENTS = 100

bounding_box = shapely.box(0, 0, 999, 999)
# bounding_box = Point([500, 500]).buffer(450)

density_map = np.full([1000, 1000], LINE_DISTANCE[0], dtype=float)

point_map = {}
for x in range(1000):
    for y in range(1000):
        point_map[f"{x},{y}"] = []

point_raster = np.zeros([1000, 1000], dtype=bool)


def _collision_approximate(x: float, y: float, density_map: np.ndarray) -> bool:
    x = round(x)
    y = round(y)
    half_d = round(density_map[y, x] / 2)

    return np.any(point_raster[
                  max(y - half_d, 0):min(y + half_d, point_raster.shape[0]),
                  max(x - half_d, 0):min(x + half_d, point_raster.shape[1])
                  ])


def _collision_precise(x: float, y: float, density_map: np.ndarray) -> bool:
    x = round(x)
    y = round(y)
    d = round(density_map[y, x])
    half_d = round(d / 2)

    x_minmax = [max(x - half_d, 0), min(x + half_d, point_raster.shape[1])]
    y_minmax = [max(y - half_d, 0), min(y + half_d, point_raster.shape[0])]

    for ix in range(*x_minmax):
        for iy in range(*y_minmax):
            for p in point_map[f"{ix},{iy}"]:
                if math.sqrt((p[0]-x)**2 + (p[1]-y)**2) < d:
                    return True

    return False

def _collision(x: float, y: float, density_map: np.ndarray) -> bool:
    return _collision_precise(x, y, density_map)
    # return _collision_approximate(x, y, density_map)

def _next_point(x1: float, y1: float, angles: np.ndarray, inclination: np.ndarray, forwards: bool) -> float:
    a1 = angles[int(y1), int(x1)]
    inc = inclination[int(y1), int(x1)]

    if abs(inc) < MIN_INCLINATION:
        return None

    dir = 1
    if not forwards:
        dir = -1

    x2 = x1 + LINE_STEP_DISTANCE * math.cos(a1) * dir
    y2 = y1 + LINE_STEP_DISTANCE * math.sin(a1) * dir

    if not bounding_box.contains(Point([x2, y2])):
        return None

    if _collision(x2, y2, density_map):
        return None

    a2 = angles[round(y2), round(x2)]

    if abs(a2 - a1) > MAX_ANGLE_DISCONTINUITY:
        return None

    return (x2, y2)


def _seed_points(line_points: list[tuple[float, float]], density_map: np.ndarray) -> list[tuple[float, float]]:
    num_seedpoints = 1
    seed_points = []

    if len(line_points) > SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS:
        num_seedpoints = (len(line_points) - 1) // SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS

    for i in range(num_seedpoints):
        x1, y1 = line_points[i * SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS]
        x2, y2 = line_points[i * SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS + 1]

        # midpoint
        x3 = x1 + (x2 - x1) / 2.0
        y3 = y1 + (y2 - y1) / 2.0

        a1 = math.atan2(y1 - y3, x1 - x3)

        a2 = a1

        if i % 2 == 0:
            a2 += math.radians(90)
        else:
            a2 -= math.radians(90)

        x4 = density_map[round(y3), round(x3)]
        y4 = 0

        x5 = x4 * math.cos(a2) - y4 * math.sin(a2) + x3
        y5 = x4 * math.sin(a2) + y4 * math.cos(a2) + y3

        if x5 < 0 or x5 > 999 or y5 < 0 or y5 > 999:
            continue

        seed_points.append([x5, y5])

    return seed_points


if __name__ == "__main__":

    # sanity checks:

    # if not (LINE_STEP_DISTANCE < LINE_DISTANCE[0]):
    #     raise Exception("distance between points of a line must be smaller than the distance between lines")

    data = cv2.imread(str(INPUT_FILE), cv2.IMREAD_UNCHANGED)

    if not data.shape == [1000, 1000]:
        data = cv2.resize(data, [1000, 1000])

    logger.debug(f"data {INPUT_FILE} min: {np.min(data)} | max: {np.max(data)}")

    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    density_map = density_map + (data_normalized * (LINE_DISTANCE[1] - LINE_DISTANCE[0]))

    X, Y, dX, dY, angles, inclination = get_slope(data, 10)

    angles_processed = cv2.blur(angles, (30, 30))
    # angles_processed = cv2.GaussianBlur(angles,(11, 11), 0)

    output = np.zeros([1000, 1000, 3], dtype=np.uint8)

    timer = datetime.datetime.now()

    linestrings = []
    starting_points = deque()

    for i in np.linspace(0 + 1, 1000 - 1, num=100):
        for j in np.linspace(0 + 1, 1000 - 1, num=100):
            starting_points.append([i, j])

    while len(starting_points) > 0:

        seed = starting_points.popleft()

        if _collision(*seed, density_map):
            continue

        line_points = deque([seed])

        # follow gradient up
        for i in range(10000):

            if LINE_MAX_SEGMENTS > 0 and len(line_points) >= LINE_MAX_SEGMENTS:
                break

            p = _next_point(*line_points[-1], angles_processed, inclination, True)

            if p is None:
                break

            line_points.append(p)

        # follow gradient down
        for i in range(10000):

            if LINE_MAX_SEGMENTS > 0 and len(line_points) >= LINE_MAX_SEGMENTS:
                break

            p = _next_point(*line_points[0], angles_processed, inclination, False)

            if p is None:
                break

            line_points.appendleft(p)

        if len(line_points) < 2:
            continue

        linestrings.append(LineString(line_points))

        # seed points
        seed_points = _seed_points(line_points, density_map)
        starting_points.extendleft(seed_points)

        # collision checks
        for lp in line_points:
            x = round(lp[0])
            y = round(lp[1])
            point_map[f"{x},{y}"].append(lp)
            point_raster[y, x] = True

        # viz
        cv2.circle(output, (round(seed[0]), round(seed[1])), 2, (255, 0, 0), -1)
        for i in range(len(line_points) - 1):
            start = (round(line_points[i][0]), round(line_points[i][1]))
            end = (round(line_points[i + 1][0]), round(line_points[i + 1][1]))
            cv2.line(output, start, end, (255, 255, 255), 2)

    total_time = (datetime.datetime.now() - timer).total_seconds()
    avg_line_length = sum([x.length for x in linestrings]) / len(linestrings)

    logger.info(f"total time:         {total_time:5.2f}s")
    logger.info(f"avg line length:    {avg_line_length:5.2f}")

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(data)
    ax2.imshow(angles, interpolation="none")
    ax3.imshow(inclination)
    ax4.imshow(point_raster)
    ax5.imshow(angles_processed, interpolation="none")
    ax6.imshow(output)

    fig.set_figheight(12)
    fig.set_figwidth(20)
    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    plt.savefig(Path(OUTPUT_PATH, "flowlines.png"))

    svg = SvgWriter(Path(OUTPUT_PATH, "flowlines.svg"), [1000, 1000])

    options = {
        "fill": "none",
        "stroke": "black",
        "stroke-width": "2"
    }

    svg.add("flowlines", linestrings, options=options)

    land_polys = _extract_polygons(data, *get_elevation_bounds([0, 10_000], 1)[0], True)

    options_land = {
        "fill": "green",
        "stroke": "none",
        "fill-opacity": "0.5"
    }

    svg.add("land", land_polys, options=options_land)

    svg.write()
