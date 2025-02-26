import csv
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import transform
from shapely import Point, LineString, MultiLineString, STRtree
from shapely.affinity import affine_transform, translate

from lineworld.core import maptools
from lineworld.core.maptools import Projection
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.hersheyfont import HersheyFont, Align

from loguru import logger

INPUT_FILE = Path("experiments/labelplacement/cities.csv")
OUTPUT_PATH = Path("experiments/labelplacement/output")

DOCUMENT_SIZE = [1000, 1000]

FONT_SIZE = 5
OFFSET_FROM_CENTER = 5
CIRCLE_RADIUS = 1.5
BOX_SAFETY_MARGIN = 1.0

FILTER_MIN_POPULATION = 1_000_000

cities = []

debug_map_circles = []
debug_map_labels = []

document_info = maptools.DocumentInfo({})
project_func = document_info.get_projection_func(Projection.WGS84)
mat = document_info.get_transformation_matrix()

font = HersheyFont()

positions = {
    "top-right": {"pos": 315, "align": Align.LEFT, "error": 0},
    "top-left": {"pos": 225, "align": Align.RIGHT, "error": 1},
    "bottom-right": {"pos": 45, "align": Align.LEFT, "error": 2},
    "bottom-left": {"pos": 135, "align": Align.RIGHT, "error": 3},
    "center-right": {"pos": 0, "align": Align.LEFT, "error": 4},
    "center-top": {"pos": 270, "align": Align.CENTER, "error": 5},
    "center-left": {"pos": 180, "align": Align.RIGHT, "error": 6},
    "center-bottom": {"pos": 90, "align": Align.CENTER, "error": 7},
}

position_errors = [positions[key]["error"] for key in positions.keys()]

with open(INPUT_FILE) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    for row in reader:
        population = int(row["population"])
        if population < FILTER_MIN_POPULATION:
            continue

        lon = float(row["lon"])
        lat = float(row["lat"])
        label = row["ascii_name"]

        p = Point([lon, lat])

        p = transform(project_func, p)
        p = affine_transform(p, mat)

        circle = p.buffer(CIRCLE_RADIUS)
        debug_map_circles.append(circle)

        positions_boxes = []
        positions_text = []
        for k in positions.keys():
            sin = math.sin(math.radians(positions[k]["pos"]))
            cos = math.cos(math.radians(positions[k]["pos"]))

            xnew = OFFSET_FROM_CENTER * cos - 0 * sin + p.x
            ynew = OFFSET_FROM_CENTER * sin + 0 * cos + p.y

            # TODO:
            #  hard: backproject xnew, ynew to lat lon so we compute the baseline path for the font
            #  easy: compute the baseline path only once for right and left and move it up and down to the different positions

            path_coords = None
            match positions[k]["align"]:
                case Align.LEFT:
                    path_coords = [[lon, lat], [lon + 50, lat]]
                case Align.RIGHT:
                    path_coords = [[lon - 50, lat], [lon, lat]]
                case Align.CENTER:
                    path_coords = [[lon - 25, lat], [lon + 25, lat]]
                case _:
                    raise Exception(f"unexpected enum state align: {positions[k]["align"]}")

            path = LineString(path_coords).segmentize(0.1)
            path = transform(project_func, path)
            path = affine_transform(path, mat)
            path = translate(path, xoff=xnew - p.x, yoff=ynew - p.y)
            lines = MultiLineString(
                font.lines_for_text(label, FONT_SIZE, align=positions[k]["align"], center_vertical=True, path=path)
            )

            positions_text.append(lines)

            box = lines.envelope.buffer(BOX_SAFETY_MARGIN)
            positions_boxes.append(box.envelope)

            # labels += [path, lines, box.exterior]
            debug_map_labels += [box.exterior]

        cities.append(
            {
                "pos": p,
                "label": label,
                "population": population,
                "priority": 1.0,
                "error": None,
                "circle": circle,
                "boxes": positions_boxes,
                "text": positions_text,
                "region": None,
            }
        )

        if len(cities) > 100:
            break

min_pop = FILTER_MIN_POPULATION
max_pop = max([c["population"] for c in cities])

for c in cities:
    c["priority"] = (c["population"] - min_pop) / (max_pop - min_pop)

# collision check

cities_cleaned = []
cities = list(reversed(sorted(cities, key=lambda c: c["population"])))
tree = STRtree([c["pos"].buffer(CIRCLE_RADIUS * 2.5).envelope for c in cities])
for i, c in enumerate(cities):
    collisions = tree.query(c["pos"])
    if min(collisions) < i:
        # print(cities[min(collisions)]["label"], c["label"])
        continue
    cities_cleaned.append(c)

logger.info(f"removed during collision checking: {len(cities)-len(cities_cleaned)}")

cities = cities_cleaned

# split cities into disjunct sets

label_polygons = [shapely.ops.unary_union(c["boxes"]) for c in cities]
tree = STRtree(label_polygons)

regions = []


def rec_propagate(city_index: int, region: int) -> None:
    if cities[city_index]["region"] is not None:
        return

    cities[city_index]["region"] = region
    regions[region].append(city_index)
    for overlap_index in tree.query(label_polygons[city_index]):
        rec_propagate(int(overlap_index), region)


for i in range(len(cities)):
    c = cities[i]

    if c["region"] is not None:
        continue

    region_name = len(regions)
    regions.append([])

    rec_propagate(i, region_name)

# annealing

state = np.zeros([len(cities)], dtype=int)


def fitness(cities_indices: list[int], s: np.ndarray) -> float:
    tree_box = STRtree([cities[ci]["boxes"][s[i]] for i, ci in enumerate(cities_indices)])
    tree_circle = STRtree([cities[ci]["circle"] for i, ci in enumerate(cities_indices)])

    fit = 0

    for i, ci in enumerate(cities_indices):
        overlaps = tree_box.query(cities[ci]["boxes"][s[i]])
        fit += (len(overlaps) - 1) * 100

        overlaps = tree_circle.query(cities[ci]["boxes"][s[i]])
        fit += len(overlaps) * 100

        fit += position_errors[s[i]]

        fit += fit * cities[ci]["priority"]

    return fit


def anneal(region: list[int]) -> list[int]:
    state = np.zeros([len(region)], dtype=int)
    new_fit = np.zeros([len(region)], dtype=int)
    fit = len(region) * 600

    for i in range(100000):
        s = np.copy(state)
        new_fit = np.zeros([len(region)], dtype=float)

        for i in range(s.shape[0]):
            s[i] = random.randrange(7)

        tree_box = STRtree([cities[ci]["boxes"][s[i]] for i, ci in enumerate(region)])
        tree_circle = STRtree([cities[ci]["circle"] for i, ci in enumerate(region)])

        for i, ci in enumerate(region):
            overlaps = tree_box.query(cities[ci]["boxes"][s[i]])
            new_fit[i] += (len(overlaps) - 1) * 100

            overlaps = tree_circle.query(cities[ci]["boxes"][s[i]])
            new_fit[i] += len(overlaps) * 100

            new_fit[i] += position_errors[s[i]]
            new_fit[i] += new_fit[i] * cities[ci]["priority"]

        if np.sum(new_fit) < fit:
            state = s
            fit = np.sum(new_fit)

            for i, ci in enumerate(region):
                cities[ci]["error"] = new_fit[i]

        if fit < 10:
            break

    logger.debug(f"region (size {len(region)}) fitness {fit:6.2f}")

    return state.tolist()


for region in regions:
    region_state = anneal(region)

    for i in range(len(region_state)):
        state[region[i]] = region_state[i]

    break

# debug

for c in cities:
    if c["error"] is None:
        continue
    print(f"{c["label"]} {c["error"]:>10.2f}")

# output

svg = SvgWriter(Path(OUTPUT_PATH, "labelplacement.svg"), DOCUMENT_SIZE)
options = {"fill": "none", "stroke": "black", "stroke-width": "0.2"}
svg.add("circles", [c["pos"].buffer(CIRCLE_RADIUS) for c in cities], options=options)

placed_labels = []
for i, c in enumerate(cities):
    if c["error"] is None:
        continue
    placed_labels.append(c["text"][state[i]])

svg.add("labels", placed_labels, options=options)
svg.write()

# debug

svg = SvgWriter(Path(OUTPUT_PATH, "labelplacement_debug.svg"), DOCUMENT_SIZE)
options = {"fill": "none", "stroke": "black", "stroke-width": "0.2"}
svg.add("circles", debug_map_circles, options=options)
svg.add("labels", debug_map_labels, options=options)
svg.write()
