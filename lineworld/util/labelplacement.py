import csv
import datetime
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.ops import transform
from shapely import Point, LineString, MultiLineString, STRtree, Polygon
from shapely.affinity import affine_transform, translate

import lineworld
from lineworld.core import map
from lineworld.core.map import Projection, DocumentInfo
from lineworld.core.svgwriter import SvgWriter
from lineworld.util.hersheyfont import HersheyFont, Align

from loguru import logger

DEFAULT_MAX_ITERATIONS = 10_000

DEFAULT_FONT_SIZE = 5
DEFAULT_OFFSET_FROM_CENTER = 5
DEFAULT_CIRCLE_RADIUS = 1.5
DEFAULT_BOX_SAFETY_MARGIN = 1.0

DEFAULT_FILTER_MIN_POPULATION = 1_000_000

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


@dataclass
class City:
    pos: Point
    label: str
    population: int
    priority: float
    error: float | None
    circle: Polygon
    boxes: list[Polygon]
    text: list[MultiLineString]
    region: int | None
    placement: int | None  # position index of the best label placement


def _anneal(cities: list[City], region: list[int], config: dict[str, Any]):
    # TODO: we're not exactly doing simulated annealing here because only better states are accepted,
    #  not marginally worse ones based on current temperature

    state = np.zeros([len(region)], dtype=int)
    for i in range(state.shape[0]):
        state[i] = random.randrange(8)

    error = np.full([len(region)], 1000, dtype=float)
    new_error = np.full([len(region)], 0, dtype=float)

    tree_circle = STRtree([cities[ci].circle for i, ci in enumerate(region)])

    position_errors = [positions[key]["error"] for key in positions.keys()]

    for _ in range(config.get("max_iterations", DEFAULT_MAX_ITERATIONS)):
        s = np.copy(state)
        new_error.fill(0)

        s[random.randrange(s.shape[0])] = random.randrange(8)

        tree_box = STRtree([cities[ci].boxes[s[i]] for i, ci in enumerate(region)])

        for i, ci in enumerate(region):
            overlaps = tree_box.query(cities[ci].boxes[s[i]])
            new_error[i] += (len(overlaps) - 1) * 100

            overlaps = tree_circle.query(cities[ci].boxes[s[i]])
            new_error[i] += len(overlaps) * 100

            new_error[i] += position_errors[s[i]]
            new_error[i] += new_error[i] * cities[ci].priority

        if np.sum(new_error) < np.sum(error):
            state = s
            error = new_error

        if np.sum(error) < 1:
            break

    logger.debug(f"region (size {len(region):<3}) error {np.sum(error):6.2f} | {[cities[ci].label for ci in region]}")

    for i, ci in enumerate(region):
        cities[ci].error = error[i]
        cities[ci].placement = state[i]


def read_from_file(filename: Path, document_info: DocumentInfo, config: dict[str, Any]) -> list[City]:
    cities = []

    debug_map_circles = []
    debug_map_labels = []

    project_func = document_info.get_projection_func(Projection.WGS84)
    mat = document_info.get_transformation_matrix()

    font = HersheyFont()
    filter_min_population = config.get("filter_min_population", DEFAULT_FILTER_MIN_POPULATION)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            population = int(row["population"])
            if population < filter_min_population:
                continue

            lon = float(row["lon"])
            lat = float(row["lat"])
            label = row["ascii_name"]
            font_size = config.get("font_size", DEFAULT_FONT_SIZE)

            p = Point([lon, lat])

            p = transform(project_func, p)
            p = affine_transform(p, mat)

            circle = p.buffer(config.get("circle_radius", DEFAULT_CIRCLE_RADIUS))
            debug_map_circles.append(circle)

            positions_boxes = []
            positions_text = []
            for k in positions.keys():
                sin = math.sin(math.radians(positions[k]["pos"]))
                cos = math.cos(math.radians(positions[k]["pos"]))

                offset = config.get("offset_from_center", DEFAULT_OFFSET_FROM_CENTER)

                xnew = offset * cos - 0 * sin + p.x
                ynew = offset * sin + 0 * cos + p.y

                # TODO:
                #  complex: backproject xnew, ynew to lat lon so we compute the baseline path for the font
                #  simple: compute the baseline path only once for right and left and move it up and down to the different positions

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
                    font.lines_for_text(label, font_size, align=positions[k]["align"], center_vertical=True, path=path)
                )

                positions_text.append(lines)

                box = lines.envelope.buffer(config.get("box_safety_margin", DEFAULT_BOX_SAFETY_MARGIN))
                positions_boxes.append(box.envelope)

                debug_map_labels += [box.exterior]

            cities.append(
                City(
                    pos=p,
                    label=label,
                    population=population,
                    priority=1.0,
                    error=None,
                    circle=circle,
                    boxes=positions_boxes,
                    text=positions_text,
                    region=None,
                    placement=None,
                )
            )

    return cities


def generate_placement(cities: list[City], config: dict[str, Any]) -> list[City]:
    min_pop = config.get("filter_min_population", DEFAULT_FILTER_MIN_POPULATION)
    max_pop = max([c.population for c in cities])

    for c in cities:
        c.priority = (c.population - min_pop) / (max_pop - min_pop)

    # collision check

    cities_cleaned = []
    cities = list(reversed(sorted(cities, key=lambda c: c.population)))
    tree = STRtree([c.pos.buffer(config.get("circle_radius", DEFAULT_CIRCLE_RADIUS) * 2.5).envelope for c in cities])
    for i, c in enumerate(cities):
        collisions = tree.query(c.pos)
        if min(collisions) < i:
            continue
        cities_cleaned.append(c)

    logger.info(f"removed during collision checking: {len(cities)-len(cities_cleaned)}")

    cities = cities_cleaned

    # split cities into disjunct sets

    label_polygons = [shapely.ops.unary_union(c.boxes) for c in cities]
    tree = STRtree(label_polygons)

    regions = []

    def _rec_propagate(city_index: int, region: int) -> None:
        if cities[city_index].region is not None:
            return

        cities[city_index].region = region
        regions[region].append(city_index)
        for overlap_index in tree.query(label_polygons[city_index]):
            _rec_propagate(int(overlap_index), region)

    for i in range(len(cities)):
        c = cities[i]

        if c.region is not None:
            continue

        region_name = len(regions)
        regions.append([])

        _rec_propagate(i, region_name)

    # annealing

    timer_start = datetime.datetime.now()
    for region in regions:
        _anneal(cities, region, config)

    logger.info(f"anneal total time: {(datetime.datetime.now()-timer_start).total_seconds():5.2f}")

    # remove collisions

    cities_cleaned = []
    for c in cities:
        tree_box = STRtree([cc.boxes[cc.placement] for cc in cities_cleaned])
        tree_circle = STRtree([cc.circle for cc in cities_cleaned])

        box = c.boxes[c.placement]
        circle = c.circle
        overlap = tree_box.query(box).tolist() + tree_box.query(circle).tolist() + tree_circle.query(box).tolist()

        if len(overlap) == 0:
            cities_cleaned.append(c)
        else:
            logger.debug(f"drop city: {c.label}")

    cities = cities_cleaned

    ## debug
    # for c in cities:
    #     if c["error"] is None:
    #         continue
    #     print(f"{c["label"]} {c["error"]:>10.2f}")

    return cities


if __name__ == "__main__":
    INPUT_FILE = Path("data/cities2/cities.csv")
    OUTPUT_PATH = Path("experiments/labelplacement/output")

    config = lineworld.get_config()
    document_info = maptools.DocumentInfo(config)

    cities = read_from_file(INPUT_FILE, document_info, config)
    cities = generate_placement(cities, config)

    svg = SvgWriter(Path(OUTPUT_PATH, "labelplacement.svg"), document_info.get_document_size())
    options = {"fill": "none", "stroke": "black", "stroke-width": "0.2"}
    svg.add(
        "circles", [c.pos.buffer(config.get("circle_radius", DEFAULT_CIRCLE_RADIUS)) for c in cities], options=options
    )

    placed_labels = []
    for i, c in enumerate(cities):
        if c.error is None:
            continue
        placed_labels.append(c.text[c.placement])

    svg.add("labels", placed_labels, options=options)
    svg.write()

    # debug

    # svg = SvgWriter(Path(OUTPUT_PATH, "labelplacement_debug.svg"), DOCUMENT_SIZE)
    # options = {"fill": "none", "stroke": "black", "stroke-width": "0.2"}
    # svg.add("circles", debug_map_circles, options=options)
    # svg.add("labels", debug_map_labels, options=options)
    # svg.write()
