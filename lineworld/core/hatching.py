import math
from dataclasses import dataclass
from enum import Enum
import random

import numpy as np
import shapely
from shapely.geometry import MultiLineString, LineString, MultiPoint
from shapely import Geometry, transform

from lineworld.util.geometrytools import unpack_multilinestring


class HatchingDirection(Enum):
    ANGLE_45 = 45
    VERTICAL = 0
    HORIZONTAL = 90
    ANGLE_135 = 135


@dataclass
class HatchingOptions():
    angle: float = 45.
    distance: float = 2.0
    direction: HatchingDirection = HatchingDirection.ANGLE_45
    lift: bool = True
    wiggle: float = 0.


def _create_hatch_lines(bbox: list[float], distance: float, direction: HatchingDirection) -> MultiLineString:
    """
    Note: distance is measured along an axis, not distance between parallel hatching lines
    (if hatching is done at an angle, for example 45°)
    """

    minx, miny, maxx, maxy = bbox

    minx = distance * math.floor(minx / distance)
    miny = distance * math.floor(miny / distance)
    maxx = distance * math.ceil(maxx / distance)
    maxy = distance * math.ceil(maxy / distance)

    num_x = int(round(maxx - minx) / distance)
    num_y = int(round(maxy - miny) / distance)

    # lines = np.empty([num_x + num_y + 2], dtype=Geometry)
    lines = []
    offset = -(maxx - minx)

    match direction:
        case HatchingDirection.ANGLE_45:
            for i in range(num_x + num_y + 1):
                start = [minx, miny + i * distance]
                end = [maxx, start[1] + offset]
                lines.append(LineString([start, end]))
                # lines[i] = LineString([start, end])

        case HatchingDirection.ANGLE_135:
            for i in range(num_x + num_y + 1):
                start = [maxx, miny + i * distance]
                end = [minx, start[1] + offset]
                lines.append(LineString([start, end]))
                # lines[i] = LineString([start, end])

        case _:
            raise Exception(f"unknown hatching direction: {direction}")

    return lines


def _combine(g: Geometry, hatch_lines: MultiLineString) -> MultiLineString:
    res = g.intersection(hatch_lines)
    lines = np.array(unpack_multilinestring(res))
    lines = lines[~shapely.is_empty(lines)]
    return MultiLineString(lines.tolist())

def _randomize(g: Geometry) -> Geometry:
    def random_transform(x):
        rng = np.random.default_rng()
        rng.standard_normal(x.shape)
        return x + rng.standard_normal(x.shape)/4

    return transform(g, random_transform)

def _segmentize(g: Geometry) -> Geometry:
    return shapely.segmentize(g, 25)

def create_hatching(g: Geometry, bbox: list[float] | None, hatching_options: HatchingOptions) -> MultiLineString | None:

    # if no bbox is supplied (ie. by using ST_Envelope in PostGIS),
    # we'll compute our own (may be slow)
    if bbox is None:
        bp = MultiPoint(g.exterior.coords).envelope
        bbox = [*bp.exterior.coords[0], *bp.exterior.coords[2]]

    hatch_lines = _create_hatch_lines(bbox, hatching_options.distance, hatching_options.direction)

    # sg = shapely.simplify(g, hatching_options.distance/2)
    sg = g #g.buffer(1)

    if shapely.is_empty(sg):
        return None

    if shapely.is_valid(sg):
        sg = shapely.make_valid(sg)

    # return _combine(sg, hatch_lines)
    return _randomize(_segmentize(_combine(sg, hatch_lines)))
