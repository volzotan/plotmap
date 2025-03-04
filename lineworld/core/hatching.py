import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import shapely
from shapely.geometry import MultiLineString, LineString, MultiPoint
from shapely import Geometry, transform, affinity

from lineworld.util.geometrytools import unpack_multilinestring


class HatchingDirection(Enum):
    ANGLE_45 = 45
    VERTICAL = 0
    HORIZONTAL = 90
    ANGLE_135 = 135


@dataclass
class HatchingOptions:
    angle: float = 45.0
    distance: float = 2.0
    lift: bool = True  # TODO: currently unimplemented
    wiggle: float = 0.0  # TODO: currently unimplemented


def _create_hatch_lines(bbox: list[float], distance: float, angle: float) -> MultiLineString | None:
    """
    Note: distance is measured along an axis, not distance between parallel hatching lines
    (if hatching is done at an angle, for example 45°)
    """

    if distance <= 0:
        return None

    minx, miny, maxx, maxy = bbox

    minx = distance * math.floor(minx / distance)
    miny = distance * math.floor(miny / distance)
    maxx = distance * math.ceil(maxx / distance)
    maxy = distance * math.ceil(maxy / distance)

    diag = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

    num = round(diag // distance)

    lines = []

    for i in range(num):
        offset = (distance * i) - diag / 2
        start = [-diag, offset]
        end = [+diag, offset]
        lines.append(LineString([start, end]))

    mls = MultiLineString(lines)

    mls = affinity.rotate(mls, angle)
    mls = affinity.translate(mls, xoff=minx + (maxx - minx) / 2, yoff=miny + (maxy - miny) / 2)

    return mls


def _combine(g: Geometry, hatch_lines: MultiLineString) -> MultiLineString:
    res = g.intersection(hatch_lines)
    lines = np.array(unpack_multilinestring(res))
    lines = lines[~shapely.is_empty(lines)]
    return MultiLineString(lines.tolist())


def _randomize(g: Geometry) -> Geometry:
    def random_transform(x):
        rng = np.random.default_rng()
        rng.standard_normal(x.shape)
        return x + rng.standard_normal(x.shape) / 4

    return transform(g, random_transform)


def create_hatching(g: Geometry, bbox: list[float] | None, hatching_options: HatchingOptions) -> MultiLineString | None:
    # if no bbox is supplied (ie. by using ST_Envelope in PostGIS),
    # we'll compute our own (may be slow)
    if bbox is None:
        bp = MultiPoint(g.exterior.coords).envelope
        bbox = [*bp.exterior.coords[0], *bp.exterior.coords[2]]

    # hatch_lines = _create_hatch_lines(bbox, hatching_options.distance, hatching_options.direction)
    hatch_lines = _create_hatch_lines(bbox, hatching_options.distance, hatching_options.angle)

    # sg = shapely.simplify(g, hatching_options.distance/2)
    sg = g  # g.buffer(1)

    if shapely.is_empty(sg):
        return None

    if shapely.is_valid(sg):
        sg = shapely.make_valid(sg)

    if hatch_lines is None:
        return None

    return _combine(sg, hatch_lines)
    # return _randomize(_segmentize(_combine(sg, hatch_lines)))
