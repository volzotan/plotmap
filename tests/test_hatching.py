from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely import Point, LineString

from lineworld.core.hatching import create_hatching, HatchingOptions
from lineworld.util.geometrytools import unpack_multilinestring, _linestring_to_coordinate_pairs

CANVAS_DIMENSIONS = [500, 5000]


@pytest.fixture
def canvas() -> np.ndarray:
    yield np.full(CANVAS_DIMENSIONS + [3], 255, dtype=np.uint8)


def _draw_linestrings(canvas: np.ndarray, linestrings: list[LineString]):
    for linestring in linestrings:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(canvas, pt1, pt2, (0, 0, 0), 2)


def test_hatching(canvas: np.ndarray, output_path: Path):
    for i, angle in enumerate([0, 22.5, 45, 90, 135, 180, 270, 360, 360 + 45]):
        g = Point([(i + 1) * CANVAS_DIMENSIONS[0], CANVAS_DIMENSIONS[0] // 2]).buffer(CANVAS_DIMENSIONS[0] * 0.4)

        hatching_options = HatchingOptions()
        hatching_options.angle = angle
        hatching_options.distance = 10

        hatch_lines = create_hatching(g, None, hatching_options)

        _draw_linestrings(canvas, unpack_multilinestring(hatch_lines))

    cv2.imwrite(str(Path(output_path, Path("test_hatching.png"))), canvas)


@pytest.mark.parametrize("distance", [0, 1, 2, 4, 8, 16, 32, 32.5, -1])
def test_hatching_distance(distance: float):
    g = Point([100, 100]).buffer(100)

    hatching_options = HatchingOptions()
    hatching_options.angle = 45
    hatching_options.distance = distance

    hatch_lines = create_hatching(g, None, hatching_options)

    if distance > 0:
        assert len(unpack_multilinestring(hatch_lines)) > 0
    else:
        assert hatch_lines is None


def test_hatching_wiggle(canvas: np.ndarray, output_path: Path):
    for i, wiggle in enumerate([0, 1, 2, 3, 4, 5, 6.5, 10, 100]):
        g = Point([(i + 1) * CANVAS_DIMENSIONS[0], CANVAS_DIMENSIONS[0] // 2]).buffer(CANVAS_DIMENSIONS[0] * 0.4)

        hatching_options = HatchingOptions()
        hatching_options.wiggle = wiggle
        hatching_options.distance = 10

        hatch_lines = create_hatching(g, None, hatching_options)

        _draw_linestrings(canvas, unpack_multilinestring(hatch_lines))

    cv2.imwrite(str(Path(output_path, Path("test_hatching_wiggle.png"))), canvas)
