from pathlib import Path

import cv2
import numpy as np
import pytest
import shapely
from shapely import LineString, Point

from lineworld.util.hersheyfont import HersheyFont, Align, _linestring_to_coordinate_pairs

TEXT = "The quick brown fox jumps over the lazy dog"
FONT_SIZE = 54
CANVAS_DIMENSIONS = [1200, 1200]


@pytest.fixture
def font() -> HersheyFont:
    yield HersheyFont(font_file=Path(".", Path(HersheyFont.DEFAULT_FONT)))


@pytest.fixture
def canvas() -> np.ndarray:
    yield np.full(CANVAS_DIMENSIONS + [3], 255, dtype=np.uint8)


def _draw_linestrings(canvas: np.ndarray, linestrings: list[LineString]):
    for linestring in linestrings:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(canvas, pt1, pt2, (0, 0, 0), 2)


def test_text_straight(font: HersheyFont, canvas: np.ndarray, output_path: Path):
    for i, alignment in enumerate([Align.LEFT, Align.CENTER, Align.RIGHT]):
        offset_y = int(100 + i * 100)
        path = LineString([[100, offset_y], [CANVAS_DIMENSIONS[1] - 100, offset_y]]).segmentize(0.1)
        linestrings = font.lines_for_text(TEXT, FONT_SIZE, path=path, align=alignment)
        _draw_linestrings(canvas, linestrings)

    cv2.imwrite(str(Path(output_path, Path("test_text_straight.png"))), canvas)


def test_text_fontsize(font: HersheyFont, canvas: np.ndarray, output_path: Path):
    offset_y = 0
    for font_size in [1, 10, 20, 25.5, 50, 100]:
        offset_y += int(100 + font_size)
        path = LineString([[100, offset_y], [CANVAS_DIMENSIONS[1] - 100, offset_y]]).segmentize(0.1)
        linestrings = font.lines_for_text(TEXT, font_size, path=path, align=Align.LEFT, reverse_path=False)
        _draw_linestrings(canvas, linestrings)

    cv2.imwrite(str(Path(output_path, Path("test_text_fontsize.png"))), canvas)


def test_text_curved(font: HersheyFont, canvas: np.ndarray, output_path: Path):
    path = shapely.intersection(
        LineString(
            list(
                Point([CANVAS_DIMENSIONS[0] / 2], [CANVAS_DIMENSIONS[1] * 0.7])
                .buffer(CANVAS_DIMENSIONS[0] * 0.6)
                .exterior.coords
            )
        ),
        shapely.box(100, 100, CANVAS_DIMENSIONS[0] - 100, CANVAS_DIMENSIONS[1] - 100),
    )
    path = path.segmentize(1)

    for i, alignment in enumerate([Align.LEFT, Align.CENTER, Align.RIGHT]):
        path = shapely.affinity.translate(path, yoff=200)
        linestrings = font.lines_for_text(TEXT, FONT_SIZE, path=path, align=alignment, reverse_path=True)
        _draw_linestrings(canvas, linestrings)

    cv2.imwrite(str(Path(output_path, Path("test_text_curved.png"))), canvas)
